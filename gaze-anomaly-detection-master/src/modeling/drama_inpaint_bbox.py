import json

import torch
import torchvision
from fairscale.nn.misc import checkpoint_wrapper
import random
from torch import nn
from src.modeling.load_bbox_pred_head import get_bbox_pred_model, get_bbox_pred_model_inpaint, generalized_box_iou_loss
from src.utils.matcher import build_matcher

class INPAINTDRAMAVideoTransformer(torch.nn.Module):
    def __init__(self, args, config, swin, transformer_encoder):
        super(INPAINTDRAMAVideoTransformer, self).__init__()
        self.config = config

        # get backbone
        self.use_checkpoint = args.use_checkpoint and not args.freeze_backbone
        if self.use_checkpoint:
            self.swin = checkpoint_wrapper(swin, offload_to_cpu=True)
        else:
            self.swin = swin

        # define the inter dimention of image/video features in decoder
        self.use_grid_feat = args.grid_feat
        self.latent_feat_size = self.swin.backbone.norm.normalized_shape[0]
        self.img_feature_dim = int(args.img_feature_dim)

        # transform the dimension of video features
        self.fc = torch.nn.Linear(self.latent_feat_size, self.img_feature_dim)

        # should be ture in risk localization
        self.pred_bbox = getattr(args, 'pred_bbox', False)

        # different bbox heads to choose            
        self.bbox_head = get_bbox_pred_model_inpaint(args)

        # use a sample matcher to 
        self.match = build_matcher(set_cost_class=1, set_cost_giou=2)
        self.match_weight = args.match_weight
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.num_queries = 2

    def forward(self, *args, **kwargs):
        images = kwargs['img_feats']
        B, S, C, H, W = images.shape  # batch, segment, chanel, hight, width
        # (B x S x C x H x W) --> (B x C x S x H x W)
        images = images.permute(0, 2, 1, 3, 4)

        # backbone
        vid_feats = self.swin(images)
        if self.use_grid_feat==True:
            vid_feats = vid_feats.permute(0, 2, 3, 4, 1)
        vid_feats = vid_feats.view(B, -1, self.latent_feat_size)
        vid_feats = self.fc(vid_feats)

        # bbox head
        bbox_preds, bbox_preds_2, cls_pred = self.bbox_head(vid_feats)
        pred_bboxs = torch.stack([bbox_preds, bbox_preds_2], dim=0).permute(1, 0, 2)
        target_bboxs = torch.stack([kwargs['bbox'], kwargs['inpaint_bboxes']], dim=0).permute(1, 0, 2)

        # # matcher
        # indices = self.match(pred_bboxs, target_bboxs)

        # # loss
        # loss_func1 = nn.L1Loss()
        # loss_func2 = generalized_box_iou_loss

        # level_src_boxes = []
        # level_tgt_boxes = []
        # level_losses = []
        # level_ious = []

        # for level_id in range(self.num_queries):
        #     batch_idx = []
        #     src_idx = []
        #     tgt_idx = []
        #     for batch, (src, tgt) in enumerate(indices):
        #         batch_idx.append(batch)
        #         src_idx.append(src[level_id])
        #         tgt_idx.append(tgt[level_id])
        #     src_idx = (torch.tensor(batch_idx), torch.tensor(src_idx))
        #     tgt_idx = (torch.tensor(batch_idx), torch.tensor(tgt_idx))

        #     src_boxes = pred_bboxs[src_idx]
        #     tgt_boxes = target_bboxs[tgt_idx]

        #     with torch.no_grad():
        #         bbox_iou = self.get_iou(src_boxes, tgt_boxes)

        #     level_src_boxes.append(src_boxes)
        #     level_tgt_boxes.append(tgt_boxes)
        #     level_losses.append(loss_func1(src_boxes, tgt_boxes) + loss_func2(src_boxes, tgt_boxes))
        #     level_ious.append(bbox_iou)


        # level_src_boxes = torch.stack(level_src_boxes, dim=1)
        # level_tgt_boxes = torch.stack(level_tgt_boxes, dim=1)
        # level_losses = torch.stack(level_losses, dim=-1)
        # level_ious = torch.stack(level_ious, dim=1)

        # FIXME: do not user matcher to match these two results, 
        # instead, directly specify the correspondence between the two
        loss_func1 = nn.L1Loss()
        loss_func2 = generalized_box_iou_loss
        level_src_boxes = torch.stack([bbox_preds, 
                                       bbox_preds_2], dim=1)
        level_tgt_boxes = torch.stack([kwargs['bbox'], 
                                       kwargs['inpaint_bboxes']], dim=1)
        level_losses = torch.stack([loss_func1(bbox_preds, kwargs['bbox']) + loss_func2(bbox_preds, kwargs['bbox']), 
                                    loss_func1(bbox_preds_2, kwargs['inpaint_bboxes']) + loss_func2(bbox_preds_2, kwargs['inpaint_bboxes'])], dim=-1)
        level_ious = torch.stack([self.get_iou(bbox_preds, kwargs['bbox']), 
                                  self.get_iou(bbox_preds_2, kwargs['inpaint_bboxes'])], dim=1)


        return (0.,0.,level_losses, level_ious, level_src_boxes,0.,)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_iou(self, pred, targ):
        batch_size = pred.shape[0]
        assert pred.shape[0] == targ.shape[0]
        all_iou = []
        for i in range(batch_size):
            all_iou.append(torchvision.ops.box_iou(pred[i].unsqueeze(0), targ[i].unsqueeze(0)).squeeze())
        return torch.tensor(all_iou)

    def freeze_backbone(self, freeze=True):
        for _, p in self.swin.named_parameters():
            p.requires_grad =  not freeze

 