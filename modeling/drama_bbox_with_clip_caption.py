import torch
import torchvision
from fairscale.nn.misc import checkpoint_wrapper
import random
from torch import nn
from src.modeling.load_bbox_pred_head import get_bbox_pred_model, get_bbox_pred_model_with_text, get_bbox_loss, get_iou, get_class_loss, MLP

class DRAMAVideoTransformerGTCAPTION(torch.nn.Module):
    def __init__(self, args, config, swin, transformer_encoder):
        super(DRAMAVideoTransformerGTCAPTION, self).__init__()
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
        self.bbox_head = get_bbox_pred_model_with_text(args)

        self.use_distillation = False

        self.use_text_as_target = False
        if self.use_text_as_target:
            self.text_head = MLP(self.img_feature_dim, self.img_feature_dim, 768, 2)

    def forward(self, *args, **kwargs):
        images = kwargs['img_feats']

        B, S, C, H, W = images.shape  # batch, segment, chanel, hight, width
        # (B x S x C x H x W) --> (B x C x S x H x W)
        images = images.permute(0, 2, 1, 3, 4)
        vid_feats = self.swin(images)
        if self.use_grid_feat==True:
            vid_feats = vid_feats.permute(0, 2, 3, 4, 1)
        vid_feats = vid_feats.view(B, -1, self.latent_feat_size)
        vid_feats = self.fc(vid_feats)

        text_embedding = kwargs['caption']
        
        if self.use_distillation:
            text_distillation_loss = -torch.cosine_similarity(vid_feats.mean(1), text_embedding).mean(0)
            # text_distillation_loss = text_distillation_loss-text_distillation_loss
        elif self.use_text_as_target:
            text_logits = self.text_head(vid_feats.mean(1))
            # text_distillation_loss = nn.MSELoss()(text_logits, text_embedding)
            text_distillation_loss = -torch.cosine_similarity(text_logits, text_embedding).mean(0)
        else:
            text_distillation_loss = torch.zeros(1)
            text_distillation_loss = text_distillation_loss.to(vid_feats.device)

        # bbox head
        bbox_logits = self.bbox_head(vid_feats)


        bbox_loss = get_bbox_loss(bbox_logits, kwargs['bbox'])
        with torch.no_grad():
            bbox_iou = get_iou(bbox_logits, kwargs['bbox'])


        return {
            "bbox_logits": bbox_logits,
            "bbox_loss": bbox_loss,
            "bbox_iou": bbox_iou,
            "distillation_loss": text_distillation_loss,
        }


    def freeze_backbone(self, freeze=True):
        for _, p in self.swin.named_parameters():
            p.requires_grad =  not freeze

 