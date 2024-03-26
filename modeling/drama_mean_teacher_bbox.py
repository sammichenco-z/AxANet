import random

import torch
import torch.nn.functional as F
import torchvision
from fairscale.nn.misc import checkpoint_wrapper
from torch import nn
from torchvision.ops.boxes import box_area


class DRAMAMEANTEACHERVideoTransformer(torch.nn.Module):
    def __init__(self, args, config, swin, transformer_encoder):
        super(DRAMAMEANTEACHERVideoTransformer, self).__init__()
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
        self.bbox_head = MLP(self.img_feature_dim, self.img_feature_dim, 4, 3)

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
        vid_feats = vid_feats.mean(1)

        # bbox head
        bbox_preds = self.bbox_head(vid_feats).sigmoid()
        bbox_preds = box_cxcywh_to_xyxy(bbox_preds)

        return (0.,0.,0., 0., bbox_preds,0.,)

    def get_detect_loss(self, pred, targ):
        loss_func1 = nn.L1Loss()
        loss_func2 = generalized_box_iou_loss

        return loss_func1(pred, targ) + loss_func2(pred, targ)

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

 
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def generalized_box_iou_loss(boxes1, boxes2):
    return 1-torch.diag(generalized_box_iou(boxes1, boxes2)).clamp(min=-1.0, max=1.0).mean()

# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    boxes1 = torch.clamp(boxes1, min=0, max=1)
    boxes2 = torch.clamp(boxes2, min=0, max=1)
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)