import torch
import torchvision
from fairscale.nn.misc import checkpoint_wrapper
import random
from torch import nn
from src.modeling.load_bbox_pred_head import get_bbox_pred_model, get_bbox_loss, get_iou, get_class_loss, get_bounding_box_accuracy
import copy
from .detr import build_model as build_model_detr
# from .transformer_selfattn_image_withclstoken_4token import myModelDecoder



class DRAMADetr(torch.nn.Module):
    def __init__(self, args):
        super(DRAMADetr, self).__init__()
        # self.config = config

        # # get backbone
        # self.use_checkpoint = args.use_checkpoint and not args.freeze_backbone
        # if self.use_checkpoint:
        #     self.swin = checkpoint_wrapper(swin, offload_to_cpu=True)
        # else:
        #     self.swin = swin

        # # define the inter dimention of image/video features in decoder
        # self.use_grid_feat = args.grid_feat
        # self.latent_feat_size = self.swin.backbone.norm.normalized_shape[0]
        # self.img_feature_dim = int(args.img_feature_dim)

        # # transform the dimension of video features
        # self.fc = torch.nn.Linear(self.latent_feat_size, self.img_feature_dim)

        detr_args = copy.deepcopy(args)
        detr_args.lr_backbone = detr_args.learning_rate * detr_args.backbone_coef_lr
        detr_args.masks = False
        # backbone
        detr_args.backbone = 'resnet101'
        detr_args.dilation = False
        detr_args.position_embedding = 'sine'

        # dec_layers: of no use in encoder
        detr_args.dec_layers = 6

        MODEL_SIZE = {
            "tiny":[2, 4, 128],
            "small":[4, 8, 256],
            "base":[6, 8, 256],
            "large":[8, 8, 512],
            "huge":[12, 16, 1024],
        }


        gazetr_model_size = MODEL_SIZE[args.model_size]


        # transformer
        detr_args.enc_layers = gazetr_model_size[0]

        detr_args.dim_feedforward = 2048
        detr_args.hidden_dim = gazetr_model_size[2]
        detr_args.dropout = 0.1
        detr_args.nheads = gazetr_model_size[1]
        detr_args.pre_norm = False
        self.detr_encoder = build_model_detr(detr_args)

        self.img_feature_dim = int(args.img_feature_dim)
        self.latent_feat_size = detr_args.hidden_dim
        self.fc = torch.nn.Linear(self.latent_feat_size, self.img_feature_dim)

        # should be ture in risk localization
        self.pred_bbox = getattr(args, 'pred_bbox', False)

        # different bbox heads to choose     
        self.bbox_head = get_bbox_pred_model(args)

    def forward(self, *args, **kwargs):

        # grad cam can only input a tuple (args, kwargs)
        if isinstance(args, tuple) and len(args) != 0:
            kwargs = args[0]
            args= ()

        images = kwargs['img_feats']
        B, S, C, H, W = images.shape  # batch, segment, chanel, hight, width

        images = images[:,0,...]
        vid_feats = self.detr_encoder(images)

        # print(vid_feats.shape)

        vid_feats = vid_feats.view(B, self.latent_feat_size, -1).permute(0, 2, 1)
        vid_feats = self.fc(vid_feats)

        # bbox head
        bbox_logits, class_logits = self.bbox_head(vid_feats)

        bbox_loss = get_bbox_loss(bbox_logits, kwargs['bbox'])        
        with torch.no_grad():
            bbox_iou = get_iou(bbox_logits, kwargs['bbox'])
            bbox_acc = get_bounding_box_accuracy(bbox_logits, kwargs['bbox'])


        if class_logits is not None:
            class_loss = get_class_loss(class_logits, kwargs['label'])
            class_preds = class_logits.argmax(-1)
        else:
            class_loss = torch.zeros(1).to(vid_feats.device)
            class_preds = -1*torch.ones(B).to(vid_feats.device)

        return {
            "bbox_logits": bbox_logits,
            "bbox_loss": bbox_loss,
            "bbox_iou": bbox_iou,
            "bbox_acc": bbox_acc,
            "class_logits": class_logits,
            "class_loss": class_loss,
            "class_preds": class_preds,
        }


    def freeze_backbone(self, freeze=True):
        for _, p in self.detr_encoder.backbone.named_parameters():
            p.requires_grad =  not freeze

 