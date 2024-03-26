import torch
import torchvision
from fairscale.nn.misc import checkpoint_wrapper
import random
from torch import nn
import math
# from .transformer_selfattn_image_withclstoken import myModelDecoder
# from .transformer_selfattn_image_withclstoken_noqueryselfattn import myModelDecoder
from .transformer_selfattn_image_withclstoken_4token import myModelDecoder

from .detr import build_model as build_model_detr

from IPython import embed

import copy

class BDDOIAVideoTransformer(torch.nn.Module):
    def __init__(self, args):
        super(BDDOIAVideoTransformer, self).__init__()
        # self.use_checkpoint = args.use_checkpoint and not args.freeze_backbone
        # if self.use_checkpoint:
        #     self.swin = checkpoint_wrapper(swin, offload_to_cpu=True)
        # else:
        #     self.swin = swin
        # self.latent_feat_size = self.swin.backbone.norm.normalized_shape[0]
        
        self.img_feature_dim = int(args.img_feature_dim)
        self.use_grid_feat = args.grid_feat
        
        # self.fc = torch.nn.Linear(self.latent_feat_size, self.img_feature_dim)
        #self.pooling = torch.nn.AdaptiveAvgPool2d(output_size=(2,2))
        self.compute_mask_on_the_fly = False # deprecated
        self.mask_prob = args.mask_prob
        self.mask_token_id = -1
        self.max_img_seq_length = args.max_img_seq_length

        #self.bddoia = getattr(args, 'bddoia', False)
        
        #self.img_feature_dim = int(args.img_feature_dim)
        # TODO: change
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.latent_feat_size, (4+21)*2)
        # )
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.latent_feat_size*49, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, (4+21)*2),
        # )
        
        self.visualize_attn = args.visualize_attn
        
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

        self.myModelDecoder = myModelDecoder(args, detr_args.hidden_dim)

    def forward(self, *args, **kwargs):

        # grad cam can only input a tuple (args, kwargs)
        if isinstance(args, tuple) and len(args) != 0:
            kwargs = args[0]
            args= ()

        images = kwargs['img_feats']
        B, S, C, H, W = images.shape  # batch, segment, chanel, hight, width
        # (B x S x C x H x W) --> (B x C x S x H x W)
        # images = images.permute(0, 2, 1, 3, 4)
        
        # vid_feats = self.swin(images)
        # if self.use_grid_feat==True:
        #     vid_feats = vid_feats.permute(0, 2, 3, 4, 1)
        # vid_feats = vid_feats.view(B, -1, self.latent_feat_size)
        
        images = images[:,0,...]
        vid_feats = self.detr_encoder(images)

        # bbox head
        # vid_feats = vid_feats.mean(1)
        # logits = self.classifier(vid_feats)
        decoder_output = self.myModelDecoder(vid_feats)
        logits = decoder_output['pred_logits']

        logits = logits.view(B, -1, 2) # (B, 25, 2)
        logits_action = logits[:, :4, :]
        logits_reason = logits[:, 4:, :]

        loss_action = self.get_cross_loss(logits_action.reshape(-1,2),kwargs['label_action'].view(-1))
        loss_reason = self.get_cross_loss(logits_reason.reshape(-1,2),kwargs['label_reason'].view(-1))

        attn_layers = decoder_output['attn_layers']
        # embed()

        if self.visualize_attn:
            return (logits_action, logits_reason, loss_action, loss_reason, attn_layers)
        else:
            return (logits_action, logits_reason, loss_action, loss_reason)
    
    def get_l1_loss(self, pred, targ):
        l1_loss = nn.L1Loss()
        return l1_loss(pred.to(torch.float16), targ.to(torch.float16))
    def get_cross_loss(self, pred, targ):
        cross = nn.CrossEntropyLoss()
        return cross(pred, targ)

    def freeze_backbone(self, freeze=True):
        for _, p in self.detr_encoder.backbone.named_parameters():
            p.requires_grad =  not freeze

 