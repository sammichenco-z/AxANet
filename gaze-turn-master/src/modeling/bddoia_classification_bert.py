import torch
import torchvision
from fairscale.nn.misc import checkpoint_wrapper
import random
from torch import nn
import math
# from .transformer_selfattn_image_withclstoken import myModelDecoder
from .transformer_selfattn_image_withclstoken_noqueryselfattn import myModelDecoder

from src.modeling.load_action_pred_head import get_bbox_pred_model

from IPython import embed

class BDDOIAVideoTransformer(torch.nn.Module):
    def __init__(self, args, config, swin, transformer_encoder):
        super(BDDOIAVideoTransformer, self).__init__()
        self.config = config
        self.use_checkpoint = args.use_checkpoint and not args.freeze_backbone
        if self.use_checkpoint:
            self.swin = checkpoint_wrapper(swin, offload_to_cpu=True)
        else:
            self.swin = swin
        self.trans_encoder = transformer_encoder
        self.img_feature_dim = int(args.img_feature_dim)
        self.use_grid_feat = args.grid_feat
        self.latent_feat_size = self.swin.backbone.norm.normalized_shape[0]
        
        self.fc = torch.nn.Linear(self.latent_feat_size, self.img_feature_dim)
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
        
        # self.myModelDecoder = myModelDecoder(args, self.latent_feat_size)

        self.gaze_action_head = get_bbox_pred_model(args)


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

        # bbox head
        # vid_feats = vid_feats.mean(1)
        # logits = self.classifier(vid_feats)

        # decoder_output = self.myModelDecoder(vid_feats)
        # logits = decoder_output['pred_logits']

        # logits = logits.view(B, -1, 2) # (B, 25, 2)
        # logits_action = logits[:, :4, :]
        # logits_reason = logits[:, 4:, :]

        # loss_action = self.get_cross_loss(logits_action.reshape(-1,2),kwargs['label_action'].view(-1))
        # loss_reason = self.get_cross_loss(logits_reason.reshape(-1,2),kwargs['label_reason'].view(-1))

        # attn_layers = decoder_output['attn_layers']

        logits_action, logits_reason, loss_action, loss_reason, attn_layers = self.gaze_action_head(vid_feats, kwargs['label_action'], kwargs['label_reason'])

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
        for _, p in self.swin.named_parameters():
            p.requires_grad =  not freeze

 