import torch
import torchvision
from fairscale.nn.misc import checkpoint_wrapper
import random
from torch import nn
import math
# from .transformer_selfattn_imagegaze import myModelDecoder
# from .transformer_selfattn_imagegaze_concatmean import myModelDecoder
# from .transformer_selfattn_imagegaze_withclstoken import myModelDecoder
# from .transformer_selfattn_imagegaze_withclstoken_25token import myModelDecoder
# from .transformer_selfattn_imagegaze_withclstoken_2token import myModelDecoder
# from .transformer_alldetr import myModelDecoder
from .transformer_selfattn_imagegaze_concatinput_withclstoken import myModelDecoder
# from .transformer_selfattn_imagegaze_concatinput_withclstoken_noqueryselfattn import myModelDecoder

from IPython import embed

class smallgazebkbn(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        nPlanes = [64, 128, 256, 512, 1024]
        # Block 1
        self.b1_conv1_0=nn.Sequential(nn.Conv2d(in_channels, 32, 3, 1, padding=1), nn.BatchNorm2d(32),nn.ReLU())
        
        self.b1_conv2=nn.Sequential(nn.Conv2d(32, nPlanes[0], 3, 1, padding=1), nn.BatchNorm2d(nPlanes[0]),nn.ReLU())
        self.b1_conv3=nn.Sequential(nn.Conv2d(nPlanes[0], nPlanes[0], 3, 2, padding=1), nn.BatchNorm2d(nPlanes[0]),nn.ReLU())
        self.b1_res=nn.Sequential(nn.Conv2d(32, nPlanes[0], 3, 2,padding=1), nn.BatchNorm2d(nPlanes[0]),nn.ReLU())

        # Block 2
        self.b2_conv1=nn.Sequential(nn.Conv2d(nPlanes[0], nPlanes[1], 3, 1, padding=1), nn.BatchNorm2d(nPlanes[1]),nn.ReLU())
        self.b2_conv2=nn.Sequential(nn.Conv2d(nPlanes[1], nPlanes[1], 3, 2, padding=1), nn.BatchNorm2d(nPlanes[1]),nn.ReLU())
        self.b2_res=nn.Sequential(nn.Conv2d(nPlanes[0], nPlanes[1], 3, 2, padding=1), nn.BatchNorm2d(nPlanes[1]),nn.ReLU())

        # Block 3
        self.b3_conv1=nn.Sequential(nn.Conv2d(nPlanes[1], nPlanes[2], 3, 1, padding=1), nn.BatchNorm2d(nPlanes[2]),nn.ReLU())
        self.b3_conv2=nn.Sequential(nn.Conv2d(nPlanes[2], nPlanes[2], 3, 2, padding=1), nn.BatchNorm2d(nPlanes[2]),nn.ReLU())
        self.b3_res=nn.Sequential(nn.Conv2d(nPlanes[1], nPlanes[2], 3, 2, padding=1), nn.BatchNorm2d(nPlanes[2]),nn.ReLU())

        # Block 4
        self.b4_conv1=nn.Sequential(nn.Conv2d(nPlanes[2], nPlanes[3], 3, 1, padding=1), nn.BatchNorm2d(nPlanes[3]),nn.ReLU())
        self.b4_conv2=nn.Sequential(nn.Conv2d(nPlanes[3], nPlanes[3], 3, 2, padding=1), nn.BatchNorm2d(nPlanes[3]),nn.ReLU())
        self.b4_res=nn.Sequential(nn.Conv2d(nPlanes[2], nPlanes[3], 3, 2, padding=1), nn.BatchNorm2d(nPlanes[3]),nn.ReLU())

        # Block 5
        self.b5_conv1=nn.Sequential(nn.Conv2d(nPlanes[3], nPlanes[4], 3, 1, padding=1), nn.BatchNorm2d(nPlanes[4]),nn.ReLU())
        self.b5_conv2=nn.Sequential(nn.Conv2d(nPlanes[4], nPlanes[4], 3, 2, padding=1), nn.BatchNorm2d(nPlanes[4]),nn.ReLU())
        self.b5_res=nn.Sequential(nn.Conv2d(nPlanes[3], nPlanes[4], 3, 2, padding=1), nn.BatchNorm2d(nPlanes[4]),nn.ReLU())

    def forward(self, x):
        # Block 1
        x = self.b1_conv1_0(x)

        # Block 1_1
        res_x = self.b1_res(x)
        x = self.b1_conv2(x)
        x = self.b1_conv3(x)
        x = x + res_x

        # Block 2
        res_x = self.b2_res(x)
        x = self.b2_conv1(x)
        x = self.b2_conv2(x)
        x = x + res_x

        # Block 3
        res_x = self.b3_res(x)
        x = self.b3_conv1(x)
        x = self.b3_conv2(x)
        x = x + res_x

        # Block 4
        res_x = self.b4_res(x)
        x = self.b4_conv1(x)
        x = self.b4_conv2(x)
        x = x + res_x

        # Block 5
        res_x = self.b5_res(x)
        x = self.b5_conv1(x)
        x = self.b5_conv2(x)
        x = x + res_x

        return x


class DRAMAGAZEVideoTransformer_gaze(torch.nn.Module):
    def __init__(self, args, config, swin_model_rgb, swin_model_gaze, transformer_encoder):
        super(DRAMAGAZEVideoTransformer_gaze, self).__init__()
        self.config = config
        self.use_checkpoint = args.use_checkpoint and not args.freeze_backbone
        if self.use_checkpoint:
            self.swin_rgb = checkpoint_wrapper(swin_model_rgb, offload_to_cpu=True)
        else:
            self.swin_rgb = swin_model_rgb
        # if self.use_checkpoint:
        #     self.swin_gaze = checkpoint_wrapper(swin_model_gaze, offload_to_cpu=True)
        # else:
        #     self.swin_gaze = swin_model_gaze
        self.gaze_bkbn = smallgazebkbn(in_channels=3)
        
        self.trans_encoder = transformer_encoder
        self.img_feature_dim = int(args.img_feature_dim)
        self.use_grid_feat = args.grid_feat
        self.latent_feat_size = self.swin_rgb.backbone.norm.normalized_shape[0]
        
        self.fc_rgb = torch.nn.Linear(self.latent_feat_size, self.img_feature_dim)
        self.fc_gaze = torch.nn.Linear(self.latent_feat_size, self.img_feature_dim)
        #self.pooling = torch.nn.AdaptiveAvgPool2d(output_size=(2,2))
        self.compute_mask_on_the_fly = False # deprecated
        self.mask_prob = args.mask_prob
        self.mask_token_id = -1
        self.max_img_seq_length = args.max_img_seq_length

        #self.bddoia = getattr(args, 'bddoia', False)
        
        #self.img_feature_dim = int(args.img_feature_dim)
        # TODO: change
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.img_feature_dim*2, (4+21)*2)
        # )

        self.visualize_attn = args.visualize_attn

        self.myModelDecoder = myModelDecoder(args, self.latent_feat_size)

    def forward(self, *args, **kwargs):
        images = kwargs['img_feats']
        gazes = kwargs['gaze']

        # rgb
        B, S, C, H, W = images.shape  # batch, segment, chanel, hight, width
        # (B x S x C x H x W) --> (B x C x S x H x W)
        images = images.permute(0, 2, 1, 3, 4)
        vid_feats = self.swin_rgb(images)
        # if self.use_grid_feat==True:
        #     vid_feats = vid_feats.permute(0, 2, 3, 4, 1) # B S H W C

        # gaze
        B, S, C, H, W = gazes.shape  # batch, segment, chanel, hight, width
        # (B x S x C x H x W) --> (B x C x S x H x W)
        gazes = gazes.permute(0, 2, 1, 3, 4)[:,:,0,:,:]
        # gaze_feats = self.swin_gaze(gazes)
        gaze_feats = self.gaze_bkbn(gazes)
        # if self.use_grid_feat==True:
        #     gaze_feats = gaze_feats.permute(0, 2, 3, 4, 1)

        decoder_output = self.myModelDecoder(vid_feats, gaze_feats)
        logits = decoder_output['pred_logits']

        logits = logits.view(B, -1, 2) # (B, 25, 2)
        logits_action = logits[:, :4, :]
        logits_reason = logits[:, 4:, :]

        loss_action = self.get_cross_loss(logits_action.reshape(-1,2),kwargs['label_action'].view(-1))
        loss_reason = self.get_cross_loss(logits_reason.reshape(-1,2),kwargs['label_reason'].view(-1))

        attn_layers = decoder_output['attn_layers']

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
        assert freeze==False # temp
        for _, p in self.swin_rgb.named_parameters():
            p.requires_grad =  not freeze
        # for _, p in self.swin_gaze.named_parameters():
        #     p.requires_grad =  not freeze

 
 
 