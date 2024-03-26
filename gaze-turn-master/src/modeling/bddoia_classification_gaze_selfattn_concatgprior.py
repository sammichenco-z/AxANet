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
# from .transformer_selfattn_imagegaze_concatinput_withclstoken import myModelDecoder
from .transformer_selfattn_imagegaze_concatinput_withclstoken_concatgprior import myModelDecoder
# from .transformer_selfattn_imagegaze_concatinput_withclstoken_noqueryselfattn import myModelDecoder
# from .transformer_selfattn_imagegaze_concatinput_withclstoken_4token import myModelDecoder

from itertools import product
from IPython import embed

class DRAMAGAZEVideoTransformer_gaze(torch.nn.Module):
    def __init__(self, args, config, swin_model_rgb, swin_model_gaze, transformer_encoder):
        super(DRAMAGAZEVideoTransformer_gaze, self).__init__()
        self.config = config
        self.use_checkpoint = args.use_checkpoint and not args.freeze_backbone
        if self.use_checkpoint:
            self.swin_rgb = checkpoint_wrapper(swin_model_rgb, offload_to_cpu=True)
        else:
            self.swin_rgb = swin_model_rgb
        if self.use_checkpoint:
            self.swin_gaze = checkpoint_wrapper(swin_model_gaze, offload_to_cpu=True)
        else:
            self.swin_gaze = swin_model_gaze
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

        self.gaussian_init = 'manual'
        self.gaussians = self.set_gaussians()

    def set_gaussians(self):
        if self.gaussian_init == 'manual':
            gaussians = torch.Tensor([
                    list(product([0.25, 0.5, 0.75], repeat=2)) +[(0.5, 0.5)]+
                    [(0.5, 0.25), (0.5, 0.5), (0.5, 0.75)] +
                    [(0.25, 0.5), (0.5, 0.5), (0.75, 0.5)],
                    [(-2.0, -2.0)] * 9 + [(-1.0, -1.0)]+ [(-1.0, -2.0)] * 3 + 
                    [(-2.0, -1.0)] * 3,
            ]).permute(1, 2, 0)

        elif self.gaussian_init == 'random':
            n_gaussians = 16
            with torch.no_grad():
                gaussians = torch.stack([
                        torch.randn(
                            n_gaussians, 2, dtype=torch.float) * .1 + 0.5,
                        torch.randn(
                            n_gaussians, 2, dtype=torch.float) * .2 - 1,],
                    dim=2)

        gaussians = nn.Parameter(gaussians, requires_grad=True)
        return gaussians

    def make_gaussian_maps(self, x, size=None, scaling=6.):
        """Construct prior maps from Gaussian parameters."""
        gaussians = self.gaussians
        if size is None:
            size = x.shape[-2:]
            bs = x.shape[0]
        else:
            size = [size] * 2
            bs = 1
        dtype = x.dtype
        device = x.device

        gaussian_maps = []
        map_template = torch.ones(*size, dtype=dtype, device=device)
        meshgrids = torch.meshgrid(
            [torch.linspace(0, 1, size[0], dtype=dtype, device=device),
             torch.linspace(0, 1, size[1], dtype=dtype, device=device),])

        for gaussian_idx, yx_mu_logstd in enumerate(torch.unbind(gaussians)):
            map = map_template.clone()
            for mu_logstd, mgrid in zip(yx_mu_logstd, meshgrids):
                mu = mu_logstd[0]
                std = torch.exp(mu_logstd[1])
                map *= torch.exp(-((mgrid - mu) / std) ** 2 / 2)

            map *= scaling
            gaussian_maps.append(map)

        gaussian_maps = torch.stack(gaussian_maps)
        gaussian_maps = gaussian_maps.unsqueeze(0).expand(bs, -1, -1, -1).unsqueeze(2)
        return gaussian_maps

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
        gazes = gazes.permute(0, 2, 1, 3, 4)
        gaze_feats = self.swin_gaze(gazes)
        # if self.use_grid_feat==True:
        #     gaze_feats = gaze_feats.permute(0, 2, 3, 4, 1)

        gaussian_maps = self.make_gaussian_maps(gaze_feats)
        gaze_feats = torch.cat((gaze_feats, gaussian_maps), dim=1)

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
        for _, p in self.swin_gaze.named_parameters():
            p.requires_grad =  not freeze

 
 
 