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
# from .transformer_selfattn_imagegaze_concatinput_withclstoken_noqueryselfattn import myModelDecoder
from .transformer_selfattn_imagegaze_concatinput_withclstoken_4token import myModelDecoder

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

        self.use_gaussian_prior = True
        if self.use_gaussian_prior:
            self.gaussian_learnable_mu_xy = torch.nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float16))
            self.gaussian_learnable_std_xy = torch.nn.Parameter(torch.tensor([-2.0, -2.0], dtype=torch.float16))

    def make_gaussian_map(self, x):
        """Construct prior maps from Gaussian parameters."""
        size = x.shape[-2:]
        bs = x.shape[0]
        frame_size = x.shape[1]
        dtype = x.dtype
        device = x.device

        gaussian_maps = []
        meshgrids = torch.meshgrid(
            [torch.linspace(0, 1, size[0], dtype=dtype, device=device),
             torch.linspace(0, 1, size[1], dtype=dtype, device=device),])

        gaussian_maps = torch.ones(*size, dtype=dtype, device=device)
        for mu, log_std, mgrid in zip(self.gaussian_learnable_mu_xy, self.gaussian_learnable_std_xy, meshgrids):
            tmp_mu = mu
            tmp_std = torch.exp(log_std)
            gaussian_maps *= torch.exp(-((mgrid - tmp_mu) / tmp_std) ** 2 / 2)

        # visualize
        # cv2.imwrite("a.jpg", np.array(gaussian_maps.cpu().detach().unsqueeze(-1).repeat(1,1,3)*255, dtype=np.int))

        gaussian_maps = 1-gaussian_maps
        gaussian_maps = gaussian_maps.unsqueeze(0).unsqueeze(0).repeat(bs, frame_size, 1, 1, 1)
        return gaussian_maps

    def forward(self, *args, **kwargs):
        images = kwargs['img_feats']
        gazes = kwargs['gaze']

        if self.use_gaussian_prior:
            gaussian_map = self.make_gaussian_map(gazes)
            gazes = gazes * gaussian_map
        # RGB normalize
        mean = torch.as_tensor([0.485,0.456,0.406], dtype=gazes.dtype, device=gazes.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=gazes.dtype, device=gazes.device)
        gazes.sub_(mean[None, None, :, None, None]).div_(std[None, None, :, None, None])
        
        
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

 
 
 