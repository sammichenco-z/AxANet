import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2

def rollout(attentions, discard_ratio, head_fusion, target_layer=None, target_head=None):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for layer, b_attentions in enumerate(attentions):

            if target_layer is not None and layer != target_layer:
                continue

            # for bid, attention in enumerate(b_attentions):
            if True:
                attention = b_attentions[0]

                if target_head is not None:
                    attention_heads_fused = attention[target_head]
                else:
                    # the detr attention output has been mean
                    attention_heads_fused = attention
                
                attention_heads_fused = attention_heads_fused.cpu().to(torch.float32)
                # if head_fusion == "mean":
                #     attention_heads_fused = attention.mean(axis=1)
                # elif head_fusion == "max":
                #     attention_heads_fused = attention.max(axis=1)[0]
                # elif head_fusion == "min":
                #     attention_heads_fused = attention.min(axis=1)[0]
                # else:
                #     raise "Attention head fusion type Not supported"

                # Drop the lowest attentions, but
                # don't drop the class token
                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
                indices = indices[indices != 0]
                indices = indices[indices != 1]
                indices = indices[indices != 2]
                indices = indices[indices != 3]
                flat[0, indices] = 0

                I = torch.eye(flat.size(-1))
                a = (flat + 1.0*I)/2
                a = a / a.sum(dim=-1)

                result = torch.matmul(a, result)

    # Look at the total attention between the bbox token,
    # and the image patches
    mask = result[:4, 4:]
    for mask_id in range(len(mask)):
        this_mask = mask[mask_id]
        _, indices = this_mask.topk(int(this_mask.size(-1)*discard_ratio), -1, False)
        indices = indices[indices != 0]
        indices = indices[indices != 1]
        indices = indices[indices != 2]
        indices = indices[indices != 3]
        mask[mask_id, indices] = 0


    mask = mask.view(mask.size(0), -1)


    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(4, width, width).numpy()

    # for bid in range(4):
    #     mask[bid] = mask[bid]  / np.max(mask[bid])
    return mask

class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='bert_drop', head_fusion="mean",
        discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)

        return rollout(self.attentions, self.discard_ratio, self.head_fusion), output