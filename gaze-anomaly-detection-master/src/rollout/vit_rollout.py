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
        for layer, attention in enumerate(attentions):

            if target_layer is not None and layer != target_layer:
                continue

            if target_head is not None:
                attention_heads_fused = attention[:, target_head, ...]

            elif head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

    # Look at the total attention between the bbox token,
    # and the image patches
    mask = result[0, 0, 1:]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask    

class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='bert_drop', head_fusion="mean",
        discard_ratio=0.9, per_layer_per_head=False):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

        self.attentions = []

        self.per_layer_per_head = per_layer_per_head

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)

        if not self.per_layer_per_head:
            return rollout(self.attentions, self.discard_ratio, self.head_fusion), output
    
        else:
            layer_number = len(self.attentions)
            head_number = self.attentions[0].size(1)

            rollout_output = {}

            for layer_id in range(layer_number):
                rollout_output_single_layer = {}
                for head_id in range(head_number):
                    rollout_output_single_layer[f"head_{head_id}"] = rollout(self.attentions, self.discard_ratio, self.head_fusion, layer_id, head_id)                

                rollout_output_single_layer[f'layer_{layer_id}_mean'] = rollout(self.attentions, self.discard_ratio, "mean", layer_id)

                rollout_output[f'layer_{layer_id}'] = rollout_output_single_layer
            
            return rollout_output, output