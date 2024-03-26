import math

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.ops.boxes import box_area

from src.layers.bert.modeling_bert import BertEncoder
from src.layers.bert import BertConfig, BertEncoder

from src.utils.logger import LOGGER as logger


def get_bbox_pred_model(args):
    if getattr(args, 'use_trans', False):
        return bbox_Pred_Head_tr(args)
    else:
        return bbox_Pred_Head(args)

class bbox_Pred_Head(torch.nn.Module):
    def __init__(self, args):
        super(bbox_Pred_Head, self).__init__()

        self.img_feature_dim = int(args.img_feature_dim)
        self.bbox_head = MLP(self.img_feature_dim, self.img_feature_dim, 4, 3)

        self.pred_class = getattr(args, 'pred_class', False)
        if self.pred_class:
            self.class_head = nn.Linear(self.img_feature_dim, 4)

    def forward(self, vid_feats):
        vid_feats = vid_feats.mean(1)

        bbox_pred = self.bbox_head(vid_feats).sigmoid()
        bbox_pred = box_cxcywh_to_xyxy(bbox_pred)
        # return  bbox_pred

        if self.pred_class:
            class_pred = self.class_head(vid_feats)
        else:
            class_pred = None

        return bbox_pred, class_pred


class bbox_Pred_Head_tr(torch.nn.Module):
    def __init__(self, args):
        super(bbox_Pred_Head_tr, self).__init__()

        self.img_feature_dim = int(args.img_feature_dim)


        self.config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
        MODEL_SIZE = {
            "tiny":[2, 4, 128],
            "small":[4, 8, 256],
            "base":[6, 8, 256],
            "large":[8, 8, 512],
            "huge":[12, 16, 1024],
        }
        gazetr_model_size = MODEL_SIZE[args.model_size]

        self.config.hidden_size = gazetr_model_size[2]
        self.config.num_attention_heads = gazetr_model_size[1]
        self.config.num_hidden_layers = gazetr_model_size[0]
        self.config.intermediate_size = 2048

        self.trans_encoder = BertEncoder(self.config)

        self.num_queries = 1
        self.query_embed = nn.Embedding(self.num_queries, self.config.hidden_size)

        self.img_embedding = nn.Linear(self.img_feature_dim, self.config.hidden_size, bias=True)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.bbox_head = MLP(self.config.hidden_size, self.config.hidden_size,  4, 3)

        self.pred_class = getattr(args, 'pred_class', False)
        if self.pred_class:
            self.class_head = nn.Linear(self.config.hidden_size, 4)

        # self.pooler = BertPooler(self.config)
        # self.bert_attention_mask = self.get_bert_attention_mask()

    def forward(self, vid_feats):
        B, num_tokens, latent_feat_size = vid_feats.shape

        vid_feats = self.img_embedding(vid_feats)
        video_tokens = self.dropout(vid_feats)

        bbox_tokens = self.query_embed.weight.repeat(B, 1, 1)

        input_tokens = torch.cat((bbox_tokens, video_tokens), dim=1)
        attention_mask = self.get_bert_attention_mask(num_tokens)
        attention_mask = self.expand_and_repeat(attention_mask, B)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask).to(vid_feats.device)

        if self.trans_encoder.output_attentions:
            self.trans_encoder.set_output_attentions(False)
        
        # print(input_tokens.shape)
        encoder_outputs = self.trans_encoder(input_tokens, 
                                    extended_attention_mask, 
                                    head_mask=[None] * self.config.num_hidden_layers, 
                                    encoder_history_states=None
                                    )

        sequence_output = encoder_outputs[0][:, :self.num_queries, :]
        # pooled_output = self.pooler(sequence_output)

        risk_feats = sequence_output[:, 0, :]
        bbox_pred = self.bbox_head(risk_feats).sigmoid()
        bbox_pred = box_cxcywh_to_xyxy(bbox_pred)

        if self.pred_class:
            class_logits = self.class_head(risk_feats)
        else:
            class_logits = None
        return bbox_pred, class_logits

    def expand_and_repeat(self, input_tensor, batch_size):
        ndim = len(input_tensor.shape)
        new_shape = [1]+list(input_tensor.shape)
        return input_tensor.reshape(*new_shape).repeat((batch_size,)+(1,) * ndim)

    def get_bert_attention_mask(self, num_tokens, attention_type="selfatten"):
        max_len = self.num_queries + num_tokens

        q_start, q_end = 0, self.num_queries
        v_start, v_end = self.num_queries, self.num_queries+num_tokens

        attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
        # diagonal mask for query
        attention_mask[q_start:q_end, q_start:q_end].copy_(torch.eye(self.num_queries))
        # full attention for q-v
        attention_mask[q_start:q_end, v_start:v_end] = 1
        # full attention for video tokens
        attention_mask[v_start:v_end, v_start:v_end] = 1

        return attention_mask

    def get_extended_attention_mask(self, attention_mask):
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask


    def get_bbox_loss(self, pred, targ):
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

def get_bbox_pred_model_with_text(args):
    return bbox_Pred_Head_with_text(args)

class bbox_Pred_Head_with_text(torch.nn.Module):
    def __init__(self, args):
        super(bbox_Pred_Head_with_text, self).__init__()

        self.img_feature_dim = int(args.img_feature_dim)
        self.bbox_head = MLP(self.img_feature_dim, self.img_feature_dim, 4, 3)

    def forward(self, vid_feats, text_feats=None):

        vid_feats = vid_feats.mean(1)
        if text_feats is not None:
            vid_feats = torch.cat((vid_feats, text_feats), dim=-1)

        bbox_pred = self.bbox_head(vid_feats).sigmoid()
        bbox_pred = box_cxcywh_to_xyxy(bbox_pred)
        # return  bbox_pred

        # class_pred = self.class_head(vid_feats)

        return bbox_pred

def get_bbox_pred_model_tr_with_text(args):
    return bbox_Pred_Head_tr_with_text(args)

class bbox_Pred_Head_tr_with_text(torch.nn.Module):
    def __init__(self, args):
        super(bbox_Pred_Head_tr_with_text, self).__init__()

        self.img_feature_dim = int(args.img_feature_dim)


        self.config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
        self.config.hidden_size=768
        self.config.num_attention_heads = 12
        self.config.num_hidden_layers = 12
        self.config.intermediate_size = 3072

        self.trans_encoder = BertEncoder(self.config)

        self.num_queries = 1
        self.query_embed = nn.Embedding(self.num_queries, self.config.hidden_size)

        self.img_embedding = nn.Linear(self.img_feature_dim, self.config.hidden_size, bias=True)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.bbox_head = MLP(self.config.hidden_size, self.config.hidden_size,  4, 3)

        self.text_head = MLP(self.config.hidden_size, self.config.hidden_size, 768, 1)

        # self.pooler = BertPooler(self.config)
        # self.bert_attention_mask = self.get_bert_attention_mask()

    def forward(self, vid_feats, text_embedding=None):
        B, num_tokens, latent_feat_size = vid_feats.shape

        vid_feats = self.img_embedding(vid_feats)
        video_tokens = self.dropout(vid_feats)

        bbox_tokens = self.query_embed.weight.repeat(B, 1, 1)

        input_tokens = torch.cat((bbox_tokens, video_tokens), dim=1)
        attention_mask = self.get_bert_attention_mask(num_tokens)
        attention_mask = self.expand_and_repeat(attention_mask, B)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask).to(vid_feats.device)

        if self.trans_encoder.output_attentions:
            self.trans_encoder.set_output_attentions(False)
        
        # print(input_tokens.shape)
        encoder_outputs = self.trans_encoder(input_tokens, 
                                    extended_attention_mask, 
                                    head_mask=[None] * self.config.num_hidden_layers, 
                                    encoder_history_states=None
                                    )

        sequence_output = encoder_outputs[0][:, :self.num_queries, :]
        # pooled_output = self.pooler(sequence_output)

        # now we only have one query for the most risky objects
        risk_feats = sequence_output[:, 0, :]

        bbox_pred = self.bbox_head(risk_feats).sigmoid()
        bbox_pred = box_cxcywh_to_xyxy(bbox_pred)

        if text_embedding is not None:
            text_pred = self.text_head(risk_feats)
            # text_pred = sequence_output[:, 0, :]
            return bbox_pred, text_pred

        return bbox_pred

    def expand_and_repeat(self, input_tensor, batch_size):
        ndim = len(input_tensor.shape)
        new_shape = [1]+list(input_tensor.shape)
        return input_tensor.reshape(*new_shape).repeat((batch_size,)+(1,) * ndim)

    def get_bert_attention_mask(self, num_tokens, attention_type="selfatten"):
        max_len = self.num_queries + num_tokens

        q_start, q_end = 0, self.num_queries
        v_start, v_end = self.num_queries, self.num_queries+num_tokens

        attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
        # diagonal mask for query
        attention_mask[q_start:q_end, q_start:q_end].copy_(torch.eye(self.num_queries))
        # full attention for q-v
        attention_mask[q_start:q_end, v_start:v_end] = 1
        # full attention for video tokens
        attention_mask[v_start:v_end, v_start:v_end] = 1

        return attention_mask

    def get_extended_attention_mask(self, attention_mask):
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask


    def get_bbox_loss(self, pred, targ):
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


def get_bbox_pred_model_tr_with_text_input(args):
    return bbox_Pred_Head_tr_with_text_input(args)

class bbox_Pred_Head_tr_with_text_input(torch.nn.Module):
    def __init__(self, args):
        super(bbox_Pred_Head_tr_with_text_input, self).__init__()

        print("******************bbox_Pred_Head_tr_with_text_input******************")
        self.img_feature_dim = int(args.img_feature_dim)


        self.config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
        self.config.hidden_size=768
        self.config.num_attention_heads = 12
        self.config.num_hidden_layers = 12
        self.config.intermediate_size = 3072

        self.trans_encoder = BertEncoder(self.config)

        self.num_queries = 1
        self.query_embed = nn.Embedding(1, self.config.hidden_size)

        self.img_embedding = nn.Linear(self.img_feature_dim, self.config.hidden_size, bias=True)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.bbox_head = MLP(self.config.hidden_size, self.config.hidden_size,   4, 3)

        self.text_head = MLP(self.config.hidden_size, self.config.hidden_size, self.config.hidden_size, 2)

        # self.pooler = BertPooler(self.config)
        # self.bert_attention_mask = self.get_bert_attention_mask()

    def forward(self, vid_feats, text_embedding=None):
        B, num_tokens, latent_feat_size = vid_feats.shape

        vid_feats = self.img_embedding(vid_feats)
        video_tokens = self.dropout(vid_feats)

        bbox_tokens = self.query_embed.weight.repeat(B, 1, 1)

        if text_embedding is not None:
            text_tokens = self.text_head(text_embedding)
            input_tokens = torch.cat((bbox_tokens, video_tokens, text_tokens), dim=1)
            num_tokens = num_tokens+text_tokens.shape[1]
        else:
            input_tokens = torch.cat((bbox_tokens, video_tokens), dim=1)
        attention_mask = self.get_bert_attention_mask(num_tokens)
        attention_mask = self.expand_and_repeat(attention_mask, B)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask).to(vid_feats.device)

        if self.trans_encoder.output_attentions:
            self.trans_encoder.set_output_attentions(False)
        
        # print(input_tokens.shape)
        encoder_outputs = self.trans_encoder(input_tokens, 
                                    extended_attention_mask, 
                                    head_mask=[None] * self.config.num_hidden_layers, 
                                    encoder_history_states=None
                                    )

        sequence_output = encoder_outputs[0][:, :self.num_queries, :]
        # pooled_output = self.pooler(sequence_output)

        # now we only have one query for the most risky objects
        risk_feats = sequence_output[:, 0, :]

        bbox_pred = self.bbox_head(risk_feats).sigmoid()
        bbox_pred = box_cxcywh_to_xyxy(bbox_pred)


        return bbox_pred

    def expand_and_repeat(self, input_tensor, batch_size):
        ndim = len(input_tensor.shape)
        new_shape = [1]+list(input_tensor.shape)
        return input_tensor.reshape(*new_shape).repeat((batch_size,)+(1,) * ndim)

    def get_bert_attention_mask(self, num_tokens, attention_type="selfatten"):
        max_len = self.num_queries + num_tokens

        q_start, q_end = 0, self.num_queries
        v_start, v_end = self.num_queries, self.num_queries+num_tokens

        attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
        # diagonal mask for query
        attention_mask[q_start:q_end, q_start:q_end].copy_(torch.eye(self.num_queries))
        # full attention for q-v
        attention_mask[q_start:q_end, v_start:v_end] = 1
        # full attention for video tokens
        attention_mask[v_start:v_end, v_start:v_end] = 1

        return attention_mask

    def get_extended_attention_mask(self, attention_mask):
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask


    def get_bbox_loss(self, pred, targ):
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

def get_bbox_pred_model_inpaint(args):
    return bbox_Pred_Head_inpaint(args)

class bbox_Pred_Head_inpaint(torch.nn.Module):
    def __init__(self, args):
        super(bbox_Pred_Head_inpaint, self).__init__()

        self.img_feature_dim = int(args.img_feature_dim)
        self.bbox_head = MLP(self.img_feature_dim, self.img_feature_dim, 4, 3)
        self.bbox_head2 = MLP(self.img_feature_dim, self.img_feature_dim, 4, 3)

        # the labels means whether the box is the most risky region
        self.cls_head = nn.Linear(self.img_feature_dim, 2)

    def forward(self, vid_feats):

        # sampling pooling, should be optimized
        vid_feats = vid_feats.mean(1)

        # different heads
        bbox_pred = self.bbox_head(vid_feats).sigmoid()
        bbox_pred = box_cxcywh_to_xyxy(bbox_pred)
        bbox_pred_2 = self.bbox_head2(vid_feats).sigmoid()
        bbox_pred_2 = box_cxcywh_to_xyxy(bbox_pred_2)
        # cls_pred = self.cls_head(vid_feats)

        return bbox_pred, bbox_pred_2, 0.



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

def get_bbox_loss(pred, targ):
    loss_func1 = nn.L1Loss()
    loss_func2 = generalized_box_iou_loss

    return loss_func1(pred, targ) + loss_func2(pred, targ)

def get_class_loss(pred, targ):
    loss_func = nn.CrossEntropyLoss()
    return loss_func(pred, targ)

def get_iou(pred, targ):
    batch_size = pred.shape[0]
    assert pred.shape[0] == targ.shape[0]
    all_iou = []
    for i in range(batch_size):
        all_iou.append(torchvision.ops.box_iou(pred[i].unsqueeze(0), targ[i].unsqueeze(0)).squeeze())
    return torch.tensor(all_iou)



def get_bounding_box_accuracy(pred, targ):
    """
    Calculate the bounding box accuracy.
    Bounding box accuracy is defined as the percentage of predicted boxes whose
    center point falls inside the target box.

    Args:
    pred (torch.Tensor): Predicted bounding boxes of shape (batch_size, 4),
                         where each box is represented as [xmin, ymin, xmax, ymax].
    targ (torch.Tensor): Target bounding boxes of shape (batch_size, 4),
                         where each box is represented as [xmin, ymin, xmax, ymax].

    Returns:
    torch.Tensor: The accuracy for each item in the batch.
    """
    batch_size = pred.shape[0]
    assert pred.shape == targ.shape

    # Calculate the center of each predicted box
    pred_centers = (pred[:, :2] + pred[:, 2:]) / 2

    # Check if the centers are within the target boxes
    within_target = (
        (pred_centers[:, 0] >= targ[:, 0]) & (pred_centers[:, 0] <= targ[:, 2]) &
        (pred_centers[:, 1] >= targ[:, 1]) & (pred_centers[:, 1] <= targ[:, 3])
    )

    # Calculate accuracy
    accuracy = within_target.float()

    return torch.tensor(accuracy)


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

def ciou_loss(pred, target, eps=1e-7):
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    ious = overlap / union

    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw**2 + ch**2 + eps
    
    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
    rho2 = left + right
    
    factor = 4 / math.pi**2
    v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

    with torch.no_grad():
        alpha = (ious > 0.5).float() * v / (1 - ious + v)

    # CIoU
    cious = ious - (rho2 / c2 + alpha * v)
    loss = 1 - cious.clamp(min=-1.0, max=1.0)
    return loss.mean(0)

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