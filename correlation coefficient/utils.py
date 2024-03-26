import numpy as np
import scipy
import cv2


def corr_coeff(pred, target):
    x = pred
    y = target
    xm, ym = x - x.mean(), y - y.mean()
    r_num = np.mean(xm * ym)
    r_den = np.sqrt(
        np.mean(np.square(xm)) * np.mean(np.square(ym)))
    r = r_num / (r_den+1e-20)
    p = scipy.stats.pearsonr(x.reshape(-1), y.reshape(-1)).pvalue
    return r, p

def sim(pred, target):
    sim = np.minimum(pred, target).sum()
    return sim


def normlize_each_attn_map(attn_list, resize_size=224):
    '''normilize each attn map'''
    new_attn_list = []
    for i in range(len(attn_list)):
        attn = attn_list[i].copy()
        attn = cv2.resize(attn, dsize=(resize_size,resize_size))
        attn = attn / (attn.sum()+1e-20)
        new_attn_list.append(attn)

    return new_attn_list
