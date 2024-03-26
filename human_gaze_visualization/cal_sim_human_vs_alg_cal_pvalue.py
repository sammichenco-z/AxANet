import os

import torch
import torch.nn.functional as F
import numpy as np

from IPython import embed
from scipy import interpolate
import cv2
from PIL import Image

import copy
import scipy

import matplotlib.pyplot as plt
import pandas as pd

def corr_coeff(pred, target):
    x = pred
    y = target
    xm, ym = x - x.mean(), y - y.mean()
    r_num = np.mean(xm * ym)
    r_den = np.sqrt(
        np.mean(np.square(xm)) * np.mean(np.square(ym)))
    r = r_num / (r_den+1e-20)
    # if r == 0:
        # embed()
    # print('===================')
    # print(r)
    # print(scipy.stats.pearsonr(x.reshape(-1), y.reshape(-1)))
    p = scipy.stats.pearsonr(x.reshape(-1), y.reshape(-1)).pvalue
    return r, p

def sim(pred, target):
    sim = np.minimum(pred, target).sum()
    return sim


def cal_sim_metric_self(goal_anomaly, goal_hazard):
    assert len(goal_anomaly) == len(goal_hazard)
    
    cc_list = []
    sim_list = []
    
    for i in range(len(goal_anomaly)):
        attn_1 = goal_anomaly[i]
        attn_2 = goal_hazard[i]
        
        cc_list.append(corr_coeff(attn_1, attn_2))
        sim_list.append(sim(attn_1, attn_2))

    final_cc = np.mean(cc_list)
    final_sim = np.mean(sim_list)

    return final_cc, final_sim

def cal_sim_metric_mean(goal_anomaly, goal_hazard):
    attn_mean_1 = np.sum(np.stack(goal_anomaly), axis=0)
    attn_mean_2 = np.sum(np.stack(goal_hazard), axis=0)
    attn_mean_1 = attn_mean_1 / attn_mean_1.sum()
    attn_mean_2 = attn_mean_2 / attn_mean_2.sum()
    
    cc = corr_coeff(attn_mean_1, attn_mean_2)
    sim_data = sim(attn_mean_1, attn_mean_2)

    return cc, sim_data
    

def cal_sim_metric_cross(goal_anomaly, goal_hazard):
    all_cc = []
    all_sim = []
    for i in range(len(goal_anomaly)):
        attn_1 = goal_anomaly[i]
        for j in range(len(goal_hazard)):
            attn_2 = goal_hazard[j]
            
            cc = corr_coeff(attn_1, attn_2)
            sim = sim(attn_1, attn_2)

            all_cc.append(cc)
            all_sim.append(sim)
            
    final_cc = np.mean(all_cc)
    final_sim = np.mean(all_sim)

    return final_cc, final_sim            

def cal_sim_metric_avg_1(goal_anomaly, goal_hazard):
    all_cc = []
    all_sim = []
    for i in range(len(goal_anomaly)):
        attn_1 = goal_anomaly[i]
        attn_1_cc = []
        attn_1_sim = []
        for j in range(len(goal_hazard)):
            attn_2 = goal_hazard[j]
            if ((attn_1 - attn_2) * (attn_1 - attn_2)).sum() == 0:
                continue
            cc_data = corr_coeff(attn_1, attn_2)
            sim_data = sim(attn_1, attn_2)

            attn_1_cc.append(cc_data)
            attn_1_sim.append(sim_data)
        
        all_cc.append(np.mean(attn_1_cc))
        all_sim.append(np.mean(attn_1_sim))
        
    final_cc = np.mean(all_cc)
    final_sim = np.mean(all_sim)

    return final_cc, final_sim            

def cal_sim_metric_avg_2(goal_anomaly, goal_hazard):
    all_cc = []
    all_sim = []
    
    attn_mean_2 = np.sum(np.stack(goal_hazard), axis=0)
    attn_mean_2 = attn_mean_2 / attn_mean_2.sum()

    for i in range(len(goal_anomaly)):
        attn_1 = goal_anomaly[i]

        cc_data = corr_coeff(attn_1, attn_mean_2)
        sim_data = sim(attn_1, attn_mean_2)

        # print(sim_data)
        all_cc.append(cc_data)
        all_sim.append(sim_data)
            
    final_cc = np.mean(all_cc)
    final_sim = np.mean(all_sim)

    return final_cc, final_sim            

def cal_sim_metric_avg_3(goal_anomaly, goal_hazard):
    '''avg(所有图avg（每张图和自己均值（所有goal、算法）的相似度）) vs avg(所有图avg（每张图和别的图集均值（所有goal、算法）的相似度) ）'''
    assert len(goal_anomaly[0]) == len(goal_hazard[0])
    rgb_image_count = len(goal_anomaly[0])
    single_image_feat_count = len(goal_anomaly)

    all_cc = []
    all_sim = []

    for image_idx in range(rgb_image_count):
        attn_image_idx_list = [goal_hazard[i][image_idx] for i in range(single_image_feat_count)]
        attn_mean_image_idx = np.sum(np.stack(attn_image_idx_list), axis=0)
        attn_mean_image_idx = attn_mean_image_idx / attn_mean_image_idx.sum()
        
        cc_image = []
        sim_image = []
        for feat_idx in range(single_image_feat_count):
            attn_image_feat = goal_anomaly[feat_idx][image_idx]
    
            cc_data = corr_coeff(attn_image_feat, attn_mean_image_idx)
            sim_data = sim(attn_image_feat, attn_mean_image_idx)

            cc_image.append(cc_data)
            sim_image.append(sim_data)
            
        all_cc.append(np.mean(cc_image))
        all_sim.append(np.mean(sim_image))

    final_cc = np.mean(all_cc)
    final_sim = np.mean(all_sim)

    return final_cc, final_sim            

def cal_sim_metric_avg_4(goal_anomaly, goal_hazard):
    '''avg(所有图 avg（每张图和自己均值（所有goal、所有人）的相似度）) vs avg(所有图avg（每张图和别的图集均值（所有goal、所有人）的相似度) ） 所有goal'''
        
    assert len(goal_anomaly[0]) == len(goal_hazard[0])
    assert len(goal_anomaly[0][0]) == len(goal_hazard[0][0])
    single_image_feat_count_human = len(goal_anomaly)
    single_image_feat_count_goal = len(goal_anomaly[0])
    rgb_image_count = len(goal_anomaly[0][0])

    all_cc = []
    all_sim = []

    for image_idx in range(rgb_image_count):
        attn_image_idx_list = [goal_hazard[i][j][image_idx] for j in range(single_image_feat_count_goal) for i in range(single_image_feat_count_human)]
        assert len(attn_image_idx_list) == single_image_feat_count_goal * single_image_feat_count_human
        
        attn_mean_image_idx = np.sum(np.stack(attn_image_idx_list), axis=0)
        attn_mean_image_idx = attn_mean_image_idx / attn_mean_image_idx.sum()
        
        cc_image = []
        sim_image = []
        for feat_idx_human in range(single_image_feat_count_human):
            for feat_idx_goal in range(single_image_feat_count_goal):
                attn_image_feat = goal_anomaly[feat_idx_human][feat_idx_goal][image_idx]
    
                cc_data = corr_coeff(attn_image_feat, attn_mean_image_idx)
                sim_data = sim(attn_image_feat, attn_mean_image_idx)

                cc_image.append(cc_data)
                sim_image.append(sim_data)
            
        all_cc.append(np.mean(cc_image))
        all_sim.append(np.mean(sim_image))

    final_cc = np.mean(all_cc)
    final_sim = np.mean(all_sim)

    return final_cc, final_sim            

def cal_sim_metric_avg_5(goal_anomaly, goal_hazard):
    '''avg(所有图 avg（每张图和自己均值（所有goal、所有人）的相似度）) vs avg(所有图avg（每张图和别的图集均值（所有goal、所有人）的相似度) ） goal 3选1''' 
    assert len(goal_anomaly[0]) == len(goal_hazard[0])
    rgb_image_count = len(goal_anomaly[0])
    single_image_feat_count = len(goal_anomaly)

    all_cc = []
    all_sim = []

    for image_idx in range(rgb_image_count):
        attn_image_idx_list = [goal_hazard[i][image_idx] for i in range(single_image_feat_count)]
        attn_mean_image_idx = np.sum(np.stack(attn_image_idx_list), axis=0)
        attn_mean_image_idx = attn_mean_image_idx / attn_mean_image_idx.sum()
        
        cc_image = []
        sim_image = []
        for feat_idx in range(single_image_feat_count):
            attn_image_feat = goal_anomaly[feat_idx][image_idx]
    
            cc_data = corr_coeff(attn_image_feat, attn_mean_image_idx)
            sim_data = sim(attn_image_feat, attn_mean_image_idx)

            cc_image.append(cc_data)
            sim_image.append(sim_data)
            
        all_cc.append(np.mean(cc_image))
        all_sim.append(np.mean(sim_image))

    final_cc = np.mean(all_cc)
    final_sim = np.mean(all_sim)

    return final_cc, final_sim            

def cal_sim_metric_avg_6(goal_anomaly, goal_hazard):        
    assert len(goal_anomaly[0]) == len(goal_hazard[0])
    assert len(goal_anomaly[0][0]) == len(goal_hazard[0][0])
    single_image_feat_count_human = len(goal_anomaly)
    single_image_feat_count_data = len(goal_anomaly[0])
    rgb_image_count = len(goal_anomaly[0][0])

    all_cc = []
    all_sim = []

    for data_idx in range(single_image_feat_count_data):
        for image_idx in range(rgb_image_count):
            attn_image_idx_list = [goal_hazard[i][data_idx][image_idx] for i in range(single_image_feat_count_human)]
            assert len(attn_image_idx_list) == single_image_feat_count_human
            
            attn_mean_image_idx = np.sum(np.stack(attn_image_idx_list), axis=0)
            attn_mean_image_idx = attn_mean_image_idx / attn_mean_image_idx.sum()
            
            cc_image = []
            sim_image = []
            for feat_idx_human in range(single_image_feat_count_human):
                attn_image_feat = goal_anomaly[feat_idx_human][data_idx][image_idx]
    
                cc_data = corr_coeff(attn_image_feat, attn_mean_image_idx)
                sim_data = sim(attn_image_feat, attn_mean_image_idx)

                cc_image.append(cc_data)
                sim_image.append(sim_data)
                
            all_cc.append(np.mean(cc_image))
            all_sim.append(np.mean(sim_image))

    final_cc = np.mean(all_cc)
    final_sim = np.mean(all_sim)

    return final_cc, final_sim            

def cal_sim_metric_avg_7(goal_anomaly, goal_hazard):
    assert len(goal_anomaly[0]) == len(goal_hazard[0])
    rgb_image_count = len(goal_anomaly[0])
    single_image_feat_count = len(goal_anomaly)

    all_cc = []
    all_sim = []

    for image_idx in range(rgb_image_count):
        attn_image_idx_list = [goal_hazard[i][image_idx] for i in range(single_image_feat_count)]
        attn_mean_image_idx = np.sum(np.stack(attn_image_idx_list), axis=0)
        attn_mean_image_idx = attn_mean_image_idx / attn_mean_image_idx.sum()
        
        cc_image = []
        sim_image = []
        for feat_idx in range(single_image_feat_count):
            attn_image_feat = goal_anomaly[feat_idx][image_idx]
    
            cc_data = corr_coeff(attn_image_feat, attn_mean_image_idx)
            sim_data = sim(attn_image_feat, attn_mean_image_idx)

            cc_image.append(cc_data)
            sim_image.append(sim_data)
            
        all_cc.append(np.mean(cc_image))
        all_sim.append(np.mean(sim_image))

    final_cc = np.mean(all_cc)
    final_sim = np.mean(all_sim)

    return final_cc, final_sim            

COUNT=0
all_cc_list = []
all_name = []
# fig, axs = plt.subplots(2, 2)
file_name = 'r_p_file.xlsx'
writer = pd.ExcelWriter(file_name, mode='w', engine='openpyxl')

def cal_sim_metric_avg_8(goal_anomaly, goal_hazard, name):
    '''算法vs人： avg(所有data的所有图 avg（每张图和自己均值（所有goal、所有人）的相似度）) vs avg(所有data的所有图 avg（每张图和别的图集均值（所有goal、所有人）的相似度) ）'''
    
    assert len(goal_anomaly[0]) == len(goal_hazard[0])
    assert len(goal_anomaly[0][0]) == len(goal_hazard[0][0])
    assert len(goal_anomaly[0][0][0]) == len(goal_hazard[0][0][0])
    single_image_feat_count_human_0 = len(goal_anomaly)
    single_image_feat_count_human_1 = len(goal_hazard)
    single_image_feat_count_data = len(goal_anomaly[0])
    single_image_feat_count_goal = len(goal_anomaly[0][0])
    rgb_image_count = len(goal_anomaly[0][0][0])

    all_cc = []
    all_p = []
    all_sim = []
    all_alg_filename = []
    all_human_filename = []

    for data_idx in range(single_image_feat_count_data):
        for image_idx in range(rgb_image_count):
            attn_image_idx_list = [goal_hazard[i][data_idx][j][image_idx] for j in range(single_image_feat_count_goal) for i in range(single_image_feat_count_human_1)]
            assert len(attn_image_idx_list) == single_image_feat_count_goal * single_image_feat_count_human_1
            
            attn_mean_image_idx = np.sum(np.stack(attn_image_idx_list), axis=0)
            if attn_mean_image_idx.sum() == 0:
                print(1)
            attn_mean_image_idx = attn_mean_image_idx / attn_mean_image_idx.sum()
            
            cc_image = []
            p_image = []
            sim_image = []
            for feat_idx_human in range(single_image_feat_count_human_0):
                for feat_idx_goal in range(single_image_feat_count_goal):
                    attn_image_feat = goal_anomaly[feat_idx_human][data_idx][feat_idx_goal][image_idx]
                    # alg_filename = alg_data_filename_list_goal[feat_idx_human][data_idx][feat_idx_goal][image_idx]
                    # human_filename = human_data_filename_list_goal[feat_idx_human][data_idx][feat_idx_goal][image_idx]
                    
                    cc_p_data = corr_coeff(attn_image_feat, attn_mean_image_idx)
                    cc_data = cc_p_data[0]
                    p_data = cc_p_data[1]
                    sim_data = sim(attn_image_feat, attn_mean_image_idx)

                    cc_image.append(cc_data)
                    p_image.append(p_data)
                    sim_image.append(sim_data)
                
            all_cc.append(np.mean(cc_image))
            all_p.append(np.mean(p_image))
            all_sim.append(np.mean(sim_image))
            
            # all_alg_filename.append(alg_filename)
            # all_alg_filename.append(human_filename)

    final_cc = np.mean(all_cc)
    final_p = np.mean(all_p)
    final_sim = np.mean(all_sim)

    global COUNT
    global RESIZE_SIZE
    global all_cc_list
    global all_name
    print(COUNT)
    all_cc_list.append(all_cc)
    
    mean = np.mean(all_cc)  # 计算均值
    variance = np.var(all_cc)  # 计算方差

    # print("均值：", mean)
    # print("方差：", variance)
    
    new_name = name + '_mean_' + str(mean)[:5] + '_var_' + str(variance)[:5]
    all_name.append(new_name)
    
    n, bins, patches = plt.hist(all_cc, bins=20, range=(-1.,1.), label=new_name)
    for i in range(len(n)):
        r_size = RESIZE_SIZE * RESIZE_SIZE
        dist = scipy.stats.beta(r_size/2 - 1, r_size/2 - 1, loc=-1, scale=2)
        r = bins[i]+(bins[1]-bins[0])/2
        p = 2*dist.cdf(-abs(r))
        if n[i] > 0:
            plt.text(r, n[i]*1.01, '%.4f' % p, ha='center', va= 'bottom', fontsize=7)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join('hist_fig', '%02d'% COUNT + '_%s.png' % new_name))
    plt.clf()
    
    global writer
    header = ['cc', 'p']
    df = pd.DataFrame(np.stack([all_cc, all_p]).transpose(1,0),columns=header)
    name_split = name.split('_')
    if len(name_split) > 4:
        sheet_name = '_'.join([name_split[1], name_split[3], name_split[5], name_split[7]])
    else:
        sheet_name = name
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(sheet_name)
    
    COUNT+=1
    
    return final_cc, final_sim, final_p     

def cal_sim_metric(goal_anomaly, goal_hazard, cal_method, name):
    if cal_method == 'self':
        return cal_sim_metric_self(goal_anomaly, goal_hazard)
    elif cal_method == 'mean':
        return cal_sim_metric_mean(goal_anomaly, goal_hazard)
    elif cal_method == 'cross':
        return cal_sim_metric_cross(goal_anomaly, goal_hazard)
    elif cal_method == 'avg_1': # 算法： avg(每张图和自己数据集的9张图分别比较的相似度) vs avg(每张图和其他数据集的PER_BATCH_IMAGE_COUNT张图比较的相似度）
        return cal_sim_metric_avg_1(goal_anomaly, goal_hazard)
    elif cal_method == 'avg_2': # 算法：avg(每张图和自己的集合均值的相似度) vs avg(每张图和别的集合均值的相似度) 
        return cal_sim_metric_avg_2(goal_anomaly, goal_hazard)
    elif cal_method == 'avg_3': # 算法：avg(所有图avg（每张图和自己均值（所有goal、算法）的相似度）) vs avg(所有图avg（每张图和别的图集均值（所有goal、算法）的相似度) ）
        return cal_sim_metric_avg_3(goal_anomaly, goal_hazard)
    elif cal_method == 'avg_4': # 人： avg(所有图 avg（每张图和自己均值（所有goal、所有人）的相似度）) vs avg(所有图avg（每张图和别的图集均值（所有goal、所有人）的相似度) ） 所有goal
        return cal_sim_metric_avg_4(goal_anomaly, goal_hazard)
    elif cal_method == 'avg_5': # 人： avg(所有图 avg（每张图和自己均值（所有goal、所有人）的相似度）) vs avg(所有图avg（每张图和别的图集均值（所有goal、所有人）的相似度) ）goal 3选1
        return cal_sim_metric_avg_5(goal_anomaly, goal_hazard)
    elif cal_method == 'avg_6': # 人： avg(每个goal的所有图 avg(每张图 和 这张图均值（所有人）））所有data
        return cal_sim_metric_avg_6(goal_anomaly, goal_hazard)
    elif cal_method == 'avg_7': # 人： avg(每个goal的所有图 avg(每张图 和 这张图均值（所有人）））datas3选1
        return cal_sim_metric_avg_7(goal_anomaly, goal_hazard)
    elif cal_method == 'avg_8': # 算法vs人： avg(所有data的所有图 avg（每张图和自己均值（所有goal、所有人）的相似度）) vs avg(所有data的所有图 avg（每张图和别的图集均值（所有goal、所有人）的相似度) ）
        return cal_sim_metric_avg_8(goal_anomaly, goal_hazard, name)
    else:
        assert False, 'cal_method error!'

RESIZE_SIZE = 224
def normlize_each_attn_map(attn_list):
    '''normilize each attn map'''
    global RESIZE_SIZE
    new_attn_list = []
    for i in range(len(attn_list)):
        attn = attn_list[i].copy()
        attn = cv2.resize(attn, dsize=(RESIZE_SIZE,RESIZE_SIZE))
        # if attn.sum() == 0:
        #     continue
        attn = attn / (attn.sum()+1e-20)
        # attn_list[i] = attn
        new_attn_list.append(attn)

    return new_attn_list



def main():
    for alg_idx in range(2):
        CAL_METHOD = 'avg_8'
        
        SEPERATE_DATA_OR_NOT = False # give each data/goal/alg a seperate cc
        
        DATA_SPLIT_VERSION = 'v3'

        ############################ alg
        # DATA_PATH_ALG = '/DATA_EDS/lpf/gaze/cal_sim/alg_data/data_v2'
        alg_old_new = ['old', 'new']
        DATA_PATH_ALG = '/DATA_EDS/lpf/gaze/cal_sim/v3_alg_data/alg_'+alg_old_new[alg_idx] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ''' load data '''
        goal_anomaly_path = os.path.join(DATA_PATH_ALG, 'Anomaly')
        goal_hazard_path = os.path.join(DATA_PATH_ALG, 'Hazard')
        goal_turn_path = os.path.join(DATA_PATH_ALG, 'Turn')

        goal_anomaly_file_list = sorted([os.path.join(goal_anomaly_path, item) for item in os.listdir(goal_anomaly_path) if item.endswith('.npy')])
        goal_hazard_file_list = sorted([os.path.join(goal_hazard_path, item) for item in os.listdir(goal_hazard_path) if item.endswith('.npy')])
        goal_turn_file_list = sorted([os.path.join(goal_turn_path, item) for item in os.listdir(goal_turn_path) if item.endswith('.npy')])

        goal_anomaly = [np.load(item) for item in goal_anomaly_file_list]
        goal_hazard = [np.load(item) for item in goal_hazard_file_list]
        goal_turn = [np.load(item) for item in goal_turn_file_list]
        
        '''normalize each attn map'''
        goal_anomaly = normlize_each_attn_map(goal_anomaly)
        goal_hazard = normlize_each_attn_map(goal_hazard)
        goal_turn = normlize_each_attn_map(goal_turn)

        '''split data'''
        if DATA_SPLIT_VERSION == 'v2':
            attn_list_data_anomaly__goal_anomaly = goal_anomaly[1:31] # data anomaly
            attn_list_data_anomaly__goal_hazard = goal_hazard[1:31]
            attn_list_data_anomaly__goal_turn = goal_turn[1:31]
                
            attn_list_data_hazard__goal_anomaly = goal_anomaly[49:75]+goal_anomaly[86:] # data drama
            attn_list_data_hazard__goal_hazard = goal_hazard[49:75]+goal_hazard[86:]
            attn_list_data_hazard__goal_turn = goal_turn[49:75]+goal_turn[86:]

            attn_list_data_turn__goal_anomaly = goal_anomaly[0:1]+goal_anomaly[31:49]+goal_anomaly[75:86] # data oia
            attn_list_data_turn__goal_hazard = goal_hazard[0:1]+goal_hazard[31:49]+goal_hazard[75:86]
            attn_list_data_turn__goal_turn = goal_turn[0:1]+goal_turn[31:49]+goal_turn[75:86]
        elif DATA_SPLIT_VERSION == 'v3':
            attn_list_data_anomaly__goal_anomaly = goal_anomaly[1:31] # data anomaly
            attn_list_data_anomaly__goal_hazard = goal_hazard[1:31]
            attn_list_data_anomaly__goal_turn = goal_turn[1:31]
                
            attn_list_data_hazard__goal_anomaly = goal_anomaly[44:70]+goal_anomaly[86:] # data drama
            attn_list_data_hazard__goal_hazard = goal_hazard[44:70]+goal_hazard[86:]
            attn_list_data_hazard__goal_turn = goal_turn[44:70]+goal_turn[86:]
            
            attn_list_data_turn__goal_anomaly = goal_anomaly[0:1]+goal_anomaly[31:44]+goal_anomaly[70:86] # data oia
            attn_list_data_turn__goal_hazard = goal_hazard[0:1]+goal_hazard[31:44]+goal_hazard[70:86]
            attn_list_data_turn__goal_turn = goal_turn[0:1]+goal_turn[31:44]+goal_turn[70:86]
            
        else:
            # v1
            # attn_list_data_anomaly__goal_anomaly = goal_anomaly[:30] # data anomaly
            # attn_list_data_anomaly__goal_hazard = goal_hazard[:30]
            # attn_list_data_anomaly__goal_turn = goal_turn[:30]
                
            # attn_list_data_hazard__goal_anomaly = goal_anomaly[40:66]+goal_anomaly[86:] # data drama
            # attn_list_data_hazard__goal_hazard = goal_hazard[40:66]+goal_hazard[86:]
            # attn_list_data_hazard__goal_turn = goal_turn[40:66]+goal_turn[86:]

            # attn_list_data_turn__goal_anomaly = goal_anomaly[30:40]+goal_anomaly[66:86] # data oia
            # attn_list_data_turn__goal_hazard = goal_hazard[30:40]+goal_hazard[66:86]
            # attn_list_data_turn__goal_turn = goal_turn[30:40]+goal_turn[66:86]
            assert False, 'DATA_SPLIT_VERSION error!'

        '''all data'''
        all_list_goal = [
            [attn_list_data_anomaly__goal_anomaly, attn_list_data_anomaly__goal_hazard, attn_list_data_anomaly__goal_turn],
            [attn_list_data_hazard__goal_anomaly, attn_list_data_hazard__goal_hazard, attn_list_data_hazard__goal_turn],
            [attn_list_data_turn__goal_anomaly, attn_list_data_turn__goal_hazard, attn_list_data_turn__goal_turn],
        ]
            
        all_list_data = {
            'goal_anomaly': [attn_list_data_anomaly__goal_anomaly, attn_list_data_hazard__goal_anomaly, attn_list_data_turn__goal_anomaly],
            'goal_hazard': [attn_list_data_anomaly__goal_hazard, attn_list_data_hazard__goal_hazard, attn_list_data_turn__goal_hazard],
            'goal_turn': [attn_list_data_anomaly__goal_turn, attn_list_data_hazard__goal_turn, attn_list_data_turn__goal_turn],
        }

        alg_data_list_goal = [copy.deepcopy(all_list_goal)]
        alg_data_list_data = [copy.deepcopy(all_list_data)]
        

        for human_idx in range(2):

            ############################ human
            exp_list = ['old_mean', 'new_mean'] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # DATA_PATH_HUMAN = os.path.join('/data/jinbu/gaze_compare_8_15', exp_list[0]) # only for ALG_OR_HUMAN == 'human'
            DATA_PATH_HUMAN = os.path.join('/data18/jinbu/gaze_visualize_utils/gaze_compare_personalised_t0_t1_35_hazard_anomaly', exp_list[human_idx]) # only for ALG_OR_HUMAN == 'human'
                
            # data_type = ['_fixation_only_200_20_0.3_0.1', '_raw_200_20_0.3_0.1', '_removefix_200_20_0.3_0.1']   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            data_type = ['_fixation_only_200_35_0.0_1.0']   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            data_dir = sorted([os.path.join(DATA_PATH_HUMAN,f) for f in os.listdir(DATA_PATH_HUMAN) if f.endswith(data_type[0])])
            
            human_data_list_goal = []
            human_data_list_data = []
            
            human_data_filename_list_goal = []
            for data_dir_item in data_dir:
                ''' load data '''
                goal_anomaly_path = os.path.join(data_dir_item, 'Anomaly')
                goal_hazard_path = os.path.join(data_dir_item, 'Hazard')
                goal_turn_path = os.path.join(data_dir_item, 'Turn')
        
                ####################### NOTE #######################
                goal_turn_path = goal_turn_path.replace('gaze_compare_personalised_t0_t1_35_hazard_anomaly', 'gaze_compare_personalised_t0_t1_65_oia').replace('200_35', '200_65')
                ####################### NOTE #######################

                goal_anomaly_file_list = sorted([os.path.join(goal_anomaly_path, item) for item in os.listdir(goal_anomaly_path) if item.endswith('.npy')])
                goal_hazard_file_list = sorted([os.path.join(goal_hazard_path, item) for item in os.listdir(goal_hazard_path) if item.endswith('.npy')])
                goal_turn_file_list = sorted([os.path.join(goal_turn_path, item) for item in os.listdir(goal_turn_path) if item.endswith('.npy')])

                goal_anomaly = [np.load(item) for item in goal_anomaly_file_list]
                goal_hazard = [np.load(item) for item in goal_hazard_file_list]
                goal_turn = [np.load(item) for item in goal_turn_file_list]
                
                '''normalize each attn map'''
                goal_anomaly = normlize_each_attn_map(goal_anomaly)
                goal_hazard = normlize_each_attn_map(goal_hazard)
                goal_turn = normlize_each_attn_map(goal_turn)

                '''split data'''
                if DATA_SPLIT_VERSION == 'v2':
                    attn_list_data_anomaly__goal_anomaly = goal_anomaly[1:31] # data anomaly
                    attn_list_data_anomaly__goal_hazard = goal_hazard[1:31]
                    attn_list_data_anomaly__goal_turn = goal_turn[1:31]
                        
                    attn_list_data_hazard__goal_anomaly = goal_anomaly[49:75]+goal_anomaly[86:] # data drama
                    attn_list_data_hazard__goal_hazard = goal_hazard[49:75]+goal_hazard[86:]
                    attn_list_data_hazard__goal_turn = goal_turn[49:75]+goal_turn[86:]

                    attn_list_data_turn__goal_anomaly = goal_anomaly[0:1]+goal_anomaly[31:49]+goal_anomaly[75:86] # data oia
                    attn_list_data_turn__goal_hazard = goal_hazard[0:1]+goal_hazard[31:49]+goal_hazard[75:86]
                    attn_list_data_turn__goal_turn = goal_turn[0:1]+goal_turn[31:49]+goal_turn[75:86]
                elif DATA_SPLIT_VERSION == 'v3':
                    attn_list_data_anomaly__goal_anomaly = goal_anomaly[1:31] # data anomaly
                    attn_list_data_anomaly__goal_hazard = goal_hazard[1:31]
                    attn_list_data_anomaly__goal_turn = goal_turn[1:31]
                        
                    attn_list_data_hazard__goal_anomaly = goal_anomaly[44:70]+goal_anomaly[86:] # data drama
                    attn_list_data_hazard__goal_hazard = goal_hazard[44:70]+goal_hazard[86:]
                    attn_list_data_hazard__goal_turn = goal_turn[44:70]+goal_turn[86:]
                    
                    attn_list_data_turn__goal_anomaly = goal_anomaly[0:1]+goal_anomaly[31:44]+goal_anomaly[70:86] # data oia
                    attn_list_data_turn__goal_hazard = goal_hazard[0:1]+goal_hazard[31:44]+goal_hazard[70:86]
                    attn_list_data_turn__goal_turn = goal_turn[0:1]+goal_turn[31:44]+goal_turn[70:86]

                    # filename_list generation
                    filename_list_data_anomaly__goal_anomaly = goal_anomaly_file_list[1:31] # data anomaly
                    filename_list_data_anomaly__goal_hazard = goal_hazard_file_list[1:31]
                    filename_list_data_anomaly__goal_turn = goal_turn_file_list[1:31]
                    
                    filename_list_data_hazard__goal_anomaly = goal_anomaly_file_list[44:70]+goal_anomaly_file_list[86:] # data drama
                    filename_list_data_hazard__goal_hazard = goal_hazard_file_list[44:70]+goal_hazard_file_list[86:]
                    filename_list_data_hazard__goal_turn = goal_turn_file_list[44:70]+goal_turn_file_list[86:]
                    
                    filename_list_data_turn__goal_anomaly = goal_anomaly_file_list[0:1]+goal_anomaly_file_list[31:44]+goal_anomaly_file_list[70:86] # data oia
                    filename_list_data_turn__goal_hazard = goal_hazard_file_list[0:1]+goal_hazard_file_list[31:44]+goal_hazard_file_list[70:86]
                    filename_list_data_turn__goal_turn = goal_turn_file_list[0:1]+goal_turn_file_list[31:44]+goal_turn_file_list[70:86]

                else:
                    # v1
                    # attn_list_data_anomaly__goal_anomaly = goal_anomaly[:30] # data anomaly
                    # attn_list_data_anomaly__goal_hazard = goal_hazard[:30]
                    # attn_list_data_anomaly__goal_turn = goal_turn[:30]
                        
                    # attn_list_data_hazard__goal_anomaly = goal_anomaly[40:66]+goal_anomaly[86:] # data drama
                    # attn_list_data_hazard__goal_hazard = goal_hazard[40:66]+goal_hazard[86:]
                    # attn_list_data_hazard__goal_turn = goal_turn[40:66]+goal_turn[86:]

                    # attn_list_data_turn__goal_anomaly = goal_anomaly[30:40]+goal_anomaly[66:86] # data oia
                    # attn_list_data_turn__goal_hazard = goal_hazard[30:40]+goal_hazard[66:86]
                    # attn_list_data_turn__goal_turn = goal_turn[30:40]+goal_turn[66:86]
                    assert False, 'DATA_SPLIT_VERSION error!'
                
                '''all data'''
                all_list_goal = [
                    [attn_list_data_anomaly__goal_anomaly, attn_list_data_anomaly__goal_hazard, attn_list_data_anomaly__goal_turn],
                    [attn_list_data_hazard__goal_anomaly, attn_list_data_hazard__goal_hazard, attn_list_data_hazard__goal_turn],
                    [attn_list_data_turn__goal_anomaly, attn_list_data_turn__goal_hazard, attn_list_data_turn__goal_turn],
                ]

                all_filename_list_goal = [
                    [filename_list_data_anomaly__goal_anomaly, filename_list_data_anomaly__goal_hazard, filename_list_data_anomaly__goal_turn],
                    [filename_list_data_hazard__goal_anomaly, filename_list_data_hazard__goal_hazard, filename_list_data_hazard__goal_turn],
                    [filename_list_data_turn__goal_anomaly, filename_list_data_turn__goal_hazard, filename_list_data_turn__goal_turn],
                ]

                all_list_data = {
                    'goal_anomaly': [attn_list_data_anomaly__goal_anomaly, attn_list_data_hazard__goal_anomaly, attn_list_data_turn__goal_anomaly],
                    'goal_hazard': [attn_list_data_anomaly__goal_hazard, attn_list_data_hazard__goal_hazard, attn_list_data_turn__goal_hazard],
                    'goal_turn': [attn_list_data_anomaly__goal_turn, attn_list_data_hazard__goal_turn, attn_list_data_turn__goal_turn],
                }
            
                human_data_list_goal.append(all_list_goal)
                human_data_list_data.append(all_list_data)
                
                human_data_filename_list_goal.append(all_filename_list_goal)
            
            alg_and_human = [alg_data_list_goal, human_data_list_goal]
            
            nonzero_alg_data_list_goal = []
            nonzero_human_data_list_goal = []
            for humandata_idx in range(len(alg_data_list_goal)):
                alg_data_list_humancount = []
                human_data_list_humancount = []
                for data_idx in range(len(alg_data_list_goal[humandata_idx])):
                    alg_data_list_humancount_data = []
                    human_data_list_humancount_data = []
                    for goal_idx in range(len(alg_data_list_goal[humandata_idx][data_idx])):
                        alg_data_list_humancount_data_goal = []
                        human_data_list_humancount_data_goal = []
                        for image_idx in range(len(alg_data_list_goal[humandata_idx][data_idx][goal_idx])):
                            if alg_data_list_goal[humandata_idx][data_idx][goal_idx][image_idx].sum() != 0 and human_data_list_goal[humandata_idx][data_idx][goal_idx][image_idx].sum() != 0:
                                alg_data_list_humancount_data_goal.append(copy.deepcopy(alg_data_list_goal[humandata_idx][data_idx][goal_idx][image_idx]))
                                human_data_list_humancount_data_goal.append(copy.deepcopy(human_data_list_goal[humandata_idx][data_idx][goal_idx][image_idx]))
                        alg_data_list_humancount_data.append(alg_data_list_humancount_data_goal)
                        human_data_list_humancount_data.append(human_data_list_humancount_data_goal)
                    alg_data_list_humancount.append(alg_data_list_humancount_data)
                    human_data_list_humancount.append(human_data_list_humancount_data)
                nonzero_alg_data_list_goal.append(alg_data_list_humancount)
                nonzero_human_data_list_goal.append(human_data_list_humancount)
                
            alg_and_human = [nonzero_alg_data_list_goal, nonzero_human_data_list_goal] # 1,3,3,30 

            different_data = True #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
            data_type = ['anomaly', 'hazard', 'turn']
            different_goal = True #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
            goal_type = ['anomaly', 'hazard', 'turn']

            person_by_person = False


            print('*************************************************************')
            print('alg_idx: ', alg_old_new[alg_idx], 'human_idx: ', exp_list[human_idx])

            if CAL_METHOD == 'avg_8':
                if different_data and (not different_goal):
                    for data_idx in range(len(data_type)):
                        print('====================')
                        print('data type: ', data_type[data_idx])
                        
                        diff_data_alg_and_human = split_diff_data(alg_and_human, data_idx)
                
                        cc_matrix = np.zeros([2,2])
                        for i in range(len(diff_data_alg_and_human)):
                            for j in range(len(diff_data_alg_and_human)):
                                cc, sim = cal_sim_metric(diff_data_alg_and_human[i], diff_data_alg_and_human[j], CAL_METHOD)
                                cc_matrix[i,j] = cc
                        print('cc_matrix: ')
                        print(cc_matrix.reshape([2,2]))

                elif (not different_data) and different_goal:
                    for goal_idx in range(len(goal_type)):
                        print('====================')
                        print('goal type: ', goal_type[goal_idx])
                        
                        diff_goal_alg_and_human = split_diff_goal(alg_and_human, goal_idx)
                
                        cc_matrix = np.zeros([2,2])
                        for i in range(len(diff_goal_alg_and_human)):
                            for j in range(len(diff_goal_alg_and_human)):
                                cc, sim = cal_sim_metric(diff_goal_alg_and_human[i], diff_goal_alg_and_human[j], CAL_METHOD)
                                cc_matrix[i,j] = cc
                        print('cc_matrix: ')
                        print(cc_matrix.reshape([2,2]))

                elif different_data and different_goal:
                    for data_idx in range(len(data_type)):
                        print('====================')
                        print('data type: ', data_type[data_idx])
                        diff_data_alg_and_human = split_diff_data(alg_and_human, data_idx)

                        for goal_idx in range(len(goal_type)):
                            print('----------------------')
                            print('goal type: ', goal_type[goal_idx])
                            
                            diff_goal_alg_and_human = split_diff_goal(diff_data_alg_and_human, goal_idx)

                    
                            # cc_matrix = np.zeros([2,2])
                            # p_matrix = np.zeros([2,2])
                            # for i in range(len(diff_goal_alg_and_human)):
                            #     for j in range(len(diff_goal_alg_and_human)):
                            #         cc, sim, p = cal_sim_metric(diff_goal_alg_and_human[i], diff_goal_alg_and_human[j], CAL_METHOD)
                            #         cc_matrix[i,j] = cc
                            #         p_matrix[i,j] = p
                            # print('cc_matrix: ')
                            # print(cc_matrix.reshape([2,2]))
                            # print('p_matrix: ')
                            # print(p_matrix.reshape([2,2]))
                
                            name = 'alg_' + alg_old_new[alg_idx] + '_human_' + exp_list[human_idx][:3] + '_data_' + data_type[data_idx] + '_goal_' + goal_type[goal_idx]
                            cc, sim, p = cal_sim_metric(diff_goal_alg_and_human[0], diff_goal_alg_and_human[1], CAL_METHOD, name)
                            print('cc: ', cc, 'p: ', p)

                else:
                    if person_by_person:
                        for human_idx in range(len(alg_and_human[1])):
                            print('====================')
                            print('person: ', human_idx)
                            new_alg_and_human = []
                            new_alg_and_human.append(alg_and_human[0])
                            new_alg_and_human.append([alg_and_human[1][human_idx]])
                                                
                            cc_matrix = np.zeros([2,2])
                            for i in range(len(new_alg_and_human)):
                                for j in range(len(new_alg_and_human)):
                                    cc, sim = cal_sim_metric(new_alg_and_human[i], new_alg_and_human[j], CAL_METHOD)
                                    cc_matrix[i,j] = cc
                            print('cc_matrix: ')
                            print(cc_matrix.reshape([2,2]))
                    else:
                        # cc_matrix = np.zeros([2,2])
                        # p_matrix = np.zeros([2,2])
                        # print('====================')
                        # for i in range(len(alg_and_human)):
                        #     for j in range(len(alg_and_human)):
                        #         cc, sim, p = cal_sim_metric(alg_and_human[i], alg_and_human[j], CAL_METHOD)
                        #         cc_matrix[i,j] = cc
                        #         p_matrix[i,j] = p
                        # print('cc_matrix: ')
                        # print(cc_matrix.reshape([2,2]))
                        # print('p_matrix: ')
                        # print(p_matrix.reshape([2,2]))

                        name = 'alg_' + alg_old_new[alg_idx] + '_human_' + exp_list[human_idx][:3]
                        cc, sim, p = cal_sim_metric(alg_and_human[0], alg_and_human[1], alg_data_filename_list_goal, human_data_filename_list_goal, CAL_METHOD, name)
                        print('cc: ', cc, 'p: ', p)
    global all_cc_list
    global all_name
    plt.hist(all_cc_list, bins=20, range=(-1.,1.), label=all_name)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join('hist_fig', 'all.png'))
    plt.clf()

    global writer
    writer.close()
         

def split_diff_data(alg_and_human, i):
    alg_data = alg_and_human[0]
    human_data = alg_and_human[1]
    new_alg_data = []
    new_human_data = []
    for alg_data_item in alg_data:
        new_alg_data.append([copy.deepcopy(alg_data_item[i])])
    for human_data_item in human_data:
        new_human_data.append([copy.deepcopy(human_data_item[i])])

    return [new_alg_data, new_human_data]

def split_diff_goal(alg_and_human, i):
    alg_data = alg_and_human[0]
    human_data = alg_and_human[1]
    new_alg_data = []
    new_human_data = []
    for alg_data_item in alg_data:
        new_alg_data_item = []
        for alg_data_item_item in alg_data_item:
            new_alg_data_item.append([copy.deepcopy(alg_data_item_item[i])])
        new_alg_data.append(new_alg_data_item)

    for human_data_item in human_data:
        new_human_data_item = []
        for human_data_item_item in human_data_item:
            new_human_data_item.append([copy.deepcopy(human_data_item_item[i])])
        new_human_data.append(new_human_data_item)


    return [new_alg_data, new_human_data]


if __name__ == '__main__':
    main()
