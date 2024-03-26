import os
import argparse

import numpy as np
import re

from IPython import embed

from utils import normlize_each_attn_map, corr_coeff, sim


def cal_sim_metric(goal_anomaly, goal_hazard):
    assert len(goal_anomaly[0]) == len(goal_hazard[0])
    rgb_image_count = len(goal_anomaly[0])
    single_image_feat_count = len(goal_anomaly)

    all_cc = []
    all_sim = []

    for image_idx in range(rgb_image_count):
        
        for feat_idx in range(single_image_feat_count):
            attn_image_feat = goal_anomaly[feat_idx][image_idx]
            attn_image_feat_1 = goal_hazard[feat_idx][image_idx]
    
            cc_data, p_data = corr_coeff(attn_image_feat, attn_image_feat_1)
            cc_data = abs(cc_data)
            sim_data = sim(attn_image_feat, attn_image_feat_1)

            all_cc.append(cc_data)
            all_sim.append(sim_data)
            
    final_cc = np.mean(all_cc)
    final_sim = np.mean(all_sim)

    return final_cc, final_sim            

def cal_sim(goal_anomaly_path, goal_hazard_path, goal_turn_path, per_batch_image_count):
    file_name_list = sorted([item for item in os.listdir(goal_anomaly_path) if item.endswith('.npy')])

    goal_anomaly = [np.load(os.path.join(goal_anomaly_path, item)) for item in file_name_list]
    goal_hazard = [np.load(os.path.join(goal_hazard_path, item)) for item in file_name_list]
    goal_turn = [np.load(os.path.join(goal_turn_path, item)) for item in file_name_list]
    
    '''normalize each attn map'''
    goal_anomaly = normlize_each_attn_map(goal_anomaly)
    goal_hazard = normlize_each_attn_map(goal_hazard)
    goal_turn = normlize_each_attn_map(goal_turn)

    '''split data'''
    data_anomaly_file_idx = [i for i in range(len(file_name_list)) if len(re.findall(r'(\d+_\d+_\d+)', file_name_list[i])) == 1]
    data_hazard_file_idx = [i for i in range(len(file_name_list)) \
        if len(re.findall(r'(\d+-\d+-\d+_\d+)', file_name_list[i])) == 1 or len(re.findall(r'(titan_clip_\d+_\d+)', file_name_list[i])) == 1]
    data_turn_file_idx = [i for i in range(len(file_name_list)) if len(re.findall(r'(\w{8}-\w+_\w+)', file_name_list[i])) == 1]
    
    attn_list_data_anomaly__goal_anomaly = [goal_anomaly[i] for i in data_anomaly_file_idx] # data anomaly
    attn_list_data_anomaly__goal_hazard = [goal_hazard[i] for i in data_anomaly_file_idx]
    attn_list_data_anomaly__goal_turn = [goal_turn[i] for i in data_anomaly_file_idx]
    
    attn_list_data_hazard__goal_anomaly = [goal_anomaly[i] for i in data_hazard_file_idx] # data drama
    attn_list_data_hazard__goal_hazard = [goal_hazard[i] for i in data_hazard_file_idx]
    attn_list_data_hazard__goal_turn = [goal_turn[i] for i in data_hazard_file_idx]
    
    attn_list_data_turn__goal_anomaly = [goal_anomaly[i] for i in data_turn_file_idx] # data oia
    attn_list_data_turn__goal_hazard = [goal_hazard[i] for i in data_turn_file_idx]
    attn_list_data_turn__goal_turn = [goal_turn[i] for i in data_turn_file_idx]
    
    '''get non zero data'''
    nonzero_idx_data_anomaly_ = []
    for i in range(len(attn_list_data_anomaly__goal_anomaly)):
        if attn_list_data_anomaly__goal_anomaly[i].sum() != 0 and attn_list_data_anomaly__goal_hazard[i].sum() != 0 and attn_list_data_anomaly__goal_turn[i].sum() != 0:
            nonzero_idx_data_anomaly_.append(i)
    nonzero_idx_data_hazard_ = []
    for i in range(len(attn_list_data_hazard__goal_anomaly)):
        if attn_list_data_hazard__goal_anomaly[i].sum() != 0 and attn_list_data_hazard__goal_hazard[i].sum() != 0 and attn_list_data_hazard__goal_turn[i].sum() != 0:
            nonzero_idx_data_hazard_.append(i)        
    nonzero_idx_data_turn_ = []
    for i in range(len(attn_list_data_turn__goal_anomaly)):
        if attn_list_data_turn__goal_anomaly[i].sum() != 0 and attn_list_data_turn__goal_hazard[i].sum() != 0 and attn_list_data_turn__goal_turn[i].sum() != 0:
            nonzero_idx_data_turn_.append(i)        
    max_count = min(len(nonzero_idx_data_anomaly_), len(nonzero_idx_data_hazard_), len(nonzero_idx_data_turn_), per_batch_image_count)
    nonzero_idx_data_anomaly_ = nonzero_idx_data_anomaly_[:max_count]
    nonzero_idx_data_hazard_ = nonzero_idx_data_hazard_[:max_count]
    nonzero_idx_data_turn_ = nonzero_idx_data_turn_[:max_count]

    attn_list_data_anomaly__goal_anomaly_0 = [attn_list_data_anomaly__goal_anomaly[i] for i in nonzero_idx_data_anomaly_]
    attn_list_data_anomaly__goal_hazard_0 = [attn_list_data_anomaly__goal_hazard[i] for i in nonzero_idx_data_anomaly_]
    attn_list_data_anomaly__goal_turn_0 = [attn_list_data_anomaly__goal_turn[i] for i in nonzero_idx_data_anomaly_]
    attn_list_data_hazard__goal_anomaly_0 = [attn_list_data_hazard__goal_anomaly[i] for i in nonzero_idx_data_hazard_]
    attn_list_data_hazard__goal_hazard_0 = [attn_list_data_hazard__goal_hazard[i] for i in nonzero_idx_data_hazard_]
    attn_list_data_hazard__goal_turn_0 = [attn_list_data_hazard__goal_turn[i] for i in nonzero_idx_data_hazard_]
    attn_list_data_turn__goal_anomaly_0 = [attn_list_data_turn__goal_anomaly[i] for i in nonzero_idx_data_turn_]
    attn_list_data_turn__goal_hazard_0 = [attn_list_data_turn__goal_hazard[i] for i in nonzero_idx_data_turn_]
    attn_list_data_turn__goal_turn_0 = [attn_list_data_turn__goal_turn[i] for i in nonzero_idx_data_turn_]
    '''all data'''
    all_list_goal = {
        'data_anomaly': [attn_list_data_anomaly__goal_anomaly_0, attn_list_data_anomaly__goal_hazard_0, attn_list_data_anomaly__goal_turn_0],
        'data_hazard': [attn_list_data_hazard__goal_anomaly_0, attn_list_data_hazard__goal_hazard_0, attn_list_data_hazard__goal_turn_0],
        'data_turn': [attn_list_data_turn__goal_anomaly_0, attn_list_data_turn__goal_hazard_0, attn_list_data_turn__goal_turn_0],
    }
    
    '''get non zero goal'''
    nonzero_idx_goal_anomaly_ = []
    for i in range(len(attn_list_data_anomaly__goal_anomaly)):
        if attn_list_data_anomaly__goal_anomaly[i].sum() != 0 and attn_list_data_hazard__goal_anomaly[i].sum() != 0 and attn_list_data_turn__goal_anomaly[i].sum() != 0:
            nonzero_idx_goal_anomaly_.append(i)
    nonzero_idx_goal_hazard_ = []
    for i in range(len(attn_list_data_anomaly__goal_hazard)):
        if attn_list_data_anomaly__goal_hazard[i].sum() != 0 and attn_list_data_hazard__goal_hazard[i].sum() != 0 and attn_list_data_turn__goal_hazard[i].sum() != 0:
            nonzero_idx_goal_hazard_.append(i)
    nonzero_idx_goal_turn_ = []
    for i in range(len(attn_list_data_anomaly__goal_turn)):
        if attn_list_data_anomaly__goal_turn[i].sum() != 0 and attn_list_data_hazard__goal_turn[i].sum() != 0 and attn_list_data_turn__goal_turn[i].sum() != 0:
            nonzero_idx_goal_turn_.append(i)
    max_count = min(len(nonzero_idx_goal_anomaly_), len(nonzero_idx_goal_hazard_), len(nonzero_idx_goal_turn_), per_batch_image_count)
    nonzero_idx_goal_anomaly_ = nonzero_idx_goal_anomaly_[:max_count]
    nonzero_idx_goal_hazard_ = nonzero_idx_goal_hazard_[:max_count]
    nonzero_idx_goal_turn_ = nonzero_idx_goal_turn_[:max_count]

    attn_list_data_anomaly__goal_anomaly_1 = [attn_list_data_anomaly__goal_anomaly[i] for i in nonzero_idx_goal_anomaly_]
    attn_list_data_anomaly__goal_hazard_1 = [attn_list_data_anomaly__goal_hazard[i] for i in nonzero_idx_goal_hazard_]
    attn_list_data_anomaly__goal_turn_1 = [attn_list_data_anomaly__goal_turn[i] for i in nonzero_idx_goal_turn_]
    attn_list_data_hazard__goal_anomaly_1 = [attn_list_data_hazard__goal_anomaly[i] for i in nonzero_idx_goal_anomaly_]
    attn_list_data_hazard__goal_hazard_1 = [attn_list_data_hazard__goal_hazard[i] for i in nonzero_idx_goal_hazard_]
    attn_list_data_hazard__goal_turn_1 = [attn_list_data_hazard__goal_turn[i] for i in nonzero_idx_goal_turn_]
    attn_list_data_turn__goal_anomaly_1 = [attn_list_data_turn__goal_anomaly[i] for i in nonzero_idx_goal_anomaly_]
    attn_list_data_turn__goal_hazard_1 = [attn_list_data_turn__goal_hazard[i] for i in nonzero_idx_goal_hazard_]
    attn_list_data_turn__goal_turn_1 = [attn_list_data_turn__goal_turn[i] for i in nonzero_idx_goal_turn_]
    '''all goal'''
    all_list_data = {
        'goal_anomaly': [attn_list_data_anomaly__goal_anomaly_1, attn_list_data_hazard__goal_anomaly_1, attn_list_data_turn__goal_anomaly_1],
        'goal_hazard': [attn_list_data_anomaly__goal_hazard_1, attn_list_data_hazard__goal_hazard_1, attn_list_data_turn__goal_hazard_1],
        'goal_turn': [attn_list_data_anomaly__goal_turn_1, attn_list_data_hazard__goal_turn_1, attn_list_data_turn__goal_turn_1],
    }

    print('\n******compare different goal')
    sim_matrix = np.zeros([3,3])
    cc_matrix = np.zeros([3,3])
    i = 0
    j = 0
    for k0,v0 in all_list_data.items():
        j = 0
        for k1,v1 in all_list_data.items():
            cc, sim = cal_sim_metric(v0, v1)
            sim_matrix[i,j] = sim
            cc_matrix[i,j] = cc
            j+=1
        i+=1
    print('cc_matrix: ')
    print(cc_matrix.reshape([3,3]))
    
    print('\n******compare different data')
    sim_matrix = np.zeros([3,3])
    cc_matrix = np.zeros([3,3])
    i = 0
    j = 0
    for k0,v0 in all_list_goal.items():
        j = 0
        for k1,v1 in all_list_goal.items():
            cc, sim = cal_sim_metric(v0, v1)
            sim_matrix[i,j] = sim
            cc_matrix[i,j] = cc
            j+=1
        i+=1
    print('cc_matrix: ')
    print(cc_matrix.reshape([3,3]))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--alg_pretrain', type=str, choices=['pretrained', 'no_pretrained'], default='pretrained')
    parser.add_argument('--alg_finetune', type=str, choices=['finetuned', 'no_finetuned'], default='finetuned')
    parser.add_argument('--per_batch_image_count', type=int, default=21)
    return parser.parse_args()

def main(args):
    if args.alg_finetune == 'finetuned':
        goal_anomaly_path = os.path.join(args.data_path, 'anomaly_detection', 'grad_cam_human_machine_final_version_olddriver_'+args.alg_pretrain)
        goal_hazard_path = os.path.join(args.data_path, 'drama', 'grad_cam_human_machine_final_version_olddriver_'+args.alg_pretrain)
        goal_turn_path = os.path.join(args.data_path, 'bddoia', 'code', 'grad_cam_human_machine_detr_final_version_olddriver_'+args.alg_pretrain)
    else:
        goal_anomaly_path = os.path.join(args.data_path, 'anomaly_detection', 'grad_cam_human_machine_final_version_driver_'+args.alg_pretrain+'_no_finetune')
        goal_hazard_path = os.path.join(args.data_path, 'drama', 'grad_cam_human_machine_final_version_driver_'+args.alg_pretrain+'_no_finetune')
        goal_turn_path = os.path.join(args.data_path, 'bddoia', 'code', 'grad_cam_human_machine_detr_final_version_driver_'+args.alg_pretrain+'_no_finetune')

    cal_sim(goal_anomaly_path, goal_hazard_path, goal_turn_path, args.per_batch_image_count)

if __name__ == '__main__':
    args = parse_args()
    main(args)
