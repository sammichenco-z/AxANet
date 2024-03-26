import os
import argparse

import numpy as np
import re

from IPython import embed

from utils import normlize_each_attn_map, corr_coeff, sim


def cal_sim_metric(goal_anomaly, goal_hazard):        
    assert len(goal_anomaly[0]) == len(goal_hazard[0])
    assert len(goal_anomaly[0][0]) == len(goal_hazard[0][0])
    single_image_feat_count_human = len(goal_anomaly)
    single_image_feat_count_data = len(goal_anomaly[0])
    rgb_image_count = len(goal_anomaly[0][0])

    all_cc = []
    all_p = []
    all_sim = []

    for data_idx in range(single_image_feat_count_data):
        for image_idx in range(rgb_image_count):
            attn_image_idx_list = [goal_hazard[i][data_idx][image_idx] for i in range(single_image_feat_count_human)]
            assert len(attn_image_idx_list) == single_image_feat_count_human
            
            attn_mean_image_idx = np.sum(np.stack(attn_image_idx_list), axis=0)
            attn_mean_image_idx = attn_mean_image_idx / (attn_mean_image_idx.sum()+1e-20)
            
            cc_image = []
            p_image = []
            sim_image = []
            for feat_idx_human in range(single_image_feat_count_human):
                attn_image_feat = goal_anomaly[feat_idx_human][data_idx][image_idx]
    
                cc_data = abs(corr_coeff(attn_image_feat, attn_mean_image_idx)[0])
                if np.isnan(cc_data):
                    embed()
                p_data = corr_coeff(attn_image_feat, attn_mean_image_idx)[1]
                sim_data = sim(attn_image_feat, attn_mean_image_idx)

                if cc_data > 0.999:
                    cc_data = 1.0

                cc_image.append(cc_data)
                p_image.append(p_data)
                sim_image.append(sim_data)
            all_cc.append(np.mean(cc_image))
            all_p.append(np.mean(p_image))
            all_sim.append(np.mean(sim_image))

    final_cc = np.mean(all_cc)
    final_p = np.mean(all_p)
    final_sim = np.mean(all_sim)

    return final_cc, final_sim, final_p       

def cal_sim(goal_anomaly_path_list, goal_hazard_path_list, goal_turn_path_list, per_batch_image_count):
    human_data_list_goal = []
    human_data_list_data = []
    
    for i in range(len(goal_anomaly_path_list)):
        goal_anomaly_path = goal_anomaly_path_list[i]
        goal_hazard_path = goal_hazard_path_list[i]
        goal_turn_path = goal_turn_path_list[i]

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
    
        human_data_list_goal.append(all_list_goal)
        human_data_list_data.append(all_list_data)


    print('\n******compare different data')
    sim_matrix = np.zeros([3,3])
    cc_matrix = np.zeros([3,3])
    p_matrix = np.zeros([3,3])
    i = 0
    j = 0
    for k0,v0 in human_data_list_goal[0].items():
        j = 0
        temp_data_list_v0 = [temp_data[k0] for temp_data in human_data_list_goal]
        for k1,v1 in human_data_list_goal[0].items():
            temp_data_list_v1 = [temp_data[k1] for temp_data in human_data_list_goal]
            cc, sim, p = cal_sim_metric(temp_data_list_v0, temp_data_list_v1)
            sim_matrix[i,j] = sim
            cc_matrix[i,j] = cc
            p_matrix[i,j] = p
            j+=1
        i+=1
    print('cc_matrix: ')
    print(cc_matrix.reshape([3,3]))

    print('\n******compare different goal')
    sim_matrix = np.zeros([3,3])
    cc_matrix = np.zeros([3,3])
    p_matrix = np.zeros([3,3])
    i = 0
    j = 0
    for k0,v0 in human_data_list_data[0].items():
        j = 0
        temp_data_list_v0 = [temp_data[k0] for temp_data in human_data_list_data]
        for k1,v1 in human_data_list_data[0].items():
            temp_data_list_v1 = [temp_data[k1] for temp_data in human_data_list_data]
            cc, sim, p = cal_sim_metric(temp_data_list_v0, temp_data_list_v1)
            sim_matrix[i,j] = sim
            cc_matrix[i,j] = cc
            p_matrix[i,j] = p
            j+=1
        i+=1
    print('cc_matrix: ')
    print(cc_matrix.reshape([3,3]))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--data_stage', type=str, choices=['t0_t1', 't1_t2', 't2_end', 't0_end'], default='t0_end')
    parser.add_argument('--driver_level', type=str, choices=['old_mean', 'new_mean'], default='new_mean')
    parser.add_argument('--per_batch_image_count', type=int, default=30)
    return parser.parse_args()

def main(args):
    DATA_PATH = os.path.join(args.data_path, f'gaze_compare_final_version_{args.data_stage}_35_hazard_anomaly', args.driver_level)

    data_dir = sorted([os.path.join(DATA_PATH,f) for f in os.listdir(DATA_PATH) if f.endswith('_fixation_only_200_35_0.0_1.0')])
    
    goal_anomaly_path_list = []
    goal_hazard_path_list = []
    goal_turn_path_list = []
    for data_dir_item in data_dir:
        ''' load data '''
        goal_anomaly_path = os.path.join(data_dir_item, 'Anomaly')
        goal_hazard_path = os.path.join(data_dir_item, 'Hazard')
        goal_turn_path = os.path.join(data_dir_item, 'Turn')
        goal_turn_path = goal_turn_path.replace(f'gaze_compare_final_version_{args.data_stage}_35_hazard_anomaly', f'gaze_compare_final_version_{args.data_stage}_65_oia').replace('200_35', '200_65')

        goal_anomaly_path_list.append(goal_anomaly_path)
        goal_hazard_path_list.append(goal_hazard_path)
        goal_turn_path_list.append(goal_turn_path)

    cal_sim(goal_anomaly_path_list, goal_hazard_path_list, goal_turn_path_list, args.per_batch_image_count)

if __name__ == '__main__':
    args = parse_args()
    main(args)
