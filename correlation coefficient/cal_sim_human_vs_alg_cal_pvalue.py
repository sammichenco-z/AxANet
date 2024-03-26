import os
import argparse

import numpy as np
import re
import copy
import scipy
import matplotlib.pyplot as plt
import pandas as pd

from IPython import embed

from utils import normlize_each_attn_map, corr_coeff, sim

COUNT=0
all_cc_list = []
all_name = []
def cal_sim_metric(goal_anomaly, goal_hazard, writer, name, resize_size):
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
            attn_mean_image_idx = attn_mean_image_idx / attn_mean_image_idx.sum()
            
            cc_image = []
            p_image = []
            sim_image = []
            for feat_idx_human in range(single_image_feat_count_human_0):
                for feat_idx_goal in range(single_image_feat_count_goal):
                    attn_image_feat = goal_anomaly[feat_idx_human][data_idx][feat_idx_goal][image_idx]
                    
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

    final_cc = np.mean(all_cc)
    final_p = np.mean(all_p)
    final_sim = np.mean(all_sim)

    global COUNT
    global all_cc_list
    global all_name
    all_cc_list.append(all_cc)
    
    mean = np.mean(all_cc)
    variance = np.var(all_cc)

    new_name = name + '_mean_' + str(mean)[:5] + '_var_' + str(variance)[:5]
    all_name.append(new_name)
    
    n, bins, patches = plt.hist(all_cc, bins=20, range=(-1.,1.), label=new_name)
    for i in range(len(n)):
        r_size = resize_size * resize_size
        dist = scipy.stats.beta(r_size/2 - 1, r_size/2 - 1, loc=-1, scale=2)
        r = bins[i]+(bins[1]-bins[0])/2
        p = 2*dist.cdf(-abs(r))
        if n[i] > 0:
            plt.text(r, n[i]*1.01, '%.4f' % p, ha='center', va= 'bottom', fontsize=7)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join('hist_fig', '%02d'% COUNT + '_%s.png' % new_name))
    plt.clf()
    
    header = ['cc', 'p']
    df = pd.DataFrame(np.stack([all_cc, all_p]).transpose(1,0),columns=header)
    name_split = name.split('_')
    if len(name_split) > 4:
        sheet_name = '_'.join([name_split[1], name_split[3], name_split[5], name_split[7]])
    else:
        sheet_name = name
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    COUNT+=1
    
    return final_cc, final_sim, final_p     

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

def cal_sim(data_path_alg, data_path_human_dir, human_data_stage, save_file_name, resize_size):
    writer = pd.ExcelWriter(save_file_name, mode='w', engine='openpyxl')

    for alg_idx in range(2):
        alg_old_new = ['pretrained', 'no_pretrained']
        alg_old_new_name = ['pretrained', 'nopretrained']

        ''' load data '''
        goal_anomaly_path = os.path.join(data_path_alg, 'anomaly_detection', 'grad_cam_human_machine_final_version_driver_'+alg_old_new[alg_idx]+'_no_finetune')
        goal_hazard_path = os.path.join(data_path_alg, 'drama', 'grad_cam_human_machine_final_version_driver_'+alg_old_new[alg_idx]+'_no_finetune')
        goal_turn_path = os.path.join(data_path_alg, 'bddoia', 'code', 'grad_cam_human_machine_detr_final_version_driver_'+alg_old_new[alg_idx]+'_no_finetune')

        file_name_list = sorted([item for item in os.listdir(goal_anomaly_path) if item.endswith('.npy')])

        goal_anomaly = [np.load(os.path.join(goal_anomaly_path, item)) for item in file_name_list]
        goal_hazard = [np.load(os.path.join(goal_hazard_path, item)) for item in file_name_list]
        goal_turn = [np.load(os.path.join(goal_turn_path, item)) for item in file_name_list]
        
        '''normalize each attn map'''
        goal_anomaly = normlize_each_attn_map(goal_anomaly, resize_size)
        goal_hazard = normlize_each_attn_map(goal_hazard, resize_size)
        goal_turn = normlize_each_attn_map(goal_turn, resize_size)

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

        '''all data'''
        all_list_goal = [
            [attn_list_data_anomaly__goal_anomaly, attn_list_data_anomaly__goal_hazard, attn_list_data_anomaly__goal_turn],
            [attn_list_data_hazard__goal_anomaly, attn_list_data_hazard__goal_hazard, attn_list_data_hazard__goal_turn],
            [attn_list_data_turn__goal_anomaly, attn_list_data_turn__goal_hazard, attn_list_data_turn__goal_turn],
        ]
        alg_data_list_goal = [copy.deepcopy(all_list_goal)]
        
        for human_idx in range(2):
            exp_list = ['old_mean', 'new_mean']
            data_path_human = os.path.join(data_path_human_dir, f'gaze_compare_final_version_{human_data_stage}_35_hazard_anomaly', exp_list[human_idx])
                
            data_dir = sorted([os.path.join(data_path_human,f) for f in os.listdir(data_path_human) if f.endswith('_fixation_only_200_35_0.0_1.0')])
            
            human_data_list_goal = []
            for data_dir_item in data_dir:
                ''' load data '''
                goal_anomaly_path = os.path.join(data_dir_item, 'Anomaly')
                goal_hazard_path = os.path.join(data_dir_item, 'Hazard')
                goal_turn_path = os.path.join(data_dir_item, 'Turn')
                goal_turn_path = goal_turn_path.replace(f'gaze_compare_final_version_{human_data_stage}_35_hazard_anomaly', f'gaze_compare_final_version_{human_data_stage}_65_oia').replace('200_35', '200_65')

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
    
                '''all data'''
                all_list_goal = [
                    [attn_list_data_anomaly__goal_anomaly, attn_list_data_anomaly__goal_hazard, attn_list_data_anomaly__goal_turn],
                    [attn_list_data_hazard__goal_anomaly, attn_list_data_hazard__goal_hazard, attn_list_data_hazard__goal_turn],
                    [attn_list_data_turn__goal_anomaly, attn_list_data_turn__goal_hazard, attn_list_data_turn__goal_turn],
                ]
                human_data_list_goal.append(all_list_goal)
                
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
                            else:
                                alg_data_list_humancount_data_goal.append(np.zeros_like(alg_data_list_goal[humandata_idx][data_idx][goal_idx][image_idx]))
                                human_data_list_humancount_data_goal.append(np.zeros_like(human_data_list_goal[humandata_idx][data_idx][goal_idx][image_idx]))
                        alg_data_list_humancount_data.append(alg_data_list_humancount_data_goal)
                        human_data_list_humancount_data.append(human_data_list_humancount_data_goal)
                    alg_data_list_humancount.append(alg_data_list_humancount_data)
                    human_data_list_humancount.append(human_data_list_humancount_data)
                nonzero_alg_data_list_goal.append(alg_data_list_humancount)
                nonzero_human_data_list_goal.append(human_data_list_humancount)
                
            alg_and_human = [nonzero_alg_data_list_goal, nonzero_human_data_list_goal] # 1,3,3,30 

            data_type = ['anomaly', 'hazard', 'turn']
            goal_type = ['anomaly', 'hazard', 'turn']

            print('*************************************************************')
            print('alg_idx: ', alg_old_new[alg_idx], 'human_idx: ', exp_list[human_idx])

            for data_idx in range(len(data_type)):
                print('====================')
                print('data type: ', data_type[data_idx])
                diff_data_alg_and_human = split_diff_data(alg_and_human, data_idx)

                for goal_idx in range(len(goal_type)):
                    print('----------------------')
                    print('goal type: ', goal_type[goal_idx])
                    
                    diff_goal_alg_and_human = split_diff_goal(diff_data_alg_and_human, goal_idx)

                    name = 'alg_' + alg_old_new_name[alg_idx] + '_human_' + exp_list[human_idx][:3] + '_data_' + data_type[data_idx] + '_goal_' + goal_type[goal_idx]
                    cc, sim, p = cal_sim_metric(diff_goal_alg_and_human[0], diff_goal_alg_and_human[1], writer, name, resize_size)
                    print('cc: ', cc, 'p: ', p)

    global all_cc_list
    global all_name
    plt.hist(all_cc_list, bins=20, range=(-1.,1.), label=all_name)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join('hist_fig', 'all.png'))
    plt.clf()

    writer.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path_alg', type=str)
    parser.add_argument('--data_path_human_dir', type=str)
    parser.add_argument('--human_data_stage', type=str, choices=['t0_t1', 't1_t2', 't2_end', 't0_end'], default='t1_t2')
    parser.add_argument('--save_file_name', type=str, default='r_p_file.xlsx')
    parser.add_argument('--resize_size', type=int, default=224)
    return parser.parse_args()

def main(args):
    cal_sim(args.data_path_alg, args.data_path_human_dir, args.human_data_stage, args.save_file_name, args.resize_size)

if __name__ == '__main__':
    args = parse_args()
    main(args)
