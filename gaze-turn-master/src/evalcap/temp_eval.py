from collections import OrderedDict, defaultdict
import json
import numpy as np
import os.path as op
import os
from pprint import pprint
from pprint import pformat
#import torch
import re
import subprocess
import tempfile
import time
from typing import Dict, Optional
#from src.utils.tsv_file import TSVFile, CompositeTSVFile
#from src.utils.tsv_file_ops import tsv_reader
from sklearn.metrics import f1_score, accuracy_score
from IPython import embed

def get_user_name():
    import getpass
    return getpass.getuser()
def acquireLock(lock_f='/tmp/lockfile.LOCK'):
    ''' acquire exclusive lock file access '''
    import fcntl
    locked_file_descriptor = open(lock_f, 'w+')
    fcntl.lockf(locked_file_descriptor, fcntl.LOCK_EX)
    return locked_file_descriptor
def hash_sha1(s):
    import hashlib
    if type(s) is not str:
        s = pformat(s)
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

def exclusive_open_to_read(fname, mode='r'):
    disable_lock = os.environ.get('QD_DISABLE_EXCLUSIVE_READ_BY_LOCK')
    if disable_lock is not None:
        disable_lock = int(disable_lock)
    if not disable_lock:
        user_name = get_user_name()
        lock_fd = acquireLock(op.join('/tmp',
            '{}_lock_{}'.format(user_name, hash_sha1(fname))))
def delete_tensor(str=''):
    return str[7:-1]
def tsv_reader(tsv_file, sep='\t'):
    with open(tsv_file, 'r') as fp:
        for i, line in enumerate(fp):
            yield [x.strip() for x in line.split(sep)]

# TODO: ADD EVALUATION METRICS
def class_eval(res_file, outfile=None):
    """
    res_tsv: TSV file, each row is [image_key, json format list of captions].
             Each caption is a dict, with fields "caption", "conf".
             or JSON file of coco style
    label_file: .pt file, contains dict of image key to ground truth labels.
             or JSON file of coco style
    """
    if not outfile:
        outfile = op.splitext(res_file)[0] + '.class_eval.json'

    acc = 0.
    pred_action_list=[]
    pred_reason_list=[]
    label_action_list=[]
    label_reason_list=[]
    pred_list=[]
    label_list=[]
    f_label_list=[]
    f_pred_list=[]
    s_label_list=[]
    s_pred_list=[]
    l_label_list=[]
    l_pred_list=[]
    r_label_list=[]
    r_pred_list=[]
    for i, row in enumerate(tsv_reader(res_file)):

        pred_action = json.loads(row[1])[0]['pred_action']
        pred_reason = json.loads(row[1])[0]['pred_reason']
        label_action = json.loads(row[1])[0]['label_action']
        label_reason = json.loads(row[1])[0]['label_reason']

        pred_action=eval(delete_tensor(pred_action))
        pred_reason=eval(delete_tensor(pred_reason))
        label_action=eval(delete_tensor(label_action))
        label_reason=eval(delete_tensor(label_reason))

        f_label_list.append(label_action[0])
        f_pred_list.append(pred_action[0])
        s_label_list.append(label_action[1])
        s_pred_list.append(pred_action[1])
        l_label_list.append(label_action[2])
        l_pred_list.append(pred_action[2])
        r_label_list.append(label_action[3])
        r_pred_list.append(pred_action[3])

        pred=pred_action+pred_reason
        label=label_action+label_reason

        pred_list.append(pred)
        label_list.append(label)

        pred_action_list.append(pred_action)
        pred_reason_list.append(pred_reason)

        label_action_list.append(label_action)
        label_reason_list.append(label_reason)

    embed()
    f1_reason_all=f1_score(label_reason_list,pred_reason_list,average='samples')
    f1_action_all=f1_score(label_action_list,pred_action_list,average='samples')
    f1_total_all=f1_score(label_list,pred_list,average='samples')
    
    f1_f=f1_score(f_label_list,f_pred_list,average='binary')
    f1_s=f1_score(s_label_list,s_pred_list,average='binary')
    f1_l=f1_score(l_label_list,l_pred_list,average='binary')
    f1_r=f1_score(r_label_list,r_pred_list,average='binary')
    mf1_action = (f1_f+f1_s+f1_l+f1_r)/4
    mf1_reason = np.mean(f1_score(label_reason_list,pred_reason_list,average=None))
    
    #mean_accuracy=accuracy_score(label_list,pred_list)
    print('The f1_reason_all is {}, f1_action_all is {}, f1_total_all is {}'.format(f1_reason_all,f1_action_all,f1_total_all))
    print('The f1_f is {}, f1_s is {}, f1_l is {},f1_r is {}'.format(f1_f,f1_s,f1_l,f1_r))




    f_pred_list_rand = list(np.random.randint(0,2,len(f_pred_list)))
    s_pred_list_rand = list(np.random.randint(0,2,len(s_pred_list)))
    l_pred_list_rand = list(np.random.randint(0,2,len(l_pred_list)))
    r_pred_list_rand = list(np.random.randint(0,2,len(r_pred_list)))
    f1_f_rand=f1_score(f_label_list,f_pred_list_rand,average='binary')
    f1_s_rand=f1_score(s_label_list,s_pred_list_rand,average='binary')
    f1_l_rand=f1_score(l_label_list,l_pred_list_rand,average='binary')
    f1_r_rand=f1_score(r_label_list,r_pred_list_rand,average='binary')
    print('--------------')
    print('The random f1_f is {}, f1_s is {}, f1_l is {},f1_r is {}'.format(f1_f_rand,f1_s_rand,f1_l_rand,f1_r_rand))
    mf1_action_rand = (f1_f_rand+f1_s_rand+f1_l_rand+f1_r_rand)/4
    print('mean: {}'.format(mf1_action_rand))


    f_pred_list_ones = list(np.ones(len(f_pred_list)).astype(int))
    s_pred_list_ones = list(np.ones(len(s_pred_list)).astype(int))
    l_pred_list_ones = list(np.ones(len(l_pred_list)).astype(int))
    r_pred_list_ones = list(np.ones(len(r_pred_list)).astype(int))
    f1_f_ones=f1_score(f_label_list,f_pred_list_ones,average='binary')
    f1_s_ones=f1_score(s_label_list,s_pred_list_ones,average='binary')
    f1_l_ones=f1_score(l_label_list,l_pred_list_ones,average='binary')
    f1_r_ones=f1_score(r_label_list,r_pred_list_ones,average='binary')
    print('--------------')
    print('The ones f1_f is {}, f1_s is {}, f1_l is {},f1_r is {}'.format(f1_f_ones,f1_s_ones,f1_l_ones,f1_r_ones))
    mf1_action_ones = (f1_f_ones+f1_s_ones+f1_l_ones+f1_r_ones)/4
    print('mean: {}'.format(mf1_action_ones))


    #macc = acc / (i+1)

    # if not outfile:
    #     #print('The f1_action_all is {}'.format(f1_action_all))
    #     print('The f1_f is {}, f1_s is {}, f1_l is {},f1_r is {}'.format(f1_f,f1_s,f1_l,f1_r))
    # else:
    #     with open(outfile, 'w') as fp:
    #         json.dump({'f1_reason_all': f1_reason_all,'f1_action_all':f1_action_all,'f1_total_all':f1_total_all,
    #                    'f1_f':f1_f,'f1_s':f1_s,'f1_l':f1_l,'f1_r':f1_r,
    #                    'mf1_action': mf1_action, 'mf1_reason': mf1_reason}, fp, indent=4)
    #         #json.dump({'f1_action_all':f1_action_all,'f1_f':f1_f,'f1_s':f1_s,'f1_r':f1_r,'f1_l':f1_l}, fp, indent=4)
    return f1_reason_all,f1_action_all,f1_total_all,f1_f,f1_s,f1_l,f1_r, mf1_action, mf1_reason

if __name__=='__main__':
    file='/videocap/output/test0316_lr001_bklr_10_bs32_gaze_dotinput_gazeonly/checkpoint-99-49500/pred.test.beam1.max30.tsv'
    class_eval(file,outfile=False)