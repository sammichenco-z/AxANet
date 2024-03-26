import os
import json
root = "output"

keyword = ""
exceptword = ""

depth = 1

name = "pred.test.beam1.max30.eval.json"
if depth == 1:
    max_iou = 0
    max_iou_path = ""
    for k in os.listdir(root):

        if len(keyword)!=0 and k.find(keyword)==-1:
            continue
        if len(exceptword)!=0 and k.find(exceptword)!=-1:
            continue

        new_root = os.path.join(root, k)
        for i in os.listdir(new_root):
            path = os.path.join(new_root, i, name)
            if not os.path.exists(path):
                continue
            js_file = json.load(open(path, "r"))
            if js_file['CIDEr'] > max_iou:
                max_iou = js_file['CIDEr']
                max_iou_path = path
else:
    max_iou = 0
    max_iou_path = ""
    for i in os.listdir(root):
        path = os.path.join(root, i, name)
        if not os.path.exists(path):
            continue
        js_file = json.load(open(path, "r"))
        if js_file['CIDEr'] > max_iou:
            max_iou = js_file['CIDEr']
            max_iou_path = path

print(root)
print(max_iou)
print(max_iou_path)
