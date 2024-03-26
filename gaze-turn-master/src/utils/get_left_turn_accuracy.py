import os
import pandas as pd
import json
import numpy as np
from tqdm import tqdm


human_machine_choose = os.listdir("human_machine_raw_v3")







def calculate_accuracy_recall(predictions, labels, positive_class):
    TP = sum(1 for pred, label in zip(predictions, labels) if pred == label == positive_class)
    FN = sum(1 for pred, label in zip(predictions, labels) if pred != positive_class and label == positive_class)
    accuracy = sum(1 for pred, label in zip(predictions, labels) if pred == label) / len(labels)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return accuracy, recall


# to_eval_file = "oia_output/base_pretrained/checkpoint-100-3125/pred.test.beam1.max30.tsv"
to_eval_file = "oia_output/base_no_pretrained/checkpoint-100-12500/pred.test.beam1.max30.tsv"

df = pd.read_csv(to_eval_file, sep='\t', header=None)
result_dict = {}

acc = 0.
count = 0.

predictions = []
labels = []

choose_predictions = []
choose_labels = []

for index, row in tqdm(df.iterrows()):
    dict_list = json.loads(row[1])
    pred_action = dict_list[0]['pred_action'].split("([")[-1].split("])")[0][6]
    label_action = dict_list[0]['label_action'].split("([")[-1].split("])")[0][6]
    if pred_action == label_action:
        acc += 1
    count += 1
    predictions.append(pred_action)
    labels.append(label_action)

    if row[0]+"_raw.png" in human_machine_choose:
        choose_predictions.append(pred_action)
        choose_labels.append(label_action)

print(f"Acc: {acc/count}")

print(predictions)
print("all:")
print(calculate_accuracy_recall(predictions, labels, '1'))
print("30:")
print(calculate_accuracy_recall(choose_predictions, choose_labels, '1'))
print(to_eval_file)