import csv 
import json
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

f = open('label_dict.json', "r")
label_dict = json.load(f)


label = {}
with open('label_test.csv', 'r') as fin:
    for line in fin:
        path, result=line.strip().split(',')
        label[path] = result
label_key = sorted(label.keys())
label_val = [label_dict[label[t]] for t in label_key]
# print(label_val)


pred = {}
with open('/home/dhieu/Contest/MediaEval2020/sport/Sport-Video-Classification-MediaEval2020/submission.csv', 'r') as fin:
    for line in fin:
        path, result=line.strip().split(',')
        if path in label.keys():
            pred[path] = result
pred_key = sorted(pred.keys())
pred_val = [label_dict[pred[t]] for t in pred_key]
# print(pred_val)
# print(label_dict)


conf_mat = confusion_matrix(label_val, pred_val)
plt.figure(figsize = (10,7))
df_cm = pd.DataFrame(conf_mat, range(20), range(20))
sn.set(font_scale=0.7)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
plt.savefig("model_.png")

res = 0
for key,val in pred.items():
    if key in label.keys():
        res += (pred[key] == label[key])


print(res/len(label))