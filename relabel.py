import re 
label = {}
with open('label_test.csv', 'r') as fin:
    for line in fin:
        path, result=line.strip().split(',')
        label[path] = result.split('_')[-1]
with open('hybrid_label_test_2.csv', 'w') as fout:
    for key, val in label.items():
        fout.writelines(key + "," + val + "\n")

    
