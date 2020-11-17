import time, os, cv2, sys
from tqdm import tqdm
import json
import sys
import numpy as np

if __name__ == "__main__":
    data_dir = "../data_processed"
    output_dir = "../data_processed_hybrid"
    for d in os.listdir(data_dir):
        if d == 'test':
            continue
        else:
            _d = os.path.join(data_dir, d)
            out_d = os.path.join(output_dir, d)
            with open("label_dict_hybrid.json", 'r') as f:
                label_dict = json.load(f)
            for label in sorted(os.listdir(_d)):
                out_label = label.split("_")[0] + "_" + label.split("_")[1]
                if out_label in label_dict.keys():
                    
            
    

