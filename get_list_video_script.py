from utils import make_path
import time, os, cv2, sys, json
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    data_dir = "data"
    save_path = "data_processed_1"
    f = open("data.json", 'w')
    for d in os.listdir(data_dir):
        cur_dir = os.path.join(data_dir, d)
        output_data = {}
        if d == "test":
            save_path = os.path.join(save_path, "test")
            video_list = []
            min_frame = 999999
            min_frame_video = ""
            for video in os.listdir(cur_dir):
                video_path = os.path.join(cur_dir, video)
                # video_list.append(video_path)
                video = cv2.VideoCapture(video_path)
                frame_number = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

                if (frame_number <= min_frame):
                    min_frame = frame_number
                    min_frame_video = video_path

                sample = {
                    "path": video_path,
                    "number_of_frame": frame_number 
                }
                video_list.append(sample)
            output_data[d] = video_list
            print("test:", min_frame_video + ", " + str(min_frame))
            # build_data(video_list, save_path, width_OF=320, log=None, workers=15, flow_method='DeepFlow')
        else:
            video_list = []
            min_frame = 999999
            min_frame_video = ""
            save_path = os.path.join(save_path, d)
            for label in sorted(os.listdir(cur_dir)):
                label_path = os.path.join(cur_dir, label)
                save_video_path = os.path.join(save_path, label)
                # video_list = []
                for video in os.listdir(label_path):
                    video_path = os.path.join(label_path, video)
                    # video_list.append(video_path)
                    video = cv2.VideoCapture(video_path)
                    frame_number = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

                    if (frame_number <= min_frame):
                        min_frame = frame_number
                        min_frame_video = video_path

                    sample = {
                        "path": video_path,
                        "number_of_frame": frame_number 
                    }
                    video_list.append(sample)
                # build_data(video_list, save_video_path, width_OF=320, log=None, workers=15, flow_method='DeepFlow')
            
            output_data[d] = video_list
            print(d + ": " + min_frame_video + ", " + str(min_frame))

        json.dump(output_data, f)



        
                
    