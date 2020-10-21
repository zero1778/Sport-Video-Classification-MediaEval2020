from utils import make_path
import time, os, cv2, sys
from tqdm import tqdm
import json
import sys
import numpy as np

############################################################
##################### Build the data #######################
############################################################
def build_data(video_list, save_path, width_OF=320, log=None, workers=15, flow_method='DeepFlow'):
    make_path(save_path)
    # Extract Frames
    extract_frames(video_list, save_path, width_OF, log)
    

    # Compute DeepFlow
    # compute_DeepFlow(video_list, save_path, log, workers)

    # Compute ROI
    compute_ROI(video_list, save_path, log, workers, flow_method=flow_method)


##################### RGB #######################
def extract_frames(video_list, save_path, width_OF, log):
    # Chrono
    start_time = time.time()
    print("INFO: Extracting RGB Frame...")
    for video_path in tqdm(video_list):
        
        video_name = os.path.basename(video_path)
        # progress_bar(idx, len(video_list), 'Frame extraction - %s' % (video_name))

        path_data_video = os.path.join(save_path, video_name.split('.')[0])
        make_path(path_data_video)
        path_RGB = os.path.join(path_data_video, 'RGB')
        make_path(path_RGB)
        path_DeepFlow= os.path.join(path_data_video, 'DeepFlow')
        make_path(path_DeepFlow)

        # Load Video
        cap = cv2.VideoCapture(video_path)
        length_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_number = 0
        
        # Check if video uploaded
        if not cap.isOpened():
            print("Error at ", video_path)
            sys.exit("Unable to open the video, check the path. \n")
            

        while frame_number < length_video:
            # Load video
            _, rgb = cap.read()

            # Check if load Properly
            if _ == 1:
                # Resizing and Save
                rgb = cv2.resize(rgb, (width_OF, rgb.shape[0] * width_OF // rgb.shape[1]))
                cv2.imwrite(os.path.join(path_RGB, '%08d.png' % frame_number), rgb)
                frame_number += 1
        cap.release()

    # progress_bar(idx+1, len(video_list), 'Frame extraction completed in %d s' % (time.time() - start_time), 1, log=log)

##################### Deep Flow #######################
def compute_DeepFlow(video_list, save_path, log, workers):
    start_time = time.time()
    # DeepFlow_pool = ActivePool()
  
    for idx, video_path in enumerate(video_list):

        video_name = os.path.basename(video_path).split('.')[0]
        path_data_video = os.path.join(save_path, video_name)
        # make_path()
        compute_DeepFlow_video(os.path.join(path_data_video, 'RGB'), os.path.join(path_data_video, 'DeepFlow'))
        # Split the calculation in severals process
    #     while threading.activeCount() > workers:
    #         # progress_bar(idx + 1 - threading.activeCount(), len(video_list), 'DeepFlow computation')
    #         time.sleep(0.1)

    #     if threading.activeCount() <= workers:
    #         job = threading.Thread(target = compute_DeepFlow_video, name = idx, args = (DeepFlow_pool,
    #                                                                                     os.path.join(path_data_video, 'RGB'),
    #                                                                                     os.path.join(path_data_video, 'DeepFlow')))
    #         job.daemon=True
    #         job.start()

    # while threading.activeCount()>1:
        # progress_bar(idx + 1 - threading.activeCount(), len(video_list), 'DeepFlow computation')
        time.sleep(0.1)

    # progress_bar(idx + 1, len(video_list), 'DeepFlow computation done in %d s' % (time.time() - start_time), 1, log=log)


# def compute_DeepFlow_video(pool, path_RGB, path_Flow):
def compute_DeepFlow_video(path_RGB, path_Flow):
    # name = threading.current_thread().name
    # pool.makeActive(name)
    os.system('python cv_flow.py -i %s -o %s' % (path_RGB, path_Flow))
    # pool.makeInactive(name)


##################### ROI #######################
def compute_ROI(video_list, save_path, log, workers, flow_method='CVFlow'):
    start_time = time.time()
    
    print("INFO: Computing ROI Flow...")

    # ROI_pool = ActivePool()

    for video_path in tqdm(video_list):

        video_name = os.path.basename(video_path).split('.')[0]
        # label = os.path.basename(video_path).split('.')[1]
        # print(label)
        path_data_video = os.path.join(save_path, video_name)
        make_path(path_data_video + "/RGB_cropped")
        output_rgb_cropped = os.path.join(path_data_video, "RGB_cropped")
        compute_roi_video(path_data_video, flow_method, output_rgb_cropped)
        # Split the calculation in severals process
    #     while threading.activeCount() > workers:
    #         # progress_bar(idx + 1 - threading.activeCount(), len(video_list), 'ROI computation for %s' % (flow_method))
    #         time.sleep(0.1)

    #     if threading.activeCount() <= workers:
    #         job = threading.Thread(target = compute_roi_video, name = idx, args = (ROI_pool,
    #                                                                                path_data_video,
    #                                                                                flow_method))
    #         job.daemon=True
    #         job.start()

    # while threading.activeCount()>1:
    #     # progress_bar(idx + 1 - threading.activeCount(), len(video_list), 'ROI computation for %s' % (flow_method))
    #     time.sleep(0.1)

    # join_values_flow(video_list, 'values_flow_%s' % flow_method, save_path)
    # progress_bar(len(video_list), len(video_list), 'ROI computation for %s completed in %d s' % (flow_method, int(time.time() - start_time)), 1, log=log)


def compute_roi_video(path_data_video, flow_method, output_rgb_cropped):
    # name = threading.current_thread().name
    # pool.makeActive(name)
    os.system('python roi_flow.py -i %s -m %s -o %s' % (path_data_video, flow_method, output_rgb_cropped))
    # pool.makeInactive(name)


def join_values_flow(video_list, name_values, save_path):
    values_flow = []
    for video in video_list:
        video_name = os.path.basename(video).split('.')[0]
        path_data_video = os.path.join(save_path, video_name)
        values_flow_video = np.load(os.path.join(path_data_video, '%s.npy' % name_values))
        values_flow.extend(values_flow_video)
    np.save(os.path.join(save_path, name_values), values_flow)

if __name__ == "__main__":
    # /Users/bangdang2000/Documents/AI/Contest/MediaEval2020/data/train/Offensive_Backhand_Hit/7410672998_01112_01236.mp4
    # data_dir = "data"
    # save_path = "data_processed_1"
    # for d in os.listdir(data_dir):
    #     cur_dir = os.path.join(data_dir, d)
    #     if d == "test":
    #         continue
    #         save_path = os.path.join(save_path, "test")
    #         video_list = []
    #         for video in os.listdir(cur_dir):
    #             video_path = os.path.join(cur_dir, video)
    #             video_list.append(video_path)
    #         build_data(video_list, save_path, width_OF=320, log=None, workers=15, flow_method='DeepFlow')
    #     else:
    #         if d == "val":
    #             continue
    #         save_path = os.path.join(save_path, d)
    #         for label in sorted(os.listdir(cur_dir)):
    #             label_path = os.path.join(cur_dir, label)
    #             save_path = os.path.join(save_path, label)
    #             video_list = []
    #             for video in os.listdir(label_path):
    #                 video_path = os.path.join(label_path, video)
    #                 video_list.append(video_path)
    #             build_data(video_list, save_path, width_OF=320, log=None, workers=15, flow_method='DeepFlow')
                            

            

    with open('data.json') as json_file:
        data = json.load(json_file)
       
        # for p in data['people']:
        #     print('Name: ' + p['name'])
        #     print('Website: ' + p['website'])
        #     print('From: ' + p['from'])
        #     print('')
    train_list = []
    for i in data['train']:
        train_list.append(i['path'])
    # train_list = train_list[:3]
    
    # video_list = ['data/train/Defensive_Backhand_Backspin/3197874210_00768_00952.mp4']
    # save_path = 'data_preprocessing_1/train/Defensive_Backhand_Backspin/'
    # build_data(video_list, save_path, width_OF=320, log=None, workers=15, flow_method='DeepFlow')
    # train_list = ['data/train/Defensive_Backhand_Block/786246856_03988_04040.mp4']
    # video_list = ['data/train/Offensive_Backhand_Hit/7410672998_01112_01236.mp4']
    # video_list = ['data/train/Offensive_Forehand_Loop/7410672998_07924_08136.mp4'] #good
    # video_list = ['data/train/Offensive_Forehand_Loop/268101021042_01016_01200.mp4']
    # video_list = ['data/val/Serve_Forehand_Backspin/2710727544_01352_01508.mp4']
    # video_list = ['data/train/Serve_Backhand_Topspin/715368773_00876_01044.mp4'] # 2 people
    # video_list = ['data/train/Serve_Backhand_Topspin/9841059524_02848_03036.mp4'] # 
    # video_list = ['data/train/Offensive_Forehand_Hit/7410672998_03308_03472.mp4']
    save_path = 'data/train_list/'
    make_path(save_path)
    build_data(train_list, save_path, width_OF=320, log=None, workers=15, flow_method='DeepFlow')
