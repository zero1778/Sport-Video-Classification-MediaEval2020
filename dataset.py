from torch.utils.data import Dataset, DataLoader
from model import NetSimpleBranch
import os, cv2, json, random
import albumentations as A
import numpy as np

from utils import *
from cfgs import *


class My_dataset(Dataset):
    # def __init__(self, dataset_list, size_data, augmentation=0, norm_method = norm_method, flow_method = flow_method):
    def __init__(self, mode, __C):
        self.mode = mode
        self.__C = __C

        with open(__C.LABEL_DICT, "r") as f:
            self.label_dict = json.load(f)
        self._construct()

    def _construct(self):
        self.paths = []
        self.labels = []
        self.num_frames = []
        with open(self.__C.DATA_JSON, "r") as f:
            data_dict = json.load(f)
        data_paths = data_dict[self.mode]
        # print(data_paths)
        for data_path_frame in data_paths:
            data_frame = data_path_frame["number_of_frame"]
            self.num_frames.append(data_frame)

            data_path = data_path_frame["path"]
            self.paths.append(data_path)

            data_label = os.path.split(os.path.split(data_path)[0])[1]
            self.labels.append(self.label_dict[data_label])


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        rgb, flow, label = self.get_annotation_data(idx)
        # rgb, flolabel = get_annotation_data(self.dataset_list[idx], self.size_data, augmentation = self.augmentation, norm_method = self.norm_method)
        sample = {
            'rgb': torch.from_numpy(rgb), 
            'flow': torch.from_numpy(flow), 
            'label': label
        }
        return sample

    def get_annotation_data(self, idx):
        [T, H, W] = self.__C.SIZE_DATA
        # Get video info from idx
        video_path = self.paths[idx]
        video_frame = self.num_frames[idx]
        video_label = self.labels[idx]
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        processed_video_path = os.path.join(*(video_path.split(os.path.sep)[1:]))
        processed_video_path = os.path.join(self.__C.PATH_PROCESSED_DATA, processed_video_path)
        processed_video_path = os.path.splitext(processed_video_path)[0]

        # Get T frames from video
        # start_frame_idx = (video_frame - T) // 2
        frames_interval_idx = range(1, 101)
        # print("frame: ", video_frame)
        
        # Augmentation
        seed = random.randint(0, 999999999)
        if self.mode == 'train':
            if self.__C.AUGMENTATION:
                transform = A.Compose([
                    A.Flip(),
                    #A.RandomScale(),
                    A.Rotate(limit=10)
                ])
        
        # Get vdieo
        rgb_videos = []
        flow = np.load(os.path.join(processed_video_path, "values_flow_CVFlow.npy")).astype(np.float32)
        
        # normalize flow
        
        mean, stdDev = cv2.meanStdDev(flow)
        flow = (flow - mean) / stdDev
        flow = cv2.normalize(flow, None, -1, 1, cv2.NORM_MINMAX)
        flow_augment = []
        for frame_idx in frames_interval_idx:
            rgb_crop = cv2.imread(os.path.join(processed_video_path, "RGB_cropped/%08d.png" % (frame_idx))).astype(np.float32)
            rgb_crop = rgb_crop / 255.
            if self.mode == 'train':
                if self.__C.AUGMENTATION:
                    random.seed(seed)
                    transformed = transform(image=rgb_crop, mask=flow[frame_idx - 1])
                    rgb_crop = transformed['image']
                    frame_flow = transformed['mask']
            rgb_videos.append(rgb_crop)
            flow_augment.append(frame_flow)

        if self.mode == 'train':
            if self.__C.AUGMENTATION:
                flow = np.array(flow_augment)
        
        # print(flow.shape)
        rgb_video = np.transpose(np.array(rgb_videos), (3, 0, 1, 2))
        flow_video = np.transpose(np.array(flow), (3, 0, 1, 2))
        # print(rgb_video.shape)
        # print(flow_video.shape)
        # print(processed_video_path)
        return rgb_video, flow_video, video_label
        
        
        
if __name__ == "__main__":
    cfgs = Cfgs()
    train = My_dataset("train", cfgs)
    train[5]
    # print(np.maximum(train[199]['flow']))
