import cv2
import numpy as np 
import argparse 
import os
from os import listdir
from tqdm import tqdm
from os.path import isfile, join


def get_args():
    parser = argparse.ArgumentParser(description='Extracting Optical Flow',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', metavar='I', type=str,
                        help='Path of RGB images', dest='inp')
    parser.add_argument('-o', '--output', metavar='O', type=str,
                        help='Path to Optical Flow video', dest='out')

    return parser.parse_args()

if __name__ == "__main__":

    # The video feed is read in as 
    # a VideoCapture object 
    print("INFO: Computing DeepFlow...")
    args = get_args()
    # width_OF = 320
    # mypath = args.inp + '/'
    # mypath = 'data/train/Offensive_Backhand_Hit/7410672998_01112_01236.mp4'
    out_path = args.out
    # print(out_path)
    # exit(0)
    # cap = cv2.VideoCapture(mypath) 

    # ret = a boolean return value from 
    # getting the frame, first_frame = the 
    # first frame in the entire video sequence 
    
    ##TODO##
    rgb_images = [f for f in sorted(listdir(mypath)) if isfile(join(mypath, f))]

    # print(rgb_images)
    # exit(0)
    # width_OF=320
    # ret, first_frame = cap.read() 
    # first_frame = cv2.resize(first_frame, (width_OF, first_frame.shape[0] * width_OF // first_frame.shape[1]))
    # Converts frame to grayscale because we 
    # only need the luminance channel for 
    # detecting edges - less computationally 
    # expensive 
    # print(mypath)
    ##TODO##
    first_frame = cv2.imread(mypath + rgb_images[0])
    
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY) 

    # Creates an image filled with zero 
    # intensities with the same dimensions 
    # as the frame 
    mask = np.zeros_like(first_frame) 

    # Sets image saturation to maximum 
    mask[..., 1] = 255

    mog = cv2.createBackgroundSubtractorMOG2()
    

    for frame_number, list_frame in tqdm(enumerate(rgb_images[1:])):
        # ret = a boolean return value from getting 
        # the frame, frame = the current frame being 
        # projected in the video 
        # ret, frame = cap.read() 
        # frame = cv2.resize(frame, (width_OF, frame.shape[0] * width_OF // frame.shape[1]))
        # Opens a new window and displays the input 
        # frame 
        frame = cv2.imread(path_rgb + list_frame)
        
        def pp1(fmask):    
            info = np.iinfo(fmask.dtype)
            fmask = fmask.astype(np.float64) / info.max
            fmask = fmask * 255
            fmask = fmask.astype(np.uint8) 
            return fmask
        def pp2(fmask):
            return fmask / 255
        def pp3(fmask):
            return fmask % 255
        fgmask = mog.apply(frame)
        fgmaskT = (fgmask != 0).astype(int)
        fgmaskT = np.expand_dims(fgmaskT, axis=-1)
        # Converts each frame to grayscale - we previously 
        # only converted the first frame to grayscale 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        
        # Calculates dense optical flow by Farneback method 
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, 
                                        None, 
                                        0.5, 3, 15, 3, 5, 1.2, 0) 
        flow_filter = flow * fgmaskT
        magnitude, angle = cv2.cartToPolar(flow_filter[..., 0], flow_filter[..., 1]) 
        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX) 
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2RGB) 
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) 

        # cv2.imshow("Flow_filter", gray)
              
        
        # Computes the magnitude and angle of the 2D vectors 
        # magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1]) 
        # mask[..., 0] = angle * 180 / np.pi / 2
        # mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX) 
        # rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2RGB) 
        # gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) 
        # cv2.imshow("Flow", gray)
        # # cv2.imshow("Optical Flow", gray)
        # cv2.imshow("Foreground Extracted", fgmask)
        # fmask = (gray * fgmask)
        # fmask = pp1(fmask)
        fmaskT = (gray != 0).astype(int)  

        # Updates previous frame 
        prev_gray = gray 
        
        ##############################333
        # flow1 = np.sum(np.abs(flow), axis = -1) # 180 x 320 
        # flow2 = flow1 * fmaskT

        # idx_max = np.argmax(flow2)
        # X_max, Y_max = idx_max // w_y, idx_max % w_y
        ###################################
    
        X_sum, Y_sum = 0, 0
        flow_num = (fmaskT != 0).astype(int)
        for x in range(180):
            for y in range(320):
                X_sum += x * flow_num[x][y]
                Y_sum += y * flow_num[x][y]
        flow_num = flow_num.sum()
    
        #Calculate Gravity
        X_g, Y_g = int(X_sum/flow_num), int(Y_sum/flow_num)
        print("Gravity = ", X_g, Y_g)

        X_tl, Y_tl =  X_g - 60, Y_g - 75

        if (X_tl + W > w_x): X_tl = w_x - W
        if (X_tl < 0): X_tl = 0
        if (Y_tl + H > w_y): Y_tl = w_y - H
        if (Y_tl < 0): Y_tl = 0
        crop_image = frame[X_tl: X_tl + W, Y_tl : Y_tl + H]
        
        # print(crop_image.shape)
        cv2.imshow("Cropped",crop_image)
        cv2.waitKey(0)
        cv2.imwrite(os.path.join(out_path, '%08d.png' % (frame_number + 1)), crop_image)
        # image = cv2.rectangle(fmask, (Y_g - 75,X_g - 60), (Y_g + 45,X_g + 60), color=(255), thickness=3)

        
        # cv2.imshow('ROI Flow', image)
        # cv2.waitKey(0)

        cv2.imwrite(os.path.join(out_path, '%08d.png' % (frame_number + 1)), fmask)


    # cap.release() 
    cv2.destroyAllWindows() 
