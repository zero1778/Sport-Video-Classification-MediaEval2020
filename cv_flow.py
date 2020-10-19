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
    mypath = args.inp + '/'
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
    # while (1):
        # ret = a boolean return value from getting 
        # the frame, frame = the current frame being 
        # projected in the video 
        # ret, frame = cap.read() 
        # frame = cv2.resize(frame, (width_OF, frame.shape[0] * width_OF // frame.shape[1]))
        # Opens a new window and displays the input 
        # frame 
        frame = cv2.imread(mypath + list_frame)
        
        # fgmask = 
        # cv2.imshow('frame',fgmask)
        # cv2.waitKey(0)
        # print(fgmask)
        # cv2.imshow("input", frame) 
        # cv2.waitKey(0)
        # Converts each frame to grayscale - we previously 
        # only converted the first frame to grayscale 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        
        # Calculates dense optical flow by Farneback method 
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, 
                                        None, 
                                        0.5, 3, 15, 3, 5, 1.2, 0) 
        # print(flow.shape)
        
        # Computes the magnitude and angle of the 2D vectors 
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1]) 
        
        # Sets image hue according to the optical flow 
        # direction 
        mask[..., 0] = angle * 180 / np.pi / 2
        
        # Sets image value according to the optical flow 
        # magnitude (normalized) 
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX) 
        
        # Converts HSV to RGB (BGR) color representation 
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2RGB) 

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        fgmask = mog.apply(frame)

        prev_gray = gray 

        # fgmask = cv2.resize(fgmask, (width_OF, fgmask.shape[0] * width_OF // fgmask.shape[1])) 
        # gray = cv2.resize(gray, (width_OF, gray.shape[0] * width_OF // gray.shape[1])) 
        # cv2.imshow('framev1',v1)
        # print(v1.shape)
        # print(fgmask.shape)
        # print(np.max(v1))
        # cv2.waitKey(0)
        # print(v1)
        # gray = gray / 255
        fmask = (gray * fgmask)

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

        fmask = pp1(fmask)
        
        cv2.imshow('frame',fmask)
        cv2.waitKey(0)

        cv2.imwrite(os.path.join(out_path, '%08d.png' % (frame_number + 1)), fmask)


    # cap.release() 
    cv2.destroyAllWindows() 
