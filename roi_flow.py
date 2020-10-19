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
                        help='Path of data videos', dest='inp')
    parser.add_argument('-m', '--fmethod', metavar='O', type=str,
                        help='Flow method', dest='method')

    return parser.parse_args()


def cal_roi(u_max, u_g, C, w_C, alpha=0.6):
    def func(u, C, w_C):
        return max(min(u, C - w_C/2), w_C/2)
    temp = alpha * func(u_max, C, w_C) + (1 - alpha) * func(u_g, C, w_C)
    return int(temp), temp

if __name__ == "__main__":

    # The video feed is read in as 
    # a VideoCapture object 
    args = get_args()
    path_rgb = args.inp + '/RGB/'
    flow_method = args.method
    # print(out_path)
    # exit(0)
    # cap = cv2.VideoCapture(path_rgb) 
    
    
    rgb_images = [f for f in sorted(listdir(path_rgb)) if isfile(join(path_rgb, f))]
    
    first_frame = cv2.imread(path_rgb + rgb_images[0])
    
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY) 

    # hog = cv2.HOGDescriptor()
    # hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # boxes, weights = hog.detectMultiScale(first_frame, winStride=(8,8) )
    # boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    # X_roi, Y_roi = 0,0
    # for (xA, yA, xB, yB) in boxes:
    #     # display the detected boxes in the colour picture
    #     X_roi , Y_roi = int((xA + xB)/2), int((yA + yB)/2)
    #     break

    # Creates an image filled with zero 
    # intensities with the same dimensions 
    # as the frame 
    mask = np.zeros_like(first_frame) 

    # Sets image saturation to maximum 
    mask[..., 1] = 255

    mog = cv2.createBackgroundSubtractorMOG2()
    
    W, H = 120, 120
    w_x, w_y = 180, 320
    for frame_number, list_frame in tqdm(enumerate(rgb_images[1:])):
        
        # ret = a boolean return value from getting 
        # the frame, frame = the current frame being 
        # projected in the video 
        # ret, frame = cap.read() 
        # frame = cv2.resize(frame, (width_OF, frame.shape[0] * width_OF // frame.shape[1]))
        # Opens a new window and displays the input 
        # frame 
        frame = cv2.imread(path_rgb + list_frame)
        fgmask = mog.apply(frame)
        # Converts each frame to grayscale - we previously 
        # only converted the first frame to grayscale 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        
        # Calculates dense optical flow by Farneback method 
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, 
                                        None, 
                                        0.5, 3, 15, 3, 5, 1.2, 0) 
        
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
        
        # cv2.imshow("Optical Flow", gray)
        # cv2.imshow("Foreground Extracted", fgmask)
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
        fmaskT = (fmask != 0).astype(int)  

        # Updates previous frame 
        prev_gray = gray 
        
        flow1 = np.sum(np.abs(flow), axis = -1) # 180 x 320 
        flow2 = flow1 * fmaskT

        idx_max = np.argmax(flow2)
        X_max, Y_max = idx_max // w_y, idx_max % w_y
        
    
        X_sum, Y_sum = 0, 0
        flow_num = (fmaskT != 0).astype(int)
        for x in range(180):
            for y in range(320):
                X_sum += x * flow_num[x][y]
                Y_sum += y * flow_num[x][y]
        flow_num = flow_num.sum()
    
        X_g, Y_g = int(X_sum/flow_num), int(Y_sum/flow_num)
        
        print("Gravity = ", X_g, Y_g)
        image = cv2.rectangle(fmask, (Y_g - 80,X_g - 60), (Y_g + 40,X_g + 60), color=(255), thickness=3)
        cv2.imshow('ROI Flow', image)
        cv2.waitKey(0)
        
        
        cv2.imwrite(os.path.join(out_path, '%08d.png' % (frame_number + 1)), fmask)

        
        # Frames are read by intervals of 1 millisecond. The 
        # programs breaks out of the while loop when the 
        # user presses the 'q' key 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    # The following frees up resources and 
    # closes all windows 
    # cap.release() 
    cv2.destroyAllWindows() 
