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
    parser.add_argument('-o', '--output', metavar='O', type=str,
                        help='Path to RGB_CROP image', dest='out')

    return parser.parse_args()


def cal_roi(u_max, u_g, C, w_C, alpha=0.6):
    def func(u, C, w_C):
        return max(min(u, C - w_C/2), w_C/2)
    temp = alpha * func(u_max, C, w_C) + (1 - alpha) * func(u_g, C, w_C)
    return int(temp), temp

if __name__ == "__main__":
    args = get_args()
    path_rgb = args.inp + '/RGB/'
    flow_method = args.method
    out_path = args.out
    # cap = cv2.VideoCapture(path_rgb) 
    
    
    rgb_images = [f for f in sorted(listdir(path_rgb)) if isfile(join(path_rgb, f))]
    
    first_frame = cv2.imread(path_rgb + rgb_images[0])
    
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY) 
    mask = np.zeros_like(first_frame) 
    mask[..., 1] = 255

    mog = cv2.createBackgroundSubtractorMOG2()
    
    W, H, T = 120, 120, 100
    w_x, w_y = 180, 320
    opt = []
    total = len(rgb_images)
    ksize = 41
    sigma = 0.3*((ksize - 1)*0.5 - 1)+0.8
    # gaussian = cv2.getGaussianKernel(ksize, sigma)
    # print(len(gaussian))
    # print(gaussian)

    start_frame = max(int((total - T) / 2), 0)
    end_frame = min(start_frame + T, total - 1)
    # print("(Start, End) = (%d, %d)" % (start_frame, end_frame))
    
    for frame_number, list_frame in enumerate(rgb_images[1:]):
        
        if (frame_number > end_frame): break
        # ret = a boolean return value from getting 
        # the frame, frame = the current frame being 
        # projected in the video 
        # ret, frame = cap.read() 
        # frame = cv2.resize(frame, (width_OF, frame.shape[0] * width_OF // frame.shape[1]))
        # Opens a new window and displays the input 

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
        # gaussian_x = cv2.getGaussianKernel(W, sigma_W)
        # gaussian_y = cv2.getGaussianKernel(H, sigma_H)
        # gaussian_weight = gaussian_x * np.transpose(gaussian_y)
        # fgmask = np.multiply(fgmask, gaussian_weight)
        # cv2.imshow("Foreground",fgmask)

        fgmaskT = (fgmask != 0).astype(int)
        fgmaskT = np.expand_dims(fgmaskT, axis=-1)
 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, 
                                        None, 
                                        0.5, 3, 15, 3, 5, 1.2, 0) 
        flow_filter = np.multiply(flow, fgmaskT)

        magnitude, angle = cv2.cartToPolar(flow_filter[..., 0], flow_filter[..., 1]) 
        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX) 
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2RGB) 
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) 
        
        # cv2.imshow("Flow", gray)
        # # cv2.imshow("Optical Flow", gray)
        # cv2.imshow("Optical Filter", gray)
        # fmask = (gray * fgmask)
        # fmask = pp1(fmask)
        fmaskT = (gray != 0).astype(int)  

        # Updates previous frame 
        prev_gray = gray 
        
        ###################################
        # flow1 = np.sum(np.abs(flow), axis = -1) # 180 x 320 
        # flow2 = flow1 * fmaskT

        # idx_max = np.argmax(flow2)
        # X_max, Y_max = idx_max // w_y, idx_max % w_y
        ###################################
    
        X_sum, Y_sum = 0, 0
        flow_num = (fmaskT != 0).astype(int)
        delta = (flow_num == 1)
        row, col = np.indices((180, 320))
        X_g = np.sum(row*delta) // np.sum(delta)
        Y_g = np.sum(col*delta) // np.sum(delta)
        # for x in range(180):
        #     for y in range(320):
        #         X_sum += x * flow_num[x][y]
        #         Y_sum += y * flow_num[x][y]
        # flow_num = flow_num.sum()
    
        #Calculate Gravity
        # X_g, Y_g = int(X_sum/flow_num), int(Y_sum/flow_num)
        # print("Gravity = ", X_g, Y_g)

        X_tl, Y_tl =  max(0, X_g - 60), max(0, Y_g - 75)

        if (X_tl + W > w_x): X_tl = w_x - W
        if (Y_tl + H > w_y): Y_tl = w_y - H

        crop_image = frame[X_tl: X_tl + W, Y_tl : Y_tl + H]
        crop_opt   = flow_filter[X_tl: X_tl + W, Y_tl : Y_tl + H]
        
        # gray = np.multiply(gray, gaussian_weight)
        # np.save('values_flow_%s' % flow_method, crop_opt)
        # # print(crop_image.shape)
        # cv2.imshow("Flow_filter", gray)
    
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(out_path, '%08d.png' % (frame_number + 1)), crop_image)

        # image = cv2.rectangle(frame, (Y_tl, X_tl), (Y_tl + H, X_tl + W), color=(0,0,255), thickness=2)

        if (frame_number >= start_frame and frame_number < end_frame):
            # cv2.imshow("Cropped",crop_image)
            # cv2.imshow('ROI Flow', crop_opt)
            opt.append(crop_opt)
        # cv2.waitKey(0) 
        
        # cv2.imwrite(os.path.join(out_path, '%08d.png' % (frame_number + 1)), fmask)
        # if cv2.waitKey(1) & 0xFF == ord('q'): 
        #     break
    N = len(opt)
    if (N < 100):
        delta = int((100 - N) / 2)
        first = opt[0]
        last = opt[len(opt) - 1]
        for i in range (0, delta):
            opt = [first] + opt
            opt.append(last)
        while len(opt) < 100:
            opt.append(last)
    opt = np.array(opt)
    # print(opt.shape)
    # print(args.inp)
    np.save(args.inp + "/values_flow_%s" % (flow_method) , opt)
    
          

    # cap.release() 
    # cv2.destroyAllWindows() 
