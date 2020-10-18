from utils import make_path

############################################################
##################### Build the data #######################
############################################################
def build_data(video_list, save_path, width_OF=320, log=None, workers=15, flow_method='DeepFlow'):
    make_path(save_path)

    # Extract Frames
    extract_frames(video_list, save_path, width_OF, log)
    

    # Compute DeepFlow
    compute_DeepFlow(video_list, save_path, log, workers)

    # Compute ROI
    compute_ROI(video_list, save_path, log, workers, flow_method=flow_method)


##################### RGB #######################
def extract_frames(video_list, save_path, width_OF, log):
    # Chrono
    start_time = time.time()

    for idx, video_path in enumerate(video_list):
        
        video_name = os.path.basename(video_path)
        # progress_bar(idx, len(video_list), 'Frame extraction - %s' % (video_name))

        path_data_video = os.path.join(save_path, video_name.split('.')[0])
        make_path(path_data_video)
        path_RGB = os.path.join(path_data_video, 'RGB')
        make_path(path_RGB)

        # Load Video
        cap = cv2.VideoCapture(video_path)
        length_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_number = 0
        
        # Check if video uploaded
        if not cap.isOpened():
            sys.exit("Unable to open the video, check the path.\n")

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
    DeepFlow_pool = ActivePool()

    for idx, video_path in enumerate(video_list):

        video_name = os.path.basename(video_path).split('.')[0]
        path_data_video = os.path.join(save_path, video_name)

        # Split the calculation in severals process
        while threading.activeCount() > workers:
            # progress_bar(idx + 1 - threading.activeCount(), len(video_list), 'DeepFlow computation')
            time.sleep(0.1)

        if threading.activeCount() <= workers:
            job = threading.Thread(target = compute_DeepFlow_video, name = idx, args = (DeepFlow_pool,
                                                                                        os.path.join(path_data_video, 'RGB'),
                                                                                        os.path.join(path_data_video, 'DeepFlow')))
            job.daemon=True
            job.start()

    while threading.activeCount()>1:
        # progress_bar(idx + 1 - threading.activeCount(), len(video_list), 'DeepFlow computation')
        time.sleep(0.1)

    # progress_bar(idx + 1, len(video_list), 'DeepFlow computation done in %d s' % (time.time() - start_time), 1, log=log)


def compute_DeepFlow_video(pool, path_RGB, path_Flow):
    name = threading.current_thread().name
    pool.makeActive(name)
    os.system('python cv_flow.py -i %s -o %s' % (path_RGB, path_Flow))
    pool.makeInactive(name)


##################### ROI #######################
def compute_ROI(video_list, save_path, log, workers, flow_method='DeepFlow'):
    start_time = time.time()
    ROI_pool = ActivePool()

    for idx, video_path in enumerate(video_list):

        video_name = os.path.basename(video_path).split('.')[0]
        path_data_video = os.path.join(save_path, video_name)

        # Split the calculation in severals process
        while threading.activeCount() > workers:
            # progress_bar(idx + 1 - threading.activeCount(), len(video_list), 'ROI computation for %s' % (flow_method))
            time.sleep(0.1)

        if threading.activeCount() <= workers:
            job = threading.Thread(target = compute_roi_video, name = idx, args = (ROI_pool,
                                                                                   path_data_video,
                                                                                   flow_method))
            job.daemon=True
            job.start()

    while threading.activeCount()>1:
        # progress_bar(idx + 1 - threading.activeCount(), len(video_list), 'ROI computation for %s' % (flow_method))
        time.sleep(0.1)

    join_values_flow(video_list, 'values_flow_%s' % flow_method)
    # progress_bar(len(video_list), len(video_list), 'ROI computation for %s completed in %d s' % (flow_method, int(time.time() - start_time)), 1, log=log)


def compute_roi_video(pool, path_data_video, flow_method):
    name = threading.current_thread().name
    pool.makeActive(name)
    os.system('python roi_flow.py -v %s -m %s' % (path_data_video, flow_method))
    pool.makeInactive(name)


def join_values_flow(video_list, name_values, save_path):
    values_flow = []
    for video in video_list:
        video_name = os.path.basename(video_path).split('.')[0]
        path_data_video = os.path.join(save_path, video_name)
        values_flow_video = np.load(os.path.join(path_data_video, '%s.npy' % name_values))
        values_flow.extend(values_flow_video)
    np.save(os.path.join(save_path, name_values), values_flow)

if __name__ == "__main__":
    build_data(video_list, save_path, width_OF=320, log=None, workers=15, flow_method='DeepFlow'):