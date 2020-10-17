import torch 

from train import train_model
from model import * 
from utils import *


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    #### Same seed #####
    reset_training(param.seed)

    ##### Get all the annotations in random order #####
    ###TODO###
    # annotations_list, negative_list = get_annotations_list(video_list)

    ##### Build Train, Validation and Test set #####
    ###TODO###
    train_list, validation_list, test_list = build_lists_set(data_dir)

    # Variables
    lr = 0.01

    # compute_normalization_values(os.path.join(param.path_data, 'values_flow_%s.npy' % 'DeepFlow'))
    
    args = my_variables()

    ##################
    ## Architecture ##
    ##################
    model = NetSimpleBranch(args)

    ######################
    ## Data preparation ##
    ######################
    ##### Build Dataset class and Data Loader #####
    train_set = My_dataset(train_list, augmentation = args.augmentation, norm_method = args.norm_method, flow_method = args.flow_method, data_types = args.model_type, fps=args.fps, size_data=args.size_data)
    validation_set = My_dataset(validation_list, norm_method = args.norm_method, flow_method = args.flow_method, data_types = args.model_type, fps=args.fps, size_data=args.size_data)
    test_set = My_dataset(test_list, norm_method = args.norm_method, flow_method = args.flow_method, data_types = args.model_type, fps=args.fps, size_data=args.size_data)

    ## Loaders of the Datasets
    train_loader = DataLoader(train_set, batch_size = args.batch_size, shuffle = True, num_workers = args.workers)
    validation_loader = DataLoader(validation_set, batch_size = args.batch_size, shuffle = False, num_workers = args.workers)
    test_loader = DataLoader(test_set, batch_size = args.batch_size, shuffle = False, num_workers = args.workers)

    ######################
    ## Training process ##
    ######################
    train_model(model, args, train_loader, validation_loader)
    args.load = True

    ## Load best model
    # model = NetSimpleBranch(args)

    ##################
    ## Test process ##
    ##################
    # test_model(model, args, test_loader, param.list_of_moves)