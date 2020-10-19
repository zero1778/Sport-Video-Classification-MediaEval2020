import torch 

from train import train_model
from model import * 
from utils import *

def build_lists_set():
    pass

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    #### Same seed #####
    reset_training(param.seed)

    ##### Get all the annotations in random order #####
    ###TODO###
    # annotations_list, negative_list = get_annotations_list(video_list)

    ##### Build Train, Validation and Test set #####
    ###TODO###
    train_list, validation_list, _ = build_lists_set(data_dir)

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