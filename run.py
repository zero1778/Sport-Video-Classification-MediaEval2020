import numpy as np
import os, random, datetime, gc, argparse, logging
import torch
from torch.utils.data import DataLoader
from utils import make_path
from cfgs import Cfgs
from dataset import My_dataset
from model import make_architecture
from train_test import train_model, test_model

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--PATH_DATA', dest='PATH_DATA', type=str, help='Path to data folder')
    parser.add_argument('--PATH_MODEL', dest='PATH_MODEL', type=str, help='Path to model folder')
    parser.add_argument('--MODE', dest='MODE', type=str, default='train', choices=['train', 'test'], help='Mode')
    args = parser.parse_args()
    return args

#########################################################################
###################### Reset Pytorch Session ############################
#########################################################################
def reset_training(seed):
    gc.collect()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
        
#######################################################################
############################### Train #################################
#######################################################################
def train(__C):

    ######################
    ## Data preparation ##
    ######################
    ##### Build Dataset class and Data Loader #####
    train_set = My_dataset('train', __C)
    validation_set = My_dataset('val', __C)

    ## Loaders of the Datasets
    train_loader = DataLoader(train_set, batch_size=__C.BATCH_SIZE, shuffle=True, num_workers=__C.NUM_WORKERS)
    validation_loader = DataLoader(validation_set, batch_size=__C.BATCH_SIZE, shuffle=False, num_workers=__C.NUM_WORKERS)

    ##################
    ## Architecture ##
    ##################
    model = make_architecture(__C)

    ######################
    ## Training process ##
    ######################
    train_model(model, __C, train_loader, validation_loader)

#######################################################################
############################### Test ##################################
#######################################################################
def test(__C):

    #### Same seed #####
    reset_training(__C.SEED)
    
    ######################
    ## Data preparation ##
    ######################
    ##### Build Dataset class and Data Loader #####
    test_set = TableTennis('test', __C)
    ## Loaders of the Datasets
    test_loader = DataLoader(test_set, batch_size=__C.BATCH_SIZE, shuffle=False, num_workers=__C.NUM_WORKERS)

    ##################
    ## Architecture ##
    ##################
    model = make_architecture(__C)

    ##################
    ## Test process ##
    ##################
    test_model(model, __C, test_loader)

if __name__ == "__main__":
    # Variables
    __C = Cfgs()
    args = parse_args()
    args_dict = __C.parse_to_dict(args)

    __C.add_args(args_dict)
    __C.proc()

    logger.info('Train with configs:')
    logger.info(__C)

    #### Same seed #####
    reset_training(__C.SEED)
    make_path(__C.PATH_MODEL)
    if __C.MODE == 'test':
        __C.LOAD = True
        test(__C)
    else:
        train(__C)