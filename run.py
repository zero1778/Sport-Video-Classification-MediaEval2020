import numpy as np
import os, random, datetime, gc, argparse, logging
import torch
from torch.utils.data import DataLoader
from utils import make_path
from cfgs import Cfgs
from dataset import My_dataset
from model import make_architecture
from train_test import train_model, test_model, val_model

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', dest='PATH_PROCESSED_DATA', type=str, help='Path to data folder')
    parser.add_argument('-s', '--save', dest='SAVE', type=str, help='Name of model save')
    parser.add_argument('-lp', '--load_p', dest='LOAD_PRETRAINED', type=str, help='Name of model pretrained')
    parser.add_argument('-m', '--mode', dest='MODE', type=str, default='train', choices=['train', 'test', 'val'], help='Mode')
    parser.add_argument('-e', '--epoch', dest='EPOCHS', type=int, default=100, help='Number of epochs')
    parser.add_argument('-lr', dest='LR', type=float, default=0.001, help='learning_rate')
    parser.add_argument('-bs', '--batch_size', dest='BATCH_SIZE', type=int, default=2, help='Batch size')
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
    validation_loader = DataLoader(validation_set, batch_size=2, shuffle=False, num_workers=__C.NUM_WORKERS)

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
    test_set = My_dataset('test', __C)
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

def validate(__C):

    #### Same seed #####
    reset_training(__C.SEED)
    
    ######################
    ## Data preparation ##
    ######################
    ##### Build Dataset class and Data Loader #####
    val_set = My_dataset('val', __C)
    ## Loaders of the Datasets
    val_loader = DataLoader(val_set, batch_size=__C.BATCH_SIZE, shuffle=False, num_workers=__C.NUM_WORKERS)

    ##################
    ## Architecture ##
    ##################
    model = make_architecture(__C)

    ##################
    ## Test process ##
    ##################
    val_model(model, __C, val_loader)

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
    # __C.LOAD = True
    if __C.MODE == 'test':
        test(__C)
    else:
        if __C.MODE == 'val':
            validate(__C)
        else:
            train(__C)
