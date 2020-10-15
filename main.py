import torch 

from train import train_model
from model import * 
from utils import *

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    n_classes = 21
    size_data = []
    make_the_model()
