import os, sys, logging, functools
import torch
import numpy as np
from utils import make_path
from types import MethodType

@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return open(filename, "a")


class Cfgs():
    def __init__(self):

        # Set Devices
        # If use multi-gpu training, set e.g.'0, 1, 2' instead
        self.GPU = '0'
        # Resume training
        self.LOAD = False
        # Set RNG For CPU And GPUs
        self.SEED = 31#random.randint(0, 99999999)
        self.VERSION = str(self.SEED)
        self.OUTPUT_DIR = '.'
        self.OUTPUT_FILE = os.path.join(self.OUTPUT_DIR, 'pred.txt')

        # ------------------------------
        # ---- Data Provider Params ----
        # ------------------------------

        # {'train', test'}
        self.MODE = 'train'
        self.AUGMENTATION = False
        self.FLOW = 'DeepFlow'
        self.NORM = 'normal'
        self.SIZE_DATA = np.array([100, 120, 120])
        self.PATH_DATA = 'data'
        self.PATH_PROCESSED_DATA = 'data_processed'
        self.LABEL_DICT = 'label_dict.json'
        self.DATA_JSON = 'data.json'
        self.NUM_WORKERS = 8
        self.DATA_SEPARATOR = ','

        # --------------------------
        # ---- Optimizer Params ----
        # --------------------------
        self.NESTEROV = True
        self.DECAY = 0.005
        self.LR = 0.001
        self.MOMENTUM = 0.5

        # ------------------------
        # ---- Network Params ----
        # ------------------------

        self.MODEL_TYPE = 'twin'
        self.NUM_CLASSES = 20
        self.BATCH_SIZE = 2
        self.EPOCHS = 100
        self.MODEL_NAME = '%s' % (self.MODEL_TYPE)
        self.PATH_MODEL = os.path.join(self.OUTPUT_DIR, self.MODEL_NAME)

        self.setup_logger()


    def setup_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        plain_formatter = logging.Formatter(
            "[%(asctime)s][%(levelname)s] %(filename)s: %(lineno)3d: %(message)s",
            datefmt="%m/%d %H:%M:%S",
        )

        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(plain_formatter)
        logger.addHandler(ch)

        filename = os.path.join(self.OUTPUT_DIR, 'stdout.log')
        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)


    def state_dict(self):
        dict = self.__dict__.copy()
        return dict


    def parse_to_dict(self, args):
        args_dict = {}
        for arg in dir(args):
            if not arg.startswith('_') and not isinstance(getattr(args, arg), MethodType):
                if getattr(args, arg) is not None:
                    args_dict[arg] = getattr(args, arg)

        return args_dict


    def add_args(self, args_dict):
        for arg in args_dict:
            setattr(self, arg, args_dict[arg])


    def __str__(self):
        result = '\n'
        for attr in dir(self):
            if not attr.startswith('__') and not isinstance(getattr(self, attr), MethodType):
                result += '{ %-17s }-> %s\n' % (attr, getattr(self, attr))
        return result

    def proc(self):
        assert self.MODE in ['train', 'val', 'test']

        # ------------ Devices setup
        os.environ['CUDA_VISIBLE_DEVICES'] = self.GPU
        self.N_GPU = len(self.GPU.split(','))
        self.DEVICES = [_ for _ in range(self.N_GPU)]
        torch.set_num_threads(2)
