from torch.utils.data import Dataset, DataLoader
from model import NetSimpleBranch
import os

def make_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        
#########################################################################
###################### Reset Pytorch Session ############################
#########################################################################
def reset_training(seed):
    gc.collect()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
#################################################################
###################### Model variables ##########################
#################################################################
class my_variables():
    def __init__(self, model_type='Twin', batch_size=10, augmentation=True, nesterov=True, decay=0.005, epochs=500, lr=0.001, momentum=0.5, flow_method='DeepFlow', norm_method='NORMAL', size_data=[100, 120, 120], cuda=True):

        self.model_type = model_type
        self.augmentation = augmentation
        self.nesterov = nesterov
        self.decay = decay
        self.lr = lr
        self.momentum = momentum
        self.flow_method = flow_method
        self.norm_method = norm_method
        self.size_data = np.array(size_data)
        self.model_name = 'pytorch_%s_%s_bs_%s_aug_%d_nest_%d_decay_%s_lr_%s_m_%s_OF_%s_%s_sizeinput_%s_' % (datetime.datetime.now().strftime("%d-%m-%Y_%H-%M"), self.model_type, self.batch_size, self.augmentation, self.nesterov, self.decay, self.lr, self.momentum, self.flow_method, self.norm_method, str(self.size_data))
        self.load = False

        self.epochs = epochs
        self.path_fig_model = os.path.join('Figures', self.model_name)
        make_path(self.path_fig_model)

        if cuda: #Use gpu
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        self.log = setup_logger('model_log', os.path.join(self.path_fig_model, 'log_%s.log' % datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")))

    def state_dict(self):
        dict = self.__dict__.copy()
        del dict['log']
        return dict


# def get_annotation_data(dataset_list,size_data):
    
##########################################################################
############################ Dataset Class ###############################
##########################################################################
class My_dataset(Dataset):
    # def __init__(self, dataset_list, size_data, augmentation=0, norm_method = norm_method, flow_method = flow_method):
    def __init__(self, dataset_list, size_data, augmentation=0, norm_method = 'norm_method', flow_method = 'flow_method'):
        self.dataset_list = dataset_list
        self.augmentation = augmentation
        self.norm_method = norm_method
        self.flow_method = flow_method

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        ###TODO###
        rgb, flow, label = get_annotation_data(self.dataset_list[idx], self.size_data, augmentation = self.augmentation, norm_method = self.norm_method, flow_method = self.flow_method)
        # rgb, flolabel = get_annotation_data(self.dataset_list[idx], self.size_data, augmentation = self.augmentation, norm_method = self.norm_method)
        sample = {'rgb': torch.FloatTensor(rgb), 'flow': torch.FloatTensor(flow), 'label': label}
        return sample

        
def build_lists_set(save_path):
    # for d in os.listdir(data_dir):
        # cur_dir = os.path.join(data_dir, d)
        # if d in ['train', 'val']:
        #     for id, label in enumerate(sorted(os.listdir(cur_dir))):
        #         for vid_name in sorted(os.listdir(os.path.join(cur_dir, label))):

    for d in os.listdir(save_path)


build_lists_set("data")