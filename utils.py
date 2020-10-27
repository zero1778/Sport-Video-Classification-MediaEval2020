from torch.utils.data import Dataset, DataLoader
from model import NetSimpleBranch
import os, cv2
import matplotlib.pyplot as plt


def make_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)

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

# build_lists_set("data")

def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def progress_bar(idx, size, description):
    print('Hello')


def make_train_figure(loss_train, loss_val, acc_train, acc_val, path):
    plt.plot(loss_train, label='Loss Train')
    plt.plot(loss_val, label='Loss Val')
    plt.plot(acc_train, label='Acc Train')
    plt.plot(acc_val, label='Acc Val')
    plt.legend()
    plt.savefig(path)
    plt.show()