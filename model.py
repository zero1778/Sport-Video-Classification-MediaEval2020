import torch.nn as nn
import torch.nn.functional as F
import numpy as np

##########################################################################
########################  Flatten Features  ##############################
##########################################################################
def flatten_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

##########################################################################
####################### Twin RGB OptFlow ##############################
##########################################################################
class NetTwin(nn.Module):
    def __init__(self, size_data, n_classes):
        super(NetTwin, self).__init__()

        self.size_data = np.copy(size_data)

        ####################
        ####### First ######
        ####################
        self.conv1_RGB = nn.Conv3d(3, 30, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool1_RGB = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

        self.conv1_Flow = nn.Conv3d(2, 30, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool1_Flow = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

        self.size_data //= 2

        #####################
        ####### Second ######
        #####################
        self.conv2_RGB = nn.Conv3d(30, 60, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool2_RGB = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

        self.conv2_Flow = nn.Conv3d(30, 60, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool2_Flow = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

        self.size_data //= 2

        ####################
        ####### Third ######
        ####################
        self.conv3_RGB = nn.Conv3d(60, 80, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool3_RGB = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

        self.conv3_Flow = nn.Conv3d(60, 80, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool3_Flow = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

        self.size_data //= 2

        #####################
        ####### Fusion ######
        #####################
        self.linear_RGB = nn.Linear(80*self.size_data[0]*self.size_data[1]*self.size_data[2], 500)
        self.linear_Flow = nn.Linear(80*self.size_data[0]*self.size_data[1]*self.size_data[2], 500)

        self.linear = nn.Bilinear(500, 500, n_classes)
        self.final = nn.Softmax(1)

    def forward(self, rgb, flow):

        ####################
        ####### First ######
        ####################
        rgb = self.pool1_RGB(F.relu(self.conv1_RGB(rgb)))
        flow = self.pool1_Flow(F.relu(self.conv1_Flow(flow)))

        #####################
        ####### Second ######
        #####################
        rgb = self.pool2_RGB(F.relu(self.conv2_RGB(rgb)))
        flow = self.pool2_Flow(F.relu(self.conv2_Flow(flow)))

        ####################
        ####### Third ######
        ####################
        rgb = self.pool3_RGB(F.relu(self.conv3_RGB(rgb)))
        flow = self.pool3_Flow(F.relu(self.conv3_Flow(flow)))

        #####################
        ####### Fusion ######
        #####################
        rgb = rgb.view(-1, flatten_features(rgb))
        flow = flow.view(-1, flatten_features(flow))

        rgb = F.relu(self.linear_RGB(rgb))
        flow = F.relu(self.linear_Flow(flow))

        data = self.linear(rgb, flow)
        label = self.final(data)

        return label
        
##########################################################################
#############################  One Branch ################################
##########################################################################
class NetSimpleBranch(nn.Module):
    def __init__(self, size_data, n_classes, channels=3):
        super(NetSimpleBranch, self).__init__()

        self.channels = channels
        self.size_data = np.copy(size_data)
        
        ####################
        ####### First ######
        ####################
        self.conv1 = nn.Conv3d(channels, 30, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)) # dilaion=(0,0,0) (depth, height, width)
        self.pool1 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.size_data //= 2

        ####################
        ###### Second ######
        ####################
        self.conv2 = nn.Conv3d(30, 60, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)) # dilaion=(0,0,0) (depth, height, width)
        self.pool2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.size_data //= 2

        ####################
        ####### Third ######
        ####################
        self.conv3 = nn.Conv3d(60, 80, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)) # dilaion=(0,0,0) (depth, height, width)
        self.pool3 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.size_data //= 2

        ####################
        ####### Last #######
        ####################
        self.linear1 = nn.Linear(80*self.size_data[0]*self.size_data[1]*self.size_data[2], 500) # 144000, 216000
        self.relu = nn.ReLU()

        # Fusion
        self.linear2 = nn.Linear(500, n_classes)
        self.final = nn.Softmax(1)

    def forward(self, rgb, flow):
        if self.channels == 2:
            data = flow
        else:
            data = rgb

        ####################
        ####### First ######
        ####################
        data = self.pool1(F.relu(self.conv1(data))) # data = self.pool1(F.relu(self.drop1(self.conv1(data))))

        ####################
        ###### Second ######
        ####################
        data = self.pool2(F.relu(self.conv2(data)))

        ####################
        ####### Third ######
        ####################
        data = self.pool3(F.relu(self.conv3(data)))

        ####################
        ####### Last #######
        ####################
        data = data.view(-1, flatten_features(data))
        data = self.relu(self.linear1(data))

        data = self.linear2(data)
        label = self.final(data)

        return label

    
def make_architecture(__C):
    if __C.MODEL_TYPE == 'twin':
        model = NetTwin(__C.SIZE_DATA, __C.NUM_CLASSES)
    
    model.cuda()
    if __C.N_GPU > 1:
        model = nn.DataParallel(net, device_ids=__C.DEVICES)
    return model