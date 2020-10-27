import os, time, datetime, logging, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import make_train_figure, progress_bar
from tqdm import tqdm

logger = logging.getLogger(__name__)

##########################################################################
######################## Save and Load Model #############################
##########################################################################

def save_model(model, __C, optimizer, epoch, dict_of_values):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dict_of_values': dict_of_values,
                'cfgs': __C.state_dict()}, os.path.join(__C.PATH_MODEL, 'model.tar'))

def load_model(model, weigth_path, optimizer=None):
    checkpoint = torch.load(os.path.join(weigth_path, 'model.tar'), map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    dict_of_values = checkpoint['dict_of_values']
    cfgs_dict = checkpoint['cfgs']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return epoch, dict_of_values, cfgs_dict


##########################################################################
########################### Training Process #############################
##########################################################################
def train_epoch(epoch, __C, model, data_loader, optimizer, criterion):
    model.train()
    N = len(data_loader.dataset)
    start_time = time.time()
    aLoss = 0
    Acc = 0

    for batch_idx, batch in enumerate(tqdm(data_loader)):
        # Get batch tensor
        rgb, flow, label = batch['rgb'], batch['flow'], batch['label']

        rgb = rgb.cuda()
        flow = flow.cuda()
        label = label.cuda()

        optimizer.zero_grad()
        output = model(rgb, flow)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        aLoss += loss.item()
        Acc += output.data.max(1)[1].eq(label.data).cpu().sum().numpy()

    aLoss /= (batch_idx + 1)
    Acc /= N

    logger.info('Training: [Epoch %4d] loss: %.4f accuracy: %.4f lr: %.6f' % (epoch, aLoss, Acc, __C.LR))

    return aLoss, Acc


##########################################################################
######################## Validation Process ##############################
##########################################################################
def validation_epoch(epoch, __C, model, data_loader, criterion):
    model.eval()
    N = len(data_loader.dataset)
    aLoss = 0
    Acc = 0

    for batch_idx, batch in enumerate(tqdm(data_loader)):
        # Get batch tensor
        rgb, flow, label = batch['rgb'], batch['flow'], batch['label']

        rgb = rgb.cuda()
        flow = flow.cuda()
        label = label.cuda()

        output = model(rgb, flow)
        pred = output.cpu().data.numpy()
        pred_argmax = np.argmax(pred, axis=1)

        aLoss += criterion(output, label).item()
        Acc += output.data.max(1)[1].eq(label.data).cpu().sum().numpy()

    aLoss /= (batch_idx + 1)
    Acc /= N

    logger.info('Evaluation: [Epoch %4d] loss: %.4f accuracy: %.4f' % (epoch, aLoss, Acc))

    return pred_argmax, aLoss, Acc


##########################################################################
############################# TRAINING ###################################
##########################################################################
def train_model(model, __C, train_loader, validation_loader):
    start_time = time.time()

    criterion = nn.CrossEntropyLoss() # change with reduction='sum' -> lr to change
    optimizer = optim.SGD(
        model.parameters(), 
        lr=__C.LR, 
        momentum=__C.MOMENTUM, 
        weight_decay=__C.DECAY, 
        nesterov=__C.NESTEROV
    )

    # For plot
    loss_train = []
    loss_val = []
    acc_val = []
    acc_train = []
    max_acc = -1
    acc_val_ = 1
    min_loss_train = 1000
    min_loss_val = 1000
    epoch_start = 1

    if __C.LOAD:
        logger.info('Load model %s for retraining' % (__C.PATH_MODEL))
        epoch_start, dict_of_values, cfgs_dict = load_model(model, __C.PATH_MODEL, optimizer=optimizer)
        __C.add_args(cfgs_dict)
        logger.info('Model from epoch %d' % (epoch))
        max_acc = dict_of_values['acc_val_']
        min_loss_val = dict_of_values['loss_val_']
        for key in dict_of_values:
            logger.info('%s : %g' % (key, dict_of_values[key]))
        #change_optimizer(optimizer, __C, lr=__C.lr_max)

    for epoch in range(epoch_start, __C.EPOCHS+1):

        # Train and validation step and save loss and acc for plot
        loss_train_, acc_train_ = train_epoch(epoch, __C, model, train_loader, optimizer, criterion)
        _, loss_val_, acc_val_ = validation_epoch(epoch, __C, model, validation_loader, criterion)

        loss_train.append(loss_train_)
        acc_train.append(acc_train_)
        loss_val.append(loss_val_)
        acc_val.append(acc_val_)

        #wait_change_lr += 1

        # Best model saved
        # if (acc_val_ > max_acc) or (acc_val_ >= max_acc and loss_train_ < min_loss_train):
        if min_loss_val > loss_val_:
            save_model(model, __C, optimizer=optimizer, epoch=epoch, dict_of_values={'loss_train_': loss_train_, 'acc_train_': acc_train_, 'loss_val_': loss_val_, 'acc_val_': acc_val_})
            max_acc = acc_val_
            min_loss_val = loss_val_
            min_loss_train = loss_train_


    logger.info('Trained with %d epochs, lr = %g, batchsize = %d, momentum = %g with max validation accuracy of %.2f done in %s' %\
        (__C.EPOCHS, __C.LR, __C.BATCH_SIZE, __C.MOMENTUM, max_acc, datetime.timedelta(seconds=int(time.time() - start_time))))

    make_train_figure(loss_train, loss_val, acc_train, acc_val, os.path.join(__C.PATH_MODEL, 'Train.png'))


##########################################################################
############################# TESTING ####################################
##########################################################################
def test_model(model, __C, test_loader):
    start_time = time.time()

    criterion = nn.CrossEntropyLoss() # change with reduction='sum' -> lr to change

    logger.info('Load model %s for testing' % (__C.PATH_MODEL))
    epoch, dict_of_values, cfgs_dict = load_model(model, __C.PATH_MODEL)
    __C.add_args(cfgs_dict)
    logger.info('Model from epoch %d' % (epoch))
    for key in dict_of_values:
        logger.info('%s : %g' % (key, dict_of_values[key]))

    pred, loss_test_, acc_test_ = validation_epoch(epoch, __C, model, test_loader, criterion)

    logger.info('Done in %s\nAccuracy: %.2f Loss: %.6f' % (datetime.timedelta(seconds=int(time.time() - start_time, acc_test_, loss_test_))))

    # Save results
    with open(__C.OUTPUT_FILE, 'rb') as f:
        pickle.dump(pred, f)