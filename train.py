from torch.autograd import Variable

##########################################################################
######################## Save and Load Model #############################
##########################################################################
def save_model(model, args, optimizer, epoch, dict_of_values):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dict_of_values': dict_of_values,
                'args': args.state_dict()}, os.path.join(args.path_fig_model, 'model.tar'))

def load_model(model, weigth_path, optimizer=None):
    checkpoint = torch.load(os.path.join(weigth_path, 'model.tar'), map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    dict_of_values = checkpoint['dict_of_values']
    args_dict = checkpoint['args']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return epoch, dict_of_values, args_dict

##########################################################################
########################### Training Process #############################
##########################################################################
def train_epoch(epoch, args, model, data_loader, optimizer, criterion):
    model.train()
    N = len(data_loader.dataset)
    start_time = time.time()
    aLoss = 0
    Acc = 0

    for batch_idx, batch in enumerate(data_loader):
        # Get batch tensor
        rgb, flow, label = batch['rgb'], batch['flow'], batch['label']

        rgb = Variable(rgb.type(args.dtype))
        flow = Variable(flow.type(args.dtype))
        label = Variable(label.type(args.dtype).long())

        optimizer.zero_grad()
        output = model(rgb, flow)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        aLoss += loss.item()
        Acc += output.data.max(1)[1].eq(label.data).cpu().sum().numpy()

    aLoss /= (batch_idx + 1)
    return aLoss, Acc/N


##########################################################################
######################## Validation Process ##############################
##########################################################################
def validation_epoch(epoch, args, model, data_loader, criterion):
    with torch.no_grad():
        N = len(data_loader.dataset)
        aLoss = 0
        Acc = 0

        for batch_idx, batch in enumerate(data_loader):
            # Get batch tensor
            rgb, flow, label = batch['rgb'], batch['flow'], batch['label']

            rgb = Variable(rgb.type(args.dtype))
            flow = Variable(flow.type(args.dtype))
            label = Variable(label.type(args.dtype).long())

            output = model(rgb, flow)

            aLoss += criterion(output, label).item()
            Acc += output.data.max(1)[1].eq(label.data).cpu().sum().numpy()

            progress_bar((batch_idx + 1) * args.batch_size, N, '%d - Validation' % (pid))

        aLoss /= (batch_idx + 1)

        return aLoss, Acc/N

##########################################################################
############################# TRAINING ###################################
##########################################################################
def train_model(model, args, train_loader, validation_loader):
    criterion = nn.CrossEntropyLoss() # change with reduction='sum' -> lr to change
    optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.decay, nesterov = args.nesterov)

    # For plot
    loss_train = []
    loss_val = []
    acc_val = []
    acc_train = []
    max_acc = -1
    acc_val_ = 1
    min_loss_train = 1000
    min_loss_val = 1000

    if args.load:
        print_and_log('Load previous model for retraining', log=args.log)
        epoch, dict_of_values, _ = load_model(model, args.path_fig_model, optimizer=optimizer)
        print_and_log('Model from epoch %d' % (epoch), log=args.log)
        max_acc = dict_of_values['acc_val_']
        min_loss_val = dict_of_values['loss_val_']
        for key in dict_of_values:
            print_and_log('%s : %g' % (key, dict_of_values[key]), log=args.log)
        change_optimizer(optimizer, args, lr=args.lr_max)

    for epoch in range(1, args.epochs+1):

        # Train and validation step and save loss and acc for plot
        loss_train_, acc_train_ = train_epoch(epoch, args, model, train_loader, optimizer, criterion)
        loss_val_, acc_val_ = validation_epoch(epoch, args, model, validation_loader, criterion)

        loss_train.append(loss_train_)
        acc_train.append(acc_train_)
        loss_val.append(loss_val_)
        acc_val.append(acc_val_)

        wait_change_lr += 1

        # Best model saved
        # if (acc_val_ > max_acc) or (acc_val_ >= max_acc and loss_train_ < min_loss_train):
        if min_loss_val > loss_val_:
            save_model(model, args, optimizer=optimizer, epoch=epoch, dict_of_values={'loss_train_': loss_train_, 'acc_train_': acc_train_, 'loss_val_': loss_val_, 'acc_val_': acc_val_})
            max_acc = acc_val_
            min_loss_val = loss_val_
            min_loss_train = loss_train_


    print_and_log('Trained with %d epochs, lr = %g, batchsize = %d, momentum = %g with max validation accuracy of %.2f done in %ds' %\
        (args.epochs, args.lr, args.batch_size, args.momentum, max_acc, time.time() - start_time), log=args.log)

    make_train_figure(loss_train, loss_val, acc_val, acc_train, os.path.join(args.path_fig_model, 'Train.png'))