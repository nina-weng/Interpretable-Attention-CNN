import copy

import numpy as np
import logging
import math
import random
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from config import config
import os
import torch
from model import AttentionCNN
CNNMultiTask = AttentionCNN
# from model.GCN.GCNAdj import GCNAdj

from model.utils.dataloader import create_dataloader

from model.loss_functions.WingLoss import WingLoss
from model.loss_functions.WeightedMSELoss import WeightedMSELoss
from model.loss_functions.WeightedL1Loss import WeightedL1Loss
from model.loss_functions.DistrancedMSELoss import DistancedMSELoss
from model.loss_functions.AngleLoss import AngleLoss
from model.loss_functions.CosLoss import CosLoss

from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.data_augmentation import data_augmentation
from data_pre.preprocessing import deal_wired_electrodes

def split_basedblock(ids, blocks):
    # based on block
    # block 0,1,2 as training, block 3 as validation, block 4 as testing (block 4 == block 0)
    ## both validation and testing show loss & score, no worries!
    BLOCKs =np.unique(blocks)
    assert len(BLOCKs) ==5
    print('BLOCKs: {}'.format(BLOCKs))

    train = np.isin(blocks, [0.0,1.0,2.0])
    val = np.isin(blocks, [3.0])
    test = np.isin(blocks, [4.0])


    return train, val, test


def convert2binary(indexes,num_class=129):
    '''
    indexes: shape (bs, k) , k is the defined number for detection
    '''
    bs = indexes.shape[0]
    new_indexes = []
    for i in range(bs):
        tmp = np.zeros(num_class)
        tmp[indexes[i]] = 1.0
        new_indexes.append(tmp)

    new_indexes = torch.tensor(new_indexes).float()
    if torch.cuda.is_available():
        new_indexes = new_indexes.cuda()

    return new_indexes

def one_hot(a, num_classes):
    a = a.type(torch.int32).cpu()
    np_ = np.squeeze(np.eye(num_classes)[a.reshape(-1)])
    if torch.cuda.is_available():
        return torch.tensor(np_,dtype=torch.long).cuda()
    else:
        return torch.tensor(np_,dtype=torch.long)

def data_normalization(data,data_name=None):
    mean_ = np.mean(data)
    std_ = np.std(data)
    logging.info('-----data normalization-----')
    logging.info('{}\tmean: {:.3f}\tstd: {:.3f}'.format(data_name,mean_,std_))
    return (data-mean_)/std_

def split(ids, train, val, test,removeOutlier=False,threshold=500):
    assert (train+val+test == 1)
    IDs = np.unique(ids)
    num_ids = len(IDs)
    logging.info('Number of unique IDs: {}'.format(num_ids))

    # priority given to the test/val sets
    test_split = math.ceil(test * num_ids)
    val_split = math.ceil(val * num_ids)
    train_split = num_ids - val_split - test_split

    train = np.isin(ids, IDs[:train_split])
    val = np.isin(ids, IDs[train_split:train_split+val_split])
    test = np.isin(ids, IDs[train_split+val_split:])

    # removeOutlier = True
    if removeOutlier == 'yes':
        logging.info(f'Removing outliers...')
        train = [i for i, x in enumerate(train) if x]
        val = [i for i, x in enumerate(val) if x]
        test = [i for i, x in enumerate(test) if x]

        # get outlier indexes
        # dir = 'D:\\ninavv\\master\\thesis\\data\\'
        if torch.cuda.is_available():
            dir = config['data_dir']
            # dir = '/cluster/work/hilliges/niweng/deepeye/data/'
        else:
            dir = 'D:\\ninavv\\master\\thesis\\data\\'
        outlier_index_npz_filepath = os.path.join(dir, 'analysis/dropout_trail_index_'+str(threshold)+'.npz')
        outlier_index = np.load(outlier_index_npz_filepath)['record']

        train = np.array(list(set(train) - set(outlier_index)))
        val = np.array(list(set(val) - set(outlier_index)))
        # test = np.array(list(set(test) - set(outlier_index))) # should not remove the data in test set
        logging.info(f'Removing outliers finished.')

    return train, val, test

def similarity_score(a,b):
    '''
    a: shape (batchsize, 1, input_shape[0],input_shape[1])
    b: shape (batchsize, 1, input_shape[0],input_shape[1])
    '''
    batch_size = a.shape[0]
    a = np.squeeze(a, axis=1)
    b = np.squeeze(b, axis=1)

    similarity_score_list = []
    for i in range(batch_size):
        sc = (a[i] @ b[i].T) / (np.linalg.norm(a[i]) * np.linalg.norm(b[i]))
        similarity_score_list.append(sc)
    print(len(similarity_score_list))
    print(len(similarity_score_list[0]))
    return sum(similarity_score_list)/len(similarity_score_list) # BUG!


def get_nearest_point(dots_positions,pos):
    distances = [(each[0]-pos[0])**2+(each[1]-pos[1])**2 for each in dots_positions]
    return np.argmin(distances)



def converting_ylabels(y_labels):
    dots_positions = [(400,300),(650,500),(400,100),(100,450),(700,450),(100,500),(200,350),(300,400),
                      (100,150),(150,500),(150,100),(700,100),(300,200),(100,100),(700,500),(500,400),
                      (600,250),(650,100),(200,250),(400,500),(700,150),(500,200),(100,300),(700,300),
                      (600,350)]
    new_y_labels = []
    if len(y_labels[0]) == 3:
        for each in y_labels:
            new_y_labels.append([each[0],get_nearest_point(dots_positions,each[1:])])
    else:
        for each in y_labels:
            new_y_labels.append(get_nearest_point(dots_positions,each))
    return np.array(new_y_labels)


def remove_pos(X,y,remain_pos):
    y_as_class = converting_ylabels(y)
    if len(y_as_class.shape) == 1:
        indexes = np.where(np.isin(y_as_class,remain_pos))[0]
    elif len(y_as_class.shape == 2):
        indexes = np.where(np.isin(y_as_class[:,1], remain_pos))[0]
    X = X[indexes]
    y = y[indexes]
    return X,y



def replace_electrodes(eeg_signal,ind_electrodes,replaceby):
    std_ = torch.std(eeg_signal).cpu()
    for each in ind_electrodes:
        shift = np.random.normal(0, std_*5)
        noise_volts  = np.random.normal(shift, std_, eeg_signal.shape[0])
        y_volts = noise_volts+replaceby
        eeg_signal[:, each] = torch.tensor(y_volts)

    # plt.figure(figsize=(8, 6))
    # for i in range(129):
    #     plt.plot(np.arange(0, eeg_signal.shape[0], 1), eeg_signal[:,i],color='black')
    # plt.ylim(-200, 3000)
    # plt.show()

    return eeg_signal



def manipulate_electrodes(X,k,threshold=300,replaceby=1000):
    """
    X: EEG signals in batch, shape (bs, timepoints, num_electrodes)
    k: number of electrodes being dropout
    """
    bs = X.shape[0]
    if k is None:
        indexes = [[0] for _ in range(bs)]
        indexes = torch.tensor(indexes)
        return X,indexes

    # print('replaceby={}'.format(replaceby))

    max_abs_value = torch.max(torch.abs(X),axis=1).values
    indexes = []
    for i in range(bs):
        above_threshold_elec = (max_abs_value[i] > threshold).nonzero().detach().cpu().numpy().flatten()
        if len(above_threshold_elec) < k:
            random_select = random.sample(set(np.arange(0,129,1)) -set(above_threshold_elec),k-len(above_threshold_elec))
            X[i] = replace_electrodes(X[i],random_select,replaceby=replaceby)
            indexes.append(list(random_select)+above_threshold_elec.tolist())
        else:
            indexes.append(list(above_threshold_elec[:k]))
    indexes = torch.tensor(indexes)
    return X,indexes

def try_models(trainX, trainY, ids, models, N=1, scoring=None, scale=False, save_trail='',
               isSaveModel=True,loss_func_pred_type='mse',multitask=False,data_boost=False,
               isClassification=False,saveScale=False, manipulateX=False, manipulate_parak=None,
               threshold = 500, blocks=None):

    splitMethod = 'acrossSubject' # 'acrossSubject'
    removeOutlier = 'no' # 'yes', 'no', 'replace'

    # threshold =500
    if splitMethod == 'acrossSubject':
        train, val, test = split(ids, 0.7, 0.15, 0.15, removeOutlier,threshold)
    elif splitMethod == 'blockbased':
        # based on block
        # block 0,1,2 as training, block 3 as validation, block 4 as testing (block 4 == block 0)
        ## both validation and testing show loss & score, no worries!
        train, val, test = split_basedblock(ids,blocks)
    elif splitMethod == 'withinSubject':

        if torch.cuda.is_available():
            dir = config['data_dir']
        else:
            dir = 'if local, then the local data dir'

        if removeOutlier == 'yes':
            split_npz_filepath = os.path.join(dir, 'split/split_ind_' + str(threshold) + '.npz')
        else:
            split_npz_filepath = os.path.join(dir, 'split/split_ind_ori_' + str(threshold) + '.npz')

        npz_data = np.load(split_npz_filepath)
        train,val= npz_data['train'],npz_data['val'] #,npz_data['test']

        split_npz_filepath = os.path.join(dir, 'split/split_ind_ori_' + str(threshold) + '.npz')
        npz_data = np.load(split_npz_filepath)
        test = npz_data['test'] # should not remove the data in test set

    logging.info(f"-----DATASET-SPLIT-----")
    logging.info(f"\tSplit Method       : {splitMethod}")
    logging.info(f"\tRemove Outliers    : {removeOutlier}")

    if isClassification:
        logging.info(f"-----TASK CONVERT TO CLASSIFICATION-----")
        logging.info(f"-----Converting y labels...-----")
        trainY = converting_ylabels(trainY)
        logging.info(f"-----FINISHED.-----")


    X_train, y_train = trainX[train], trainY[train]
    X_val, y_val = trainX[val], trainY[val]
    X_test, y_test = trainX[test], trainY[test]




    if removeOutlier == 'replace':
        # deal with outlier data
        dropout_num = 10
        replace = 0

        logging.info(f"-----DEAL WITH OUTLIER ELECTRODES-----")
        logging.info(f"\tDropout Number         : {dropout_num}")
        logging.info(f"\tReplace with           : {replace}")

        X_train,y_train = deal_wired_electrodes(X_train,y_train,dropout_num=dropout_num,replace=replace)
        X_val,y_val = deal_wired_electrodes(X_val, y_val,dropout_num=dropout_num,replace=replace)
        X_test,y_test = deal_wired_electrodes(X_test,y_test, dropout_num=None,replace=replace)
        logging.info('-----TRAIN:   X shape: {}\t y shape: {}.'.format(X_train.shape, y_train.shape))
        logging.info('-----VAL:     X shape: {}\t y shape: {}.'.format(X_val.shape, y_val.shape))
        logging.info('-----TEST:     X shape: {}\t y shape: {}.'.format(X_test.shape, y_test.shape))
        logging.info(f"-----FINISHED.-----")

    if data_boost:
        X_train, y_train = data_augmentation(X_train, y_train, sub_length=200, steps=100)
        X_val, y_val  = data_augmentation(X_val, y_val , sub_length=200, steps=100)
        X_test, y_test = data_augmentation(X_test, y_test, sub_length=200, steps=100)

        logging.info('-----DATA AUGMENTATION-----')
        logging.info('-----TRAIN:   X shape: {}\t y shape: {}.'.format(X_train.shape, y_train.shape))
        logging.info('-----VAL:     X shape: {}\t y shape: {}.'.format(X_val.shape, y_val.shape))
        logging.info('-----TEST:    X shape: {}\t y shape: {}.'.format(X_test.shape, y_test.shape))

    logging.info(f"-----TRAIN VAL TEST SET SIZE-----")
    logging.info(f"\tTrain Set          : {len(X_train)}")
    logging.info(f"\tVal Set            : {len(X_val)}")
    logging.info(f"\tTest Set           : {len(X_test)}")

    # hyperparameters
    batch_size = 32
    epochs = 50
    reduction ='mean'
    if isClassification:
        loss_func_pred_type = 'cross_entropy'
    # loss_func_pred_type = 'smoothl1' #'distanced_mse'
    loss_func_recon_type = 'smoothl1'#'weighted_l1' #weighted_mse
    loss_func_id_type = 'cross_entropy'

    loss_coeff_pred = 1
    loss_coeff_reconstruct = 200 # 200 if pred_type not smoothl1 or Normalization
    loss_coeff_id = 20

    loss_func_detect_type = 'bce'
    loss_coeff_detect = 0
    if manipulateX == False:
        loss_coeff_detect = 0

    if list(models.keys())[0] == 'GCNAdj':
        lr = 1e-3
    else:
        lr = 1e-3

    isNormalized = False
    isEval = True
    recon_scoring_func = 'euclidean_distance'
    isStoreTestRecon = False
    isSchedule = True

    isPretrained = False


    if recon_scoring_func == 'euclidean_distance':

        def recon_scoring(x,x_recon,scale):
            if scale is None:
                x = x.cpu().detach().numpy()
                x_recon = x_recon.cpu().detach().numpy()
                return np.linalg.norm(x - x_recon, axis=(1,2)).mean()
            else:
                x = x.cpu().detach().numpy()
                x_recon = x_recon.cpu().detach().numpy()
                scale = scale.cpu().detach().numpy()

                square_difference = np.square(x - x_recon)
                mean_each_channel = np.mean(square_difference, axis=1)
                # mean_each_channel = mean_each_channel.float()
                # out = mean_each_channel.mul(scale)
                out = np.multiply(mean_each_channel,scale)
                return out.mean()
        # recon_scoring = (lambda x, x_recon: np.linalg.norm(x - x_recon, axis=(1,2)).mean())
    elif recon_scoring_func == 'similarity':
        recon_scoring = (lambda x, x_recon: similarity_score(x,x_recon))
    else:
        raise Exception('Not implemented.')

    logging.info(f"-----HYPER-PARAMETERS-----")
    logging.info(f"\tBatch size     : {batch_size}")
    logging.info(f"\tEpochs         : {epochs}")
    logging.info(f"\tLearning Rate  : {lr}")
    logging.info(f"\tReduction Mode : {reduction}")
    logging.info(f"\tLoss Function for Prediction       : {loss_func_pred_type}")
    logging.info(f"\tLoss Function for Reconstruction   : {loss_func_recon_type}")
    logging.info(f"\tLoss Function for Id Classification    : {loss_func_id_type}")

    logging.info(f"\tCoefficient for Loss_pred          : {loss_coeff_pred}")
    logging.info(f"\tCoefficient for Loss_recon         : {loss_coeff_reconstruct}")
    logging.info(f"\tCoefficient for Loss_id            : {loss_coeff_id}")

    logging.info(f"\tInput Normalization                : {isNormalized}")
    logging.info(f"\tModel Eval                         : {isEval}")
    logging.info(f"\tScoring Function for Recon         : {recon_scoring_func}")
    logging.info(f"\tStore Test Reconstruction          : {isStoreTestRecon}")
    logging.info(f"\tSave Model                         : {isSaveModel}")
    logging.info(f"\tUse Scheduler                      : {isSchedule}")
    logging.info(f"\tUse Pretrained Model               : {isPretrained}")

    if isNormalized:
        X_train = data_normalization(X_train,data_name='X_train')
        X_val = data_normalization(X_val,data_name='X_val')
        X_test = data_normalization(X_test,data_name='X_test')


    # dataloader
    train_loader = create_dataloader(X_train, y_train, batch_size, model_name='', drop_last=False, shuffle=True,isClassification=isClassification)
    val_loader = create_dataloader(X_val, y_val, batch_size, model_name='', drop_last=False, shuffle=False,isClassification=isClassification)
    test_loader = create_dataloader(X_test, y_test, batch_size, model_name='', drop_last=False, shuffle=False,isClassification=isClassification)


    for model_name in models.keys():
        model = models[model_name][0](**models[model_name][1])
        logging.info(f"EXPERIMENT ON MODEL: {model_name}")

        if isPretrained:
            runs_dir = 'some dir'
            experiment_id = 'experiment id of pretrained model'
            pretrain_para_path = runs_dir+experiment_id+'\\checkpoint\\'+'{}_nb_0_.pth'.format(model_name)


            logging.info(f"Using pretrained model. Model load from: {pretrain_para_path}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.load_state_dict(torch.load(pretrain_para_path, map_location=device))
            logging.info(f"Load Pretrained Model Successfully.")

        if torch.cuda.is_available():
            model.cuda()

        # input_shape
        input_shape = model.input_shape

        if loss_func_pred_type == 'mse':
            loss_func_pred = torch.nn.MSELoss(reduction=reduction)
        elif loss_func_pred_type == 'smoothl1':
            loss_func_pred = torch.nn.SmoothL1Loss(reduction=reduction)
        elif loss_func_pred_type == 'angle_loss':
            loss_func_pred = AngleLoss(reduction=reduction)
        elif loss_func_pred_type == 'cosloss':
            loss_func_pred  = CosLoss(reduction=reduction)
        elif loss_func_pred_type == 'cross_entropy':
            loss_func_pred = torch.nn.CrossEntropyLoss()
        elif loss_func_pred_type == 'distanced_mse':
            loss_func_pred = DistancedMSELoss(reduction=reduction)
        elif loss_func_pred_type == 'wing':
            loss_func_pred = WingLoss(reduction = reduction)
        else:
            raise Exception('Not implemented.')

        if loss_func_recon_type == 'mse':
            loss_func_reconstruct = torch.nn.MSELoss(reduction=reduction)
        elif loss_func_recon_type == 'smoothl1':
            loss_func_reconstruct = torch.nn.SmoothL1Loss(reduction=reduction)
        elif loss_func_recon_type == 'wing':
            loss_func_reconstruct = WingLoss(reduction = reduction)
        elif loss_func_recon_type == 'l1':
            loss_func_reconstruct = torch.nn.L1Loss(reduction=reduction)
        elif loss_func_recon_type.startswith('weighted'):
            weights = np.ones(129)/129
            TOP4 = [1,32,125,128]
            TOP7 = TOP4+ [17,38,121]
            SIDE_FRONTS =TOP7+[2,3,8,9,14,21,22,23,25,26,27,33,43,120,122,123]
            w_ind = [each-1 for each in SIDE_FRONTS]
            weights[w_ind] = 10/129
            if loss_func_recon_type == 'weighted_mse':
                loss_func_reconstruct = WeightedMSELoss(weight=weights,reduction=reduction)
            elif loss_func_recon_type == 'weighted_l1':
                loss_func_reconstruct = WeightedL1Loss(weight=weights, reduction=reduction)
            else:
                raise Exception('Not implemented.')
        else:
            raise Exception('Not implemented.')

        if loss_func_id_type == 'cross_entropy':
            loss_func_id = torch.nn.CrossEntropyLoss()
        else:
            raise Exception('Not implemented.')


        if loss_func_detect_type == 'bce':
            loss_func_detect = torch.nn.BCELoss(reduction=reduction)
        else:
            raise Exception('Not implemented.')

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr)
                                    #weight_decay=1e-8)
        if isSchedule:
            scheduler = ReduceLROnPlateau(optimizer, 'min',patience =5)


        logging.info(f"-----START TRAINING-----")
        for epoch in range(epochs):
            logging.info("-------------------------------")
            logging.info(f"Epoch {epoch + 1}")

            # training
            train_size= len(train_loader.dataset)
            train_num_batches = len(train_loader)
            model.train()
            train_loss,train_loss_pred,train_loss_recon,train_loss_id = 0,0,0,0
            train_loss_detect = 0
            for batch_id,(X, y) in enumerate(train_loader):

                if torch.cuda.is_available():
                    X = X.cuda()
                    y = y.cuda()

                if len(y.size()) == 1:
                    y = y.view(-1, 1)

                if model_name == 'Autoencoder':

                    pred_y, reconstructedX, scale = model(X)
                    loss_pred_train = loss_func_pred(pred_y, y)
                    if loss_func_recon_type.startswith('weighted'):
                        loss_recon_train = loss_func_reconstruct(pred=reconstructedX, target=X,
                                                             scale=scale)
                    else:
                        loss_recon_train = loss_func_reconstruct(reconstructedX, X)
                    loss_train = loss_coeff_pred * loss_pred_train + loss_coeff_reconstruct * loss_recon_train
                    optimizer.zero_grad()
                    loss_train.backward()
                    optimizer.step()

                    train_loss += loss_train.item()
                    train_loss_pred += loss_pred_train.item()
                    train_loss_recon += loss_recon_train.item()

                elif model_name == 'CNNMultiTask' and multitask:
                    X = X  # (batch_size, timepoints, num_electrodes)
                    pred_y,pred_id = model(X)
                    loss_pred = loss_func_pred(pred_y, y[:,1:])
                    # id_onehot = one_hot(y[:,0],num_classes=72)
                    target = y[:,0].long()
                    target = torch.sub(target,1)
                    loss_id = loss_func_id(pred_id,target)
                    loss_train = loss_pred + loss_coeff_id*loss_id

                    optimizer.zero_grad()
                    loss_train.backward()
                    optimizer.step()

                    train_loss += loss_train.item()
                    train_loss_pred += loss_pred.item()
                    train_loss_id += loss_id.item()

                elif model_name == 'GCNAdj':
                    # X, indexes = manipulate_electrodes(X,k=manipulate_parak)
                    X = X # (batch_size, timepoints, num_electrodes)
                    # pred_y, sigmoid_scale = model.forward2(X,isPredict=False)

                    pred_y, scale = model.forward(X)
                    loss_train_pred = loss_func_pred(pred_y, y)
                    # bc_index = convert2binary(indexes,num_class=129)

                    # loss_train_detect = loss_func_detect(1-sigmoid_scale,bc_index)
                    loss_train = loss_train_pred #+ loss_coeff_detect*loss_train_detect

                    optimizer.zero_grad()
                    loss_train.backward()
                    optimizer.step()

                    train_loss += loss_train.item()
                    train_loss_pred += loss_train_pred.item()
                    # train_loss_detect += loss_train_detect.item()


                else:
                    X = X # (batch_size, timepoints, num_electrodes)
                    pred_y = model(X)
                    loss_train = loss_func_pred(pred_y, y)

                    optimizer.zero_grad()
                    loss_train.backward()
                    optimizer.step()

                    train_loss += loss_train.item()


                torch.cuda.empty_cache()

            if reduction == 'mean':
                loss_train_mean = train_loss / train_num_batches
                if model_name == 'Autoencoder':
                    train_loss_pred_mean = train_loss_pred / train_num_batches
                    train_loss_recon_mean = train_loss_recon / train_num_batches
                elif model_name == 'CNNMultiTask' and multitask:
                    train_loss_pred_mean = train_loss_pred / train_num_batches
                    train_loss_id_mean = train_loss_id / train_num_batches
                elif model_name == 'GCNAdj' and manipulateX:
                    train_loss_pred_mean = train_loss_pred / train_num_batches
                    train_loss_detect_mean = train_loss_detect / train_num_batches

            elif reduction == 'sum':
                loss_train_mean = train_loss / train_size
                if model_name == 'Autoencoder':
                    train_loss_pred_mean = train_loss_pred/train_size
                    train_loss_recon_mean = train_loss_recon/train_size
                elif model_name == 'CNNMultiTask' and multitask:
                    train_loss_pred_mean = train_loss_pred / train_size
                    train_loss_id_mean = train_loss_id / train_size
                elif model_name == 'GCNAdj' and manipulateX:
                    train_loss_pred_mean = train_loss_pred / train_size
                    train_loss_detect_mean = train_loss_detect / train_size

            logging.info(f"training: number of samples: {train_size}")
            logging.info(f"training: number of batches: {train_num_batches}")
            logging.info(f"Avg training loss: {loss_train_mean:>7f}")
            if model_name == 'Autoencoder':
                logging.info(f"Avg training loss - pred: {train_loss_pred_mean:>7f}")
                logging.info(f"Avg training loss - recon: {train_loss_recon_mean:>7f}")
            elif model_name == 'CNNMultiTask' and multitask:
                logging.info(f"Avg training loss - pred: {train_loss_pred_mean:>7f}")
                logging.info(f"Avg training loss - id: {train_loss_id_mean:>7f}")
            elif model_name == 'GCNAdj' and manipulateX:
                logging.info(f"Avg training loss - pred: {train_loss_pred_mean:>7f}")
                # logging.info(f"Avg training loss - detect: {train_loss_detect_mean:>7f}")

            torch.cuda.empty_cache()

            # validation
            if isEval:
                model.eval()
            val_size = len(val_loader.dataset)
            val_num_batches = len(val_loader)
            val_loss,val_loss_pred,val_loss_recon,val_loss_id = 0,0,0,0
            val_score_sum,val_score_recon_sum = 0,0
            pred_y_all, y_all = [],[]

            for batch_id,(X, y) in enumerate(val_loader):
                if torch.cuda.is_available():
                    X = X.cuda()
                    y = y.cuda()

                if len(y.size()) == 1:
                    y = y.view(-1, 1)

                if model_name == 'Autoencoder':
                    pred_y, reconstructedX, scale = model(X)
                    loss_pred_val = loss_func_pred(pred_y, y)
                    if loss_func_recon_type.startswith('weight'):
                        loss_recon_val = loss_func_reconstruct(pred=reconstructedX, target=X,
                                                                 scale=scale)
                    else:
                        loss_recon_val = loss_func_reconstruct(reconstructedX,X)
                    loss_val = loss_coeff_pred * loss_pred_val + loss_coeff_reconstruct * loss_recon_val

                    val_loss += loss_val.item()
                    val_loss_pred += loss_pred_val.item()
                    val_loss_recon += loss_recon_val.item()

                elif model_name == 'CNNMultiTask' and multitask:
                    X = X  # (batch_size, timepoints, num_electrodes)
                    pred_y,pred_id = model(X)
                    loss_pred = loss_func_pred(pred_y, y[:,1:])
                    # id_onehot = one_hot(y[:,0],num_classes=72)
                    # loss_id = loss_func_id(pred_id,id_onehot)
                    target = y[:, 0].long()
                    target = torch.sub(target, 1)
                    loss_id = loss_func_id(pred_id, target)
                    loss_val = loss_pred + loss_coeff_id*loss_id


                    val_loss += loss_val.item()
                    val_loss_pred += loss_pred.item()
                    val_loss_id += loss_id.item()

                elif model_name == 'GCNAdj':
                    # not need to manipulate for validation and test
                    X = X # (batch_size, timepoints, num_electrodes)
                    # pred_y, sigmoid_scale = model.forward2(X,isPredict=True)
                    pred_y, scale = model.forward(X)
                    loss_val = loss_func_pred(pred_y, y)
                    # bc_index = convert2binary(indexes,num_class=129)
                    # loss_train_detect = loss_func_detect(sigmoid_scale,bc_index)
                    # loss_train = loss_train_pred + loss_coeff_detect*loss_train_detect
                    val_loss += loss_val.item()
                    if batch_id == val_num_batches-1:
                        print('epoch {}, batch: {}'.format(epoch,batch_id))
                        print(scale.shape)
                        print(scale[:3,:10])



                else:
                    X = X # (batch_size,timepoints,num_electrodes)
                    pred_y = model(X)
                    loss_val = loss_func_pred(pred_y, y)
                    val_loss += loss_val.item()



                if torch.cuda.is_available():
                    pred_y = pred_y.detach().cpu().numpy()
                    y = y.detach().cpu().numpy()
                    # if model_name == 'Autoencoder':
                    #     X = X.detach().cpu().numpy()
                    #     reconstructedX = reconstructedX.detach().cpu().numpy()
                    #     scale = scale.detach().cpu().numpy()
                else:
                    pred_y = pred_y.detach().numpy()
                    y = y.detach().numpy()

                if multitask:
                    score = scoring(pred_y, y[:,1:])
                else:
                    score = scoring(pred_y, y)
                    if save_trail == '_angle' or save_trail =='_amplitude':
                        pred_y_all.append(pred_y)
                        y_all.append(y)

                val_score_sum += score * batch_size
                if model_name == 'Autoencoder':
                    score_recon = recon_scoring(x=X,x_recon=reconstructedX,scale=scale)
                    val_score_recon_sum += score_recon * batch_size

                # print('Val-batch:{}'.format(batch_id))
                # print('\tshape before mean: {}'.format(np.linalg.norm(X - reconstructedX, axis=(2,3)).shape))

                torch.cuda.empty_cache()
                # print('Training\tbatch:{}'.format(batch_id))

            if reduction == 'mean':
                loss_val_mean = val_loss / val_num_batches
                if model_name == 'Autoencoder':
                    val_loss_pred_mean = val_loss_pred / val_num_batches
                    val_loss_recon_mean = val_loss_recon / val_num_batches
                elif model_name == 'CNNMultiTask' and multitask:
                    val_loss_pred_mean = val_loss_pred / val_num_batches
                    val_loss_id_mean = val_loss_id / val_num_batches

            elif reduction == 'sum':
                loss_val_mean = val_loss / val_size
                if model_name == 'Autoencoder':
                    val_loss_pred_mean = val_loss_pred / val_size
                    val_loss_recon_mean = val_loss_recon / val_size
                elif model_name == 'CNNMultiTask' and multitask:
                    val_loss_pred_mean = val_loss_pred / val_size
                    val_loss_id_mean = val_loss_id / val_size

            val_score_mean = val_score_sum/val_size

            if save_trail == '_angle' or save_trail =='_amplitude':
                y_all = (np.vstack(y_all)).flatten().reshape(-1, 1)
                pred_y_all = (np.vstack(pred_y_all)).flatten().reshape(-1, 1)
                score_for_angle = scoring(y_all, pred_y_all)
                val_score_mean = score_for_angle

            if model_name == 'Autoencoder':
                val_score_recon_mean = val_score_recon_sum/val_size

            if isSaveModel:
                if epoch == 0:
                    best_val_loss = loss_val_mean
                    model.save()
                else:
                    if loss_val_mean < best_val_loss:
                        best_val_loss = loss_val_mean
                        model.save()

            logging.info(f"Validation: number of samples: {val_size}")
            logging.info(f"Validation: number of batches: {val_num_batches}")
            logging.info(f"Avg validation loss: {loss_val_mean:>7f}")
            if model_name == 'Autoencoder':
                logging.info(f"Avg validation loss - pred: {val_loss_pred_mean:>7f}")
                logging.info(f"Avg validation loss - recon: {val_loss_recon_mean:>7f}")
            elif model_name == 'CNNMultiTask' and multitask:
                logging.info(f"Avg validation loss - pred: {val_loss_pred_mean:>7f}")
                logging.info(f"Avg validation loss - id: {val_loss_id_mean:>7f}")
            logging.info(f"Avg score: {val_score_mean:>7f}")
            if model_name == 'Autoencoder':
                logging.info(f"Avg score for reconstruction: {val_score_recon_mean:>7f}")

            if isSchedule:
                scheduler.step(loss_val_mean)




        logging.info(f"-----FINISH TRAINING-----")


        # test
        logging.info(f"-----START TESTING-----")

        torch.cuda.empty_cache()

        # load the best model in validation
        load_model_path = model.path + model_name + \
                   '_nb_{}_{}'.format(0,model.saveModel_suffix) + '.pth'
        logging.info(f"-----Load model from: {load_model_path}")
        model = models[model_name][0](**models[model_name][1])
        # print(path + file)
        model.load_state_dict(torch.load(load_model_path))
        if torch.cuda.is_available():
            model.cuda()
        logging.info(f"-----Load finish.")



        if isEval:
            model.eval()
        test_size = len(test_loader.dataset)
        test_loss,test_loss_pred,test_loss_recon,test_loss_id=0,0,0,0
        test_score_sum,test_score_recon_sum=0,0
        pred_y_all = []
        y_all = []

        if isStoreTestRecon:
            reconX_store = []

        if saveScale:
            testScale = []

        for batch, (X, y) in enumerate(test_loader):
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()

            if len(y.size()) == 1:
                y = y.view(-1, 1)

            if model_name == 'Autoencoder':
                pred_y, reconstructedX, scale = model(X)
                loss_pred_test = loss_func_pred(pred_y, y)
                if loss_func_recon_type.startswith('weight'):
                    loss_recon_test = loss_func_reconstruct(pred=reconstructedX, target=X,
                                                           scale=scale)
                else:
                    loss_recon_test = loss_func_reconstruct(reconstructedX, X)

                loss_test = loss_coeff_pred * loss_pred_test + loss_coeff_reconstruct * loss_recon_test

            elif model_name == 'CNNMultiTask' and multitask:
                X = X  # (batch_size, timepoints, num_electrodes)
                pred_y, pred_id = model(X)
                loss_pred = loss_func_pred(pred_y, y[:, 1:])
                # id_onehot = one_hot(y[:, 0], num_classes=72)
                # loss_id = loss_func_id(pred_id, id_onehot)
                target = y[:, 0].long()
                target = torch.sub(target, 1)
                loss_id = loss_func_id(pred_id, target)
                loss_test = loss_pred + loss_coeff_id * loss_id

            elif model_name == 'GCNAdj':
                # not need to manipulate for validation and test
                X = X  # (batch_size, timepoints, num_electrodes)
                with torch.no_grad():
                    # pred_y, scale = model.forward2(X,isPredict=True)
                    pred_y, scale = model.forward(X)
                loss_test = loss_func_pred(pred_y, y)



            else:
                X = X  # (batch_size,timepoints, num_electrodes, )
                with torch.no_grad():
                    pred_y,scale = model.predict(X)
                loss_test = loss_func_pred(pred_y, y)




            current_batch_size = len(y)
            # print('current batch size: {}'.format(current_batch_size))
            if reduction == 'sum':
                test_loss += loss_test.item()
                if model_name == 'Autoencoder':
                    test_loss_pred += loss_pred_test.item()
                    test_loss_recon += loss_recon_test.item()
                elif model_name == 'CNNMultiTask' and multitask:
                    test_loss_pred += loss_pred.item()
                    test_loss_id += loss_id.item()

            elif reduction == 'mean':
                test_loss += (loss_test.item()*current_batch_size)
                if model_name == 'Autoencoder':
                    test_loss_pred += (loss_pred_test.item()*current_batch_size)
                    test_loss_recon += (loss_recon_test.item()*current_batch_size)
                elif model_name == 'CNNMultiTask' and multitask:
                    test_loss_pred += (loss_pred.item()*current_batch_size)
                    test_loss_id += (loss_id.item()*current_batch_size)


            # to host memory in order to convert to numpy
            if torch.cuda.is_available():
                pred_y = pred_y.detach().cpu().numpy()
                y = y.detach().cpu().numpy()
            else:
                pred_y = pred_y.detach().numpy()
                y = y.detach().numpy()
            if multitask:
                score = scoring(pred_y, y[:,1:])
            else:
                score = scoring(pred_y, y)
                # for test
                # print(f'----{batch}----')
                # print('y:{}'.format(y))
                # print('pred_y:{}'.format(pred_y))
                # print(f'score: {score}')
                # for angle, the scoring is quite different
                if save_trail == '_angle' or save_trail =='_amplitude':
                    print('rmse')
                    pred_y_all.append(pred_y)
                    y_all.append(y)
            if model_name == 'Autoencoder':
                score_recon = recon_scoring(x=X,x_recon=reconstructedX,scale=scale)
            current_batch_size = len(y)

            if isStoreTestRecon:
                reconX_store.append(reconstructedX)

            if saveScale:
                scale = scale.detach().cpu().numpy()
                testScale.append(scale)


            # print('{}-current batch size:{}'.format(batch,current_batch_size))
            # print('\tcurrent score: {:.4f}'.format(score))
            # print('\ty shape: {}'.format(y.shape))
            # print('\ty len: {}'.format(len(y)))
            # print('\ty: {}'.format(y))
            # print('\tpred y: {}'.format(pred_y))
            # print('\t{}'.format(np.linalg.norm(y - pred_y, axis=1)))
            test_score_sum += (score*current_batch_size)
            if model_name == 'Autoencoder':
                test_score_recon_sum += (score_recon*current_batch_size)

            torch.cuda.empty_cache()

        loss_test_mean = test_loss/test_size
        score_mean = test_score_sum / test_size

        if save_trail == '_angle' or save_trail =='_amplitude':
            y_all = (np.vstack(y_all)).flatten().reshape(-1, 1)
            pred_y_all = (np.vstack(pred_y_all)).flatten().reshape(-1, 1)
            score_for_angle = scoring(y_all,pred_y_all)
            score_mean = score_for_angle

        if model_name == 'Autoencoder':
            test_loss_pred_mean = test_loss_pred / test_size
            test_loss_recon_mean = test_loss_recon / test_size
            score_recon_mean = test_score_recon_sum / test_size
        elif model_name == 'CNNMultiTask' and multitask:
            test_loss_pred_mean = test_loss_pred / test_size
            test_loss_id_mean = test_loss_id / test_size



        logging.info(f"Test: number of test sample: {test_size}")
        logging.info(f"Avg test loss: {loss_test_mean:>7f}")
        if model_name == 'Autoencoder':
            logging.info(f"Avg test loss - pred: {test_loss_pred_mean:>7f}")
            logging.info(f"Avg test loss - recon: {test_loss_recon_mean:>7f}")
        elif model_name == 'CNNMultiTask' and multitask:
            logging.info(f"Avg test loss - pred: {test_loss_pred_mean:>7f}")
            logging.info(f"Avg test loss - id: {test_loss_id_mean:>7f}")
        logging.info(f"Avg test score: {score_mean:>7f}")
        if model_name == 'Autoencoder':
            logging.info(f"Avg test score for reconstruction: {score_recon_mean:>7f}")
        logging.info(f"-----FINISH TESTING-----")


        if isStoreTestRecon:
            logging.info(f"-----STORE RECONSTRUCTION-----")
            reconX_store = np.vstack(reconX_store)
            np.savez(config['model_dir'] + '/reconX.npz', EEG = reconX_store,labels=y_test)
            logging.info(f"-----STORE RECONSTRUCTION FINISHED-----")

        if saveScale and testScale is not None:
            logging.info(f"-----STORE TEST SCALE-----")
            testScale = np.vstack(testScale)
            np.savez(config['model_dir'] + '/test_scale.npz', scale=testScale)
            logging.info(f"-----STORE TEST SCALE FINISHED-----")

        if model_name == 'GCNAdj':
            model.save_adj_matrix()
















def experiment(trainX, trainY):

    np.savetxt(config['model_dir']+'/config.csv', [config['task'], config['dataset'], config['preprocessing']], fmt='%s')
    path = config['checkpoint_dir']


    n_rep = 2
    data_boost = False
    threshold = 500

    nb_features = 64
    kernel_size = (64,64)

    input_shape = (trainX.shape[2], trainX.shape[1])


    isClassification = False
    if isClassification:
        output_shape = 25
    else:
        if config['task'].startswith('Direction'):
            output_shape = 1
        elif config['task'].startswith('Position'):
            output_shape = 2
        elif config['task'].startswith('Advanced_Direction'):
            if config['subtask'] == 'ANG' or config['subtask'] == 'AMP':output_shape=1
            else:output_shape = 2

    multitask = False

    use_residual = True

    # use SEB and/or SAB
    use_SEB = True
    use_self_attention = True

    # save scales
    if use_SEB or use_self_attention:
        saveScale = True
    else:
        saveScale = False

    # for GCN
    manipulateX = True
    using_rank = True
    if manipulateX:
        manipulate_parak = 5
    else:
        manipulate_parak = None
    GCN_init = True
    GCN_init_domain = True
    domain_mode = '2d'

    models = {
        'CNNMultiTask':[CNNMultiTask,{'input_shape':input_shape,'output_shape':output_shape,'depth':12,
                                      'mode':'1DT','path':path,'multitask':multitask,
                                      'use_SEB':use_SEB,'use_self_attention':use_self_attention,
                                      'nb_features':nb_features, 'kernel_size':kernel_size}],
        # 'GCNAdj':[GCNAdj,{'input_shape':input_shape,'output_shape':output_shape,'path':path,
        #                   'manipulate_parak':manipulate_parak,'threshold':threshold,'init':GCN_init,
        #                   'init_domain':GCN_init_domain,'domain_mode':domain_mode,'using_rank':using_rank}]
    }

    models_direction = {
        'amplitude':{
            'CNNMultiTask': [CNNMultiTask,
                             {'input_shape': input_shape, 'output_shape': output_shape, 'depth': 12, 'mode': '1DT',
                              'path': path, 'multitask': multitask,
                              'use_SEB': use_SEB, 'use_self_attention': use_self_attention,
                              'use_residual':use_residual,
                              'nb_features':nb_features, 'kernel_size':kernel_size}],
            # 'GCNAdj': [GCNAdj, {'input_shape': input_shape, 'output_shape': output_shape, 'path': path,
            #                     'manipulate_parak': manipulate_parak, 'threshold': threshold,
            #                     'init':GCN_init,'init_domain':GCN_init_domain,
            #                     'domain_mode':domain_mode,'using_rank':using_rank}]
        },
        'angle':{
            'CNNMultiTask': [CNNMultiTask,
                             {'input_shape': input_shape, 'output_shape': output_shape, 'depth': 12, 'mode': '1DT',
                              'path': path, 'multitask': multitask,
                              'use_SEB': use_SEB, 'use_self_attention': use_self_attention,
                              'use_residual':use_residual,
                              'nb_features':nb_features, 'kernel_size':kernel_size}],
            # 'GCNAdj': [GCNAdj, {'input_shape': input_shape, 'output_shape': output_shape, 'path': path,
            #                     'manipulate_parak': manipulate_parak, 'threshold': threshold,
            #                     'init':GCN_init,'init_domain':GCN_init_domain,
            #                     'domain_mode':domain_mode,'using_rank':using_rank}]

        }

    }

    ids = trainY[:, 0]

    if config['task'] == 'LR_task':
        if config['dataset'] == 'antisaccade':
            scoring = (lambda y, y_pred: accuracy_score(y, y_pred.ravel()))  # Subject to change to mean euclidean distance.
            y = trainY[:,1] # The first column are the Id-s, we take the second which are labels
            try_models(trainX=trainX, trainY=y, ids=ids, models=models, scoring=scoring)
        else:
            raise ValueError("This task cannot be predicted (is not implemented yet) with the given dataset.")

    elif config['task'] == 'Direction_task':
        if config['dataset'] == 'dots':
            logging.info('----------AMPLITUDE----------')
            scoring = (lambda y, y_pred : np.sqrt(mean_squared_error(y, y_pred.ravel())))
            y1 = trainY[:,1] # The first column are the Id-s, we take the second which are amplitude labels
            try_models(trainX=trainX, trainY=y1, ids=ids, models=models_direction['amplitude'], scoring=scoring,
                       multitask=multitask, data_boost=data_boost,
                       isClassification=isClassification, saveScale=saveScale, manipulateX=manipulateX,
                       manipulate_parak=manipulate_parak, threshold=threshold,
                       save_trail='_amplitude',loss_func_pred_type='smoothl1')
            logging.info('----------ANGLE----------')
            scoring2 = (lambda y, y_pred: np.sqrt(np.mean(np.square(np.arctan2(np.sin(y - y_pred.ravel()), np.cos(y - y_pred.ravel()))))))
            scoring2 = (lambda y, y_pred: np.sqrt(
                np.mean(np.square(np.arctan2(np.sin(y - y_pred), np.cos(y - y_pred))))))

            y2 = trainY[:,2] # The first column are the Id-s, second are the amplitude labels, we take the third which are the angle labels
            try_models(trainX=trainX, trainY=y2, ids=ids, models=models_direction['angle'], scoring=scoring2,
                       multitask=multitask, data_boost=data_boost,
                       isClassification=isClassification, saveScale=saveScale, manipulateX=manipulateX,
                       manipulate_parak=manipulate_parak, threshold=threshold,
                       save_trail='_angle',loss_func_pred_type='angle_loss')

            # logging.info('----------SHIFT----------')
            # scoring2 = (lambda y, y_pred: np.linalg.norm(y - y_pred, axis=1).mean())  # Euclidean distance
            #
            # y_row = trainY[:,1:]
            # y = copy.deepcopy(y_row)
            # y[:,0] = np.cos(y_row[:,1])*y_row[:,0]
            # y[:,1] = np.sin(y_row[:,1])*y_row[:,0]
            #
            # try_models(trainX=trainX, trainY=y, ids=ids, models=models, scoring=scoring2, multitask=multitask,
            #                 data_boost=data_boost,
            #                 isClassification=isClassification, saveScale=saveScale, manipulateX=manipulateX,
            #                 manipulate_parak=manipulate_parak, threshold=threshold,
            #                 loss_func_pred_type='smoothl1')
        else:
            raise ValueError("This task cannot be predicted (is not implemented yet) with the given dataset.")

    elif config['task'] == 'Position_task':
        if config['dataset'] == 'dots':
            scoring = (lambda y, y_pred : np.sqrt(mean_squared_error(y, y_pred))) # Subject to change to mean euclidean distance.
            scoring2 = (lambda y, y_pred: np.linalg.norm(y - y_pred, axis=1).mean()) # Euclidean distance
            if multitask:
                y = trainY
            else:
                y = trainY[:,1:] # The first column are the Id-s, the second and third are position x and y which we use
            if isClassification:
                scoring2= (lambda y_pred,y: accuracy_score(y, np.argmax(y_pred,axis=1)))

            try_models(trainX=trainX, trainY=y, ids=ids, models=models, scoring=scoring2,multitask=multitask,data_boost=data_boost,
                       isClassification=isClassification,saveScale=saveScale,manipulateX=manipulateX,
                       manipulate_parak = manipulate_parak, threshold=threshold,
                       loss_func_pred_type='smoothl1')
        else:
            raise ValueError("This task cannot be predicted (is not implemented yet) with the given dataset.")

    elif config['task'] == 'Segmentation_task':
        if config['dataset'] == 'dots':
            scoring = (lambda y, y_pred : f1_score(np.concatenate(y), np.concatenate(np.argmax(np.reshape(np.concatenate(y_pred), (-1,3,500)), axis=1)), average='macro')) # Macro average f1 as segmentation metric
            y = trainY[:,1:] # The first column are the Id-s, the rest are the labels of the events (0=F, 1=S, 2=B)
            try_models(trainX=trainX, trainY=y, ids=ids, models=models, scoring=scoring)
        else:
            raise ValueError("This task cannot be predicted (is not implemented yet) with the given dataset.")

    elif config['task'] == 'Advanced_Direction_task':
        if config['dataset'] == 'dots':
            ids = trainY[:, 0].astype(np.float32)
            ids = trainY[:, -2]  #.astype(np.float32)
            scoring = (lambda y, y_pred : np.sqrt(mean_squared_error(y, y_pred))) # Subject to change to mean euclidean distance.
            scoring2 = (lambda y, y_pred: np.linalg.norm(y - y_pred, axis=1).mean()) # Euclidean distance
            fixation_1 = trainY[:, 5:7].astype(np.float32)  # 5:7
            fixation_2 = trainY[:, 7:9].astype(np.float32)  # 7:9
            amp = trainY[:, 9].astype(np.float32)
            ang = trainY[:, 10].astype(np.float32)
            blocks = trainY[:,12].astype(np.float32)
            mode = config['subtask']
            logging.info(f"Advanced Direction Task, mode: {mode}")
            if mode == 'START':
                # starting point
                y = fixation_1 # - fixation_1
                loss_func_pred_type = 'smoothl1'
            elif mode == 'END':
                y = fixation_2
                loss_func_pred_type = 'smoothl1'
            elif mode == 'SHIFT':
                y = fixation_2  - fixation_1
                loss_func_pred_type = 'smoothl1'
            elif mode == 'ANG':
                y = ang
                loss_func_pred_type = 'angle_loss'
                scoring2 = (lambda y, y_pred: np.sqrt(
                    np.mean(np.square(np.arctan2(np.sin(y - y_pred), np.cos(y - y_pred))))))
            elif mode == 'AMP':
                y = amp
                loss_func_pred_type = 'smoothl1'
                scoring2 = (lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred.ravel())))

            try_models(trainX=trainX, trainY=y, ids=ids, models=models, scoring=scoring2,multitask=multitask,data_boost=data_boost,
                       isClassification=isClassification,saveScale=saveScale,manipulateX=manipulateX,
                       manipulate_parak = manipulate_parak, threshold=threshold,
                       loss_func_pred_type=loss_func_pred_type,blocks=blocks)
        else:
            raise ValueError("This task cannot be predicted (is not implemented yet) with the given dataset.")


    else:
        raise NotImplementedError(f"Task {config['task']} is not implemented yet.")