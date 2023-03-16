import numpy as np
import random
import os
from data_analysis.withinsubject_analysis import load_data

def get_wired_electrodes(trail,threshold=150):
    # trail should have the shape of (500,129)
    assert trail.shape[0] == 500
    assert trail.shape[1] == 129

    max_ = np.max(np.abs(trail),axis=0)
    # tmp = np.where(max_>threshold)
    return np.where(max_>threshold)[0]


def deal_wired_electrodes(X,y,dropout_num=None,replace=0,record=False):
    random_seed = 2022
    record_data = []
    drop_index =[]
    for i in range(X.shape[0]):
        each_trail = X[i]

        # get wired electrodes:
        wired_electrode_indexes = get_wired_electrodes(each_trail)
        if record:
            record_data.append(wired_electrode_indexes)
        each_trail[:, wired_electrode_indexes] = replace

        if dropout_num != None:
            if len(wired_electrode_indexes) > 2*dropout_num:
                drop_index.append(i)
        else:
            if len(wired_electrode_indexes) > 20:
                drop_index.append(i)

        if dropout_num != None and len(wired_electrode_indexes) < dropout_num:
            left = dropout_num - len(wired_electrode_indexes)
            random.seed(random_seed)
            rest_indexes = set(range(129)) - set(wired_electrode_indexes)
            random_indexes = random.sample(rest_indexes, left)
            each_trail[:, random_indexes] = replace


        X[i] = each_trail

    if record:
        np.savez('./outlier_electrodes.npz',record=record_data)

    remain_indexes = np.array(list(set(range(X.shape[0])) - set(drop_index)))
    return X[remain_indexes],y[remain_indexes]

if __name__ == '__main__':
    data_folder = 'D:\\ninavv\\master\\thesis\\data\\'
    file_name = 'Position_task_with_dots_synchronised_min.npz'
    file_path = os.path.join(data_folder, file_name)

    X, _ = load_data(file_path=file_path)
    deal_wired_electrodes(X, dropout_num=None, replace=0, record=True)
    print('DONE')
