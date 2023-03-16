
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    print(f"-----Load data from: {file_path}")
    np_data = np.load(file_path, allow_pickle=True)
    np_eeg = np_data['EEG']
    # np_eeg = None
    np_labels = np_data['labels']
    print(f"-----Load finish.")
    return np_eeg,np_labels

def plot(record):
    record = np.array(record)
    x_pos = np.arange(len(record))
    xlabels = ['S{}'.format(int(each)) for each in record[:,0]]

    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(x_pos, record[:,1], yerr=record[:,2], align='center', alpha=0.5, ecolor='black', capsize=3)
    ax.set_ylabel('EEG signal')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xlabels,rotation=90)
    ax.set_title('Mean and std value for EEG signal per subject')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    # plt.savefig('bar_plot_with_error_bars.png')
    plt.show()


if __name__ == '__main__':
    # # load in data
    # data_folder = 'D:\\ninavv\\master\\thesis\\data\\'
    # file_name = 'Position_task_with_dots_synchronised_min.npz'
    # file_path = os.path.join(data_folder, file_name)
    #
    # X_test, y_test = load_data(file_path=file_path)
    #
    # # group according to id
    # task_type = file_name.split('_')[0]
    # if task_type == 'Position':
    #     column_names = ['id','x','y']
    # elif task_type == 'Direction':
    #     column_names = ['id','amplitude','angle']
    # else:
    #     raise Exception('not implemented')
    #
    # df = pd.DataFrame(y_test, columns=column_names)
    # df_gb = df.groupby(by=['id'])
    #
    # print('----- Label Summary -----')
    #
    # print(df_gb.mean())
    # print(df_gb.std())
    #
    # group_indexes_dict = df_gb.indices
    # keys = group_indexes_dict.keys()
    # record = []
    # for each_key in keys:
    #     eeg_signal = X_test[group_indexes_dict[each_key]]
    #     mean_ = eeg_signal.mean()
    #     std_ = eeg_signal.std()
    #     print('Subject {}\t mean:{:.4f}, std:{:.4f}'.format(each_key,mean_,std_))
    #     record.append([each_key,mean_,std_])
    #
    # record = np.array(record)
    # np.savez('withinsubject_analysis.npz',record=record)


    record = np.load('withinsubject_analysis.npz')['record']

    plot(record)
    print(record[record[:,2] >50][:,0])
