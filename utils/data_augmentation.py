import numpy as np

def get_parts(total_length,sub_length,steps):
    parts = []
    for i in range(int(total_length-sub_length)//steps + 1):
        parts.append(np.arange(0+i*steps,0+i*steps+sub_length))
    return parts



def data_augmentation(X,y,sub_length=200,steps=50):
    # X shape: (num_trails, timepoints, num_electrodes)
    new_X, new_y = [],[]
    parts = get_parts(total_length=500,sub_length=sub_length,steps=steps)
    for each_part in parts:
        new_X.append(X[:,each_part,:])
        new_y.append(y)

    new_X = np.vstack(new_X)
    new_y = np.vstack(new_y)
    return new_X,new_y



if __name__ == '__main__':
    X = np.random.random([5,500,3])
    y = np.random.random([5,2])
    new_X, new_y = data_augmentation(X,y)
