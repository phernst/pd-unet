import h5py
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


def main():
    train_path = r"sparse8_train.h5"
    val_path = r"sparse8_valid.h5"

    data = []

    dataset = h5py.File(train_path, 'r', libver='latest', swmr=True)
    for idx in tqdm(range(len(dataset))):
        h5group = dataset[list(dataset.keys())[idx]]
        fulldata = h5group['sinoMag'][()]
        data.append(fulldata)

    dataset = h5py.File(val_path, 'r', libver='latest', swmr=True)
    for idx in tqdm(range(len(dataset))):
        h5group = dataset[list(dataset.keys())[idx]]
        fulldata = h5group['sinoMag'][()]
        data.append(fulldata)

    print(len(data))
    data = np.array(data)
    print(data.shape)
    per95 = np.percentile(data, 99)
    print(per95)
    plt.hist(data.flatten(), 100)
    plt.show()


if __name__ == '__main__':
    main()
