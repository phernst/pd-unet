import h5py
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import json

with open('config.json', 'r') as json_file:
    json_data = json.load(json_file)
    train_path: str = json_data["train_path"]
    val_path: str = json_data["val_path"]

data = []

dataset = h5py.File(train_path, 'r', libver='latest', swmr=True)
for idx in tqdm(range(len(dataset))):
    h5group = dataset[list(dataset.keys())[idx]]
    # fulldata = h5group['sinoMag'][()]
    fulldata = h5group['imgMag'][()]
    data.append(fulldata)

dataset = h5py.File(val_path, 'r', libver='latest', swmr=True)
for idx in tqdm(range(len(dataset))):
    h5group = dataset[list(dataset.keys())[idx]]
    # fulldata = h5group['sinoMag'][()]
    fulldata = h5group['imgMag'][()]
    data.append(fulldata)

print(len(data))
data = np.array(data)
print(data.shape)
per95 = np.percentile(data, 99)
print(per95)
plt.hist(data.flatten(), 100); plt.show()
