import pickle
import numpy as np
from torch.utils.data import Dataset
import torch
import h5py

class YSDataset(Dataset):
    def __init__(self, path='data2048.pklh5', t='train'):
        """Init function."""
        self.data_file_path = path

        with h5py.File(self.data_file_path, "r") as f:
            wf_key = "waveforms/0"
            self.cols_len = len(f[wf_key][()].tolist())

        with h5py.File(self.data_file_path, "r") as f:
            self.index_list = f["waveforms/index"][()].tolist()

        idx = int(len(self.index_list) * 0.8)

        if t == 'train':
            self.index_list = self.index_list[:idx]
        else:
            self.index_list = self.index_list[idx:]

    def __getitem__(self, index):
        """Get item."""
        index_ = self.index_list[index]
        index_r = index_ // self.cols_len
        index_c = index_ % self.cols_len
        with h5py.File(self.data_file_path, "r") as f:
            wf_key = "waveforms/{}".format(index_r)
            d_x = f[wf_key][()].tolist()[index_c]

        return torch.from_numpy(np.array(d_x, dtype=np.float32)/16384).reshape(1,-1)  # 10位

    def __len__(self):
        """Length."""
        return len(self.index_list)


if __name__ == '__main__':

    data_set = YSDataset('./数据制作/data2048.pklh5', t='train')
    DataLoader = torch.utils.data.DataLoader(data_set, batch_size=32, shuffle=True, drop_last=True)

    print(len(DataLoader))

    for dx in DataLoader:
        print('-------------------')
        print(dx.shape)
        print(dx)
