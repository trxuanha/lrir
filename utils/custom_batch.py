# https://pytorch.org/docs/stable/data.html
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random


class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.x = torch.stack(transposed_data[0], 0)
        self.y = torch.stack(transposed_data[1], 0)
        self.e = torch.stack(transposed_data[2], 0)
        self.t = torch.stack(transposed_data[3], 0)
        self.c = torch.stack(transposed_data[4], 0)
        self.nc = torch.stack(transposed_data[4], 0)

    def pin_memory(self):
        self.x = self.x.pin_memory()
        self.y = self.y.pin_memory()
        self.e = self.e.pin_memory()
        self.t = self.a.pin_memory()
        self.c = self.a.pin_memory()
        self.nc = self.a.pin_memory()
        return self


def collate_wrapper(batch):
    return SimpleCustomBatch(batch)


def build_iterator(args, train_data, test_data, valid_data):
    train_iterator = DataLoader(
        convert_tensor(x=train_data['x'], y=train_data['y'], e=train_data['e'], t=train_data['t'], c=train_data['c'], nc=train_data['nc']),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_wrapper)

    '''
    valid_iterator = DataLoader(
        convert_tensor(x=valid_data['x'], y=valid_data['y'], e=valid_data['e'], t=valid_data['t']),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_wrapper)

    test_iterator = DataLoader(
        convert_tensor(x=test_data['x'], y=test_data['y'], e=test_data['e'], t=test_data['t']),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_wrapper)
    '''

    return {"train_iterator": train_iterator, "valid_iterator": None, "test_iterator": None}


def convert_tensor(x, y, e, t, c, nc):
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y),
                         torch.from_numpy(e), torch.from_numpy(t), torch.from_numpy(c), torch.from_numpy(nc))


def worker_init_fn(args):
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return
