import json

import numpy as np
import pandas as pd
import torch
from toposort import toposort
from torch.utils.data import DataLoader, Dataset


class NASDataset(Dataset):
    def __init__(self, file_path, split='train'):
        super(NASDataset, self).__init__()
        self.file_path = file_path

        # load the file
        df = pd.read_csv(
            self.file_path,
            header=None,
            delimiter=']],',
            engine='python')

        train_samples = df.sample(frac=0.8, random_state=42)
        val_test_samples = df.drop(train_samples.index)
        val_samples = val_test_samples.sample(frac=0.5, random_state=42)
        test_samples = val_test_samples.drop(val_samples.index)

        if split == 'train':
            self.samples = train_samples
        elif split == 'val':
            self.samples = val_samples
        elif split == 'test':
            self.samples = test_samples
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    @classmethod
    def _add_delim(cls, x):
        print(type(x), x)
        correct_json = x + ']]'
        return json.loads(correct_json)

    @classmethod
    def _process_seq(cls, x):
        node_types = 6 + 1  # 1 for EOS token
        seq_len = len(x) + 1  # 1 for EOS token

        node_encoding = np.zeros(seq_len, node_types)
        dep_dict = {}
        for index, node in enumerate(x):
            node_type = node[0]
            node_encoding[index, node_type] = 1
            for dep, val in enumerate(node[1:]):
                if val == 0:
                    continue
                if dep not in dep_dict:
                    dep_dict[dep] = set()
                dep_dict[dep].add(index)
        node_encoding[-1, -1] = 1
        node_order = [y for x in toposort(dep_dict) for y in sorted(list(x))]
        node_order.append(seq_len - 1)  # add EOS node at the end

        return node_encoding, node_order

    def __getitem__(self, index):
        item = self.samples.iloc[index]

        seq = NASDataset._add_delim(item[0])
        node_encoding, node_order = NASDataset._process_seq(seq)
        acc = float(item[1])

        return {
            'graph': seq,
            'node_encoding': node_encoding,
            'node_order': node_order,
            'acc': acc
        }


def get_dataloaders(batch_size, file_path):
    train_dataset = NASDataset(file_path, split='train')
    val_dataset = NASDataset(file_path, split='val')
    test_dataset = NASDataset(file_path, split='test')
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        pin_memory=True, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        pin_memory=True, shuffle=False),
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        pin_memory=True, shuffle=False)
    return train_loader, val_loader, test_loader
