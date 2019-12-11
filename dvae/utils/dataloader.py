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
        correct_json = x + ']]'
        return json.loads(correct_json)

    @classmethod
    def _process_seq(cls, x):
        node_types = 6 + 2  # 1 for EOS token, 1 for SOS token
        seq_len = len(x) + 2  # 1 for EOS token, 1 for SOS token

        # store 1-hot encoding for nodes in seq
        node_encoding = np.zeros((seq_len, node_types), dtype=int)
        node_encoding[0, node_types - 2] = 1
        node_encoding[-1, node_types - 1] = 1
        dep_graph = np.zeros((seq_len, seq_len), dtype=int)

        no_child_nodes = set(range(seq_len - 2))
        for seq_idx, node_info in enumerate(x):
            node_type = node_info[0]
            shifted_idx = seq_idx + 1
            # add 1 for SOS token
            node_encoding[shifted_idx, node_type] = 1

            if seq_idx > 0:
                parents = np.array(node_info[1:])
                dep_graph[shifted_idx, 1: len(parents) + 1] = parents
                if parents.sum() == 0:
                    dep_graph[shifted_idx, 0] = 1
                else:
                    parent_idx = set(np.nonzero(parents)[0])
                    no_child_nodes = no_child_nodes - parent_idx
            # always connect last node in ENAS
            dep_graph[shifted_idx, seq_idx] = 1
        for seq_idx in no_child_nodes:
            dep_graph[-1, seq_idx + 1] = 1

        return torch.from_numpy(dep_graph).float(), torch.from_numpy(node_encoding).float()

    def __getitem__(self, index):
        item = self.samples.iloc[index]

        seq = NASDataset._add_delim(item[0])
        seq, node_encoding = NASDataset._process_seq(seq)
        acc = float(item[1])

        return {
            'graph': seq,
            'node_encoding': node_encoding,
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
