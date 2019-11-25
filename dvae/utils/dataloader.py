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
        node_types = 6 + 1  # 1 for EOS token
        seq_len = len(x) + 1  # 1 for EOS token

        # store 1-hot encoding for nodes in seq
        node_encoding = np.zeros((seq_len, node_types), dtype=int)

        # capture input edges
        dep_dict = {node_idx: set() for node_idx in range(seq_len-1)}

        # capture edges with successor except eos node
        is_successor = np.zeros((seq_len - 1))

        # dependency mask for easy computation
        dep_graph = np.zeros((seq_len, seq_len))

        # iterate over network
        for index, node in enumerate(x):
            node_type = node[0]
            node_encoding[index, node_type] = 1

            # make note of input edges
            for dep, val in enumerate(node[1:]):
                if val == 0:
                    continue

                dep_dict[index].add(dep)

                # make note of output edges
                is_successor[dep] = 1

                # update the dep graph
                dep_graph[index, dep] = 1

        # store encoding for last node
        node_encoding[-1, -1] = 1

        # compute iteration order
        top_sort = list(toposort(dep_dict))
        node_order = [y for x in top_sort for y in sorted(x)]
        node_order.append(seq_len - 1)  # add EOS node at the end
        node_order = torch.as_tensor(node_order, dtype=torch.long)

        no_successor_nodes = np.argwhere(is_successor == 0)
        eos_graph_row = np.zeros((seq_len), dtype=int)
        eos_graph_row[no_successor_nodes] = 1
        dep_graph[-1, :] = eos_graph_row

        dep_graph = torch.from_numpy(dep_graph)
        node_encoding = torch.from_numpy(node_encoding)

        # reorder as per topological order
        dep_graph = torch.index_select(dep_graph, 0, node_order)
        node_encoding = torch.index_select(node_encoding, 0, node_order)
        return dep_graph, node_encoding

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
