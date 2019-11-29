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

        # capture input edges
        dep_dict = {node_idx: set() for node_idx in range(1, seq_len-1)}

        # capture edges with successor except eos and sos node
        is_successor = np.zeros((seq_len - 1))
        is_successor[0] = 1

        # dependency mask for easy computation
        dep_graph = np.zeros((seq_len, seq_len))

        # iterate over network
        for index, node in enumerate(x):
            for dep, val in enumerate(node[1:]):
                if val == 0:
                    continue
                # make note of input edges
                dep_dict[index + 1].add(dep + 1)

        top_sort = list(toposort(dep_dict))
        correct_node_order = [y for x in top_sort for y in sorted(x)]
        old_to_new_index = {old: new + 1 for new,
                            old in enumerate(correct_node_order)}
        for old_index, new_index in old_to_new_index.items():
            node = x[old_index - 1]

            node_type = node[0]
            # shift index by 1 as SOS node occupies 0th index
            node_encoding[new_index, node_type] = 1

            # make note of input edges
            for dep, val in enumerate(node[1:]):
                if val == 0:
                    continue
                # shift by 1 to account for SOS token
                new_dep_index = old_to_new_index[dep + 1]

                # make note of output edges
                is_successor[new_dep_index] = 1

                # update the dep graph
                dep_graph[new_index, new_dep_index] = 1

        # store encoding for first node
        node_encoding[0, -2] = 1
        # store encoding for last node
        node_encoding[-1, -1] = 1

        # tie all root nodes to SOS node
        root_nodes = list(top_sort[0])
        for node in root_nodes:
            dep_graph[old_to_new_index[node], 0] = 1

        # tie all end nodes to EOS node
        no_successor_nodes = np.argwhere(is_successor == 0)
        eos_graph_row = np.zeros((seq_len), dtype=int)
        eos_graph_row[no_successor_nodes] = 1
        dep_graph[-1, :] = eos_graph_row

        dep_graph = torch.from_numpy(dep_graph)
        node_encoding = torch.from_numpy(node_encoding)

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
