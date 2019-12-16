from argparse import Namespace
import argparse
import os

import torch
from tqdm import tqdm
from dvae.constants import BATCH_SIZE, DATA_DIR, MODEL_DIR
from dvae.models.dvae_model import Dvae
from dvae.models.svae_model import Svae
from dvae.utils.dataloader import get_dataloaders

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
checkpoint = torch.load(
    'dvae/checkpoints/lightning_logs/version_20/checkpoints/_ckpt_epoch_60.ckpt', map_location=device)

a = Namespace(**{
    'batch_size': None,
    'dataset_file': None,
    'mod': 'enas',
    'beta': 0.005,
    'task_type': 'bn',
    'num_classes': 8+2,
    'max_seq': 8+2,
    'bidirectional': False
})
model = Svae(a)

model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.freeze()
model.to(device)
task_type = 'bn'
dataset_file = os.path.join(DATA_DIR,  'asia_200k.txt')
train_loader, val_loader, test_loader = get_dataloaders(
    BATCH_SIZE, dataset_file, fmt='str', task_type=task_type)

# dataset_file = os.path.join(DATA_DIR,  'final_structures6.txt')
# train_loader, val_loader, test_loader = get_dataloaders(
#     BATCH_SIZE, dataset_file, fmt='str', task_type=task_type)


def check_validity(node_encoding, dep_graph, task_type):
    batch_size, seq_len, node_types = node_encoding.shape
    if task_type == 'enas':
        assert dep_graph.shape == (batch_size, seq_len, seq_len+1)

        # only one starting node
        start_node_type = node_types-2  # -1 for index
        is_only_one_start_node = (
            node_encoding[:, :, start_node_type] == 0).all(1)

        # only one ending node
        end_node_type = node_types-1
        is_last_node_end = (node_encoding[:, -1, end_node_type] == 1)
        is_only_one_last_node = (
            node_encoding[:, :-1, end_node_type] == 0).all(1)

        # each node except the first should have predecessors
        is_predecessor = (dep_graph.sum(-1) >= 1).all(1)

        # only output nodes doesn't have successors
        is_successor = (dep_graph.sum(1)[:, :-1] >= 1).all(1)

        # each node should have edge from its predecessor
        is_pred_connection = (torch.diagonal(
            dep_graph[:, :, :-1], dim1=1, dim2=2) == 1).all(1)

        # it's a DAG
        upper_triangular_mat = torch.ones_like(dep_graph)
        upper_triangular_mat = torch.triu(upper_triangular_mat)
        is_dag = (upper_triangular_mat * dep_graph == 0).all(2).all(1)

        valid = is_only_one_start_node * is_last_node_end * is_only_one_last_node * \
            is_predecessor * is_successor * is_pred_connection

    elif task_type == 'bn':
        # check if unique nodes
        expected_nodes = torch.zeros_like(node_encoding[0])
        expected_nodes[-1, -1] = 1
        diag = torch.diag(torch.as_tensor(
            [1]*(seq_len-2)), diagonal=1)[:-1, 1:].to(node_encoding.device)
        expected_nodes[:-1, :-2] = diag
        is_node_unique = (
            node_encoding == expected_nodes.unsqueeze(0)).all(2).all(1)

        # it's a DAG
        upper_triangular_mat = torch.ones_like(dep_graph)
        upper_triangular_mat = torch.triu(upper_triangular_mat)
        is_dag = (upper_triangular_mat * dep_graph == 0).all(2).all(1)

        valid = is_node_unique * is_dag

    else:
        raise NotImplementedError

    assert valid.shape == (batch_size,)
    return valid.sum().item()


# first compute mean for the training data
train_mu = []
with torch.no_grad():
    for x in tqdm(train_loader):
        x = {k: v.to(device) for k, v in x.items()}
        dep_graph = x['graph']
        mu, _ = model.encoder(dep_graph)
        train_mu.append(mu)
    train_mu = torch.cat(train_mu, dim=0)
    mean_train_mu = train_mu.mean(dim=0)
    std_train_mu = train_mu.std(dim=0)
    print(mean_train_mu, std_train_mu)
    decode_num = 10
    unique_z = 1000
    # hidden_size = mu.shape[-1]
    hidden_size = std_train_mu.shape[-1]
    sampled_z = torch.randn(unique_z, hidden_size, device=device)
    sampled_z = sampled_z * std_train_mu.unsqueeze(0) + mean_train_mu

    stack_z = sampled_z.repeat(1, decode_num, 1)
    sampled_z = stack_z.view(
        unique_z*decode_num, hidden_size)

    type_scores, edge_scores = model.decoder(sampled_z)
    gen_node_encoding, gen_dep_graph = model._predict_from_logits(
        type_scores, edge_scores, is_stochastic=True)
    print(gen_dep_graph.shape, gen_node_encoding.shape)
    valid_samples = check_validity(gen_node_encoding, gen_dep_graph, task_type)

    valid_samples /= (decode_num * unique_z)
    print(valid_samples)
