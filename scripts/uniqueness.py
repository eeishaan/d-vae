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
    'dvae/checkpoints/high_lr/_ckpt_epoch_157.ckpt', map_location=device)

a = Namespace(**{'dataset_file': None, 'mod': None, 'bidirectional': True})

model = Dvae(a)

model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.freeze()
model.to(device)
task_type = 'enas'
dataset_file = os.path.join(DATA_DIR,  'final_structures6.txt')
train_loader, val_loader, test_loader = get_dataloaders(
    BATCH_SIZE, dataset_file, task_type=task_type)


# first compute mean for the training data
train_mu = []
with torch.no_grad():
    for x in tqdm(train_loader):
        x = {k: v.to(device) for k, v in x.items()}
        _, mu, _ = model.encoder(x)
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

    gen_edges, gen_nodes = model.decoder.predict(
        sampled_z, is_stochastic=True)

    concat = torch.cat((gen_edges, gen_nodes), dim=2).view(
        sampled_z.shape[0], -1)
    unique = len(torch.unique(concat, dim=0)) / concat.shape[0]
    print(unique)
