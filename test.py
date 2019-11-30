import argparse
import os

import torch

from dvae.constants import BATCH_SIZE, DATA_DIR, MODEL_DIR
from dvae.models.dvae_model import Dvae
from dvae.utils.dataloader import get_dataloaders

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)

args = parser.parse_args()

model = Dvae.load_from_checkpoint(args.model)
device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# predict
model.eval()
model.freeze()

dataset_file = os.path.join(DATA_DIR,  'final_structures6.txt')
train_loader, val_loader, test_loader = get_dataloaders(
    BATCH_SIZE, dataset_file)
for x in test_loader:
    y_hat = model.predict(x)
