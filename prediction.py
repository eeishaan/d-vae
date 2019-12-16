from argparse import Namespace
import argparse
import os
import pickle
import torch
import sys

# sys.path.append("../")
from dvae.constants import BATCH_SIZE, DATA_DIR, MODEL_DIR
from dvae.models.dvae_model import Dvae
from dvae.models.svae_model import Svae
from dvae.utils.dataloader import get_dataloaders

checkpoint = torch.load(
    "dvae/checkpoints/high_lr/_ckpt_epoch_157.ckpt", map_location=torch.device("cpu")
)

a = Namespace(**{"dataset_file": None, "mod": None, "bidirectional": True, "sgp": True})
# model = Svae(a)
model = Dvae(a)

model.load_state_dict(checkpoint["state_dict"])
model.eval()
model.freeze()

dataset_file = os.path.join(DATA_DIR, "final_structures6.txt")
# train_loader, val_loader, test_loader = get_dataloaders(
#     BATCH_SIZE, dataset_file, fmt="str"
# )
train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE, dataset_file)
acc = 0
y_train = []
latent_vectors = []
for batch in val_loader[0]:
    true_acc = batch["acc"]
    _, mu, logvar = model.encoder(batch)
    z = model.reparamterize(mu, logvar)
    y_train.append(true_acc.detach().numpy())
    latent_vectors.append(z.detach().numpy())

print(latent_vectors[0].shape)
print(len(latent_vectors))
pickle.dump(latent_vectors, open(f"./val_latent_rep.pkl", "wb"))
pickle.dump(y_train, open(f"./val_accs.pkl", "wb"))

acc = 0
y_train = []
latent_vectors = []
for batch in train_loader:
    true_acc = batch["acc"]
    _, mu, logvar = model.encoder(batch)
    z = model.reparamterize(mu, logvar)
    y_train.append(true_acc.detach().numpy())
    latent_vectors.append(z.detach().numpy())

print(latent_vectors[0].shape)
print(len(latent_vectors))
pickle.dump(latent_vectors, open(f"./train_latent_rep.pkl", "wb"))
pickle.dump(y_train, open(f"./train_accs.pkl", "wb"))

acc = 0
y_train = []
latent_vectors = []
for batch in test_loader:
    true_acc = batch["acc"]
    _, mu, logvar = model.encoder(batch)
    z = model.reparamterize(mu, logvar)
    y_train.append(true_acc.detach().numpy())
    latent_vectors.append(z.detach().numpy())

print(latent_vectors[0].shape)
print(len(latent_vectors))
pickle.dump(latent_vectors, open(f"./test_latent_rep.pkl", "wb"))
pickle.dump(y_train, open(f"./test_accs.pkl", "wb"))
