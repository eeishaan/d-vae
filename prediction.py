from argparse import Namespace
import argparse
import os
import pickle
import torch
import sys
import glob
import torch

# sys.path.append("../")
from dvae.constants import BATCH_SIZE, DATA_DIR, MODEL_DIR
from dvae.models.dvae_model import Dvae
from dvae.models.svae_model import Svae
from dvae.utils.dataloader import get_dataloaders

# checkpoint = torch.load(
#     "dvae/checkpoints/high_lr/_ckpt_epoch_157.ckpt", map_location=torch.device("cpu")
# )

# a = Namespace(**{"dataset_file": None, "mod": None, "bidirectional": True, "sgp": True})
# model = Svae(a)
model_dir = "dvae/checkpoints/svae_bayes_bidir/"
model_name = glob.glob(model_dir + "*.ckpt")[0]
model = Svae.load_from_checkpoint(model_name)
print(model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device=device)
# model.load_state_dict(checkpoint["state_dict"])
model.eval()
model.freeze()

dataset_file = os.path.join(DATA_DIR, "asia_200k.txt") #final_structures6
train_loader, val_loader, test_loader = get_dataloaders(
     BATCH_SIZE, dataset_file, fmt="str", task_type='bn'
)
#train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE, dataset_file)

y_val = []
latent_vectors = []
for batch in val_loader[0]:
    true_acc = batch["acc"]
    batch = {k: v.to(device) for k, v in batch.items()}
    #_, mu, logvar = model.encoder(batch)
    mu, logvar = model.encoder(batch['graph'])
    z = mu #model.reparamterize(mu, logvar)
    y_val.append(true_acc.detach().numpy())
    latent_vectors.append(z.detach().cpu().numpy())

print(latent_vectors[0].shape)
print(len(latent_vectors))
print(os.path.join(model_dir, "val_latent_rep.pkl"))
pickle.dump(latent_vectors, open(os.path.join(model_dir, "val_latent_rep.pkl"), "wb"))
pickle.dump(y_val, open(os.path.join(model_dir,"val_accs.pkl"), "wb"))

y_train = []
latent_vectors = []
for batch in train_loader:
    true_acc = batch["acc"]
    batch = {k: v.to(device) for k, v in batch.items()}
    #_, mu, logvar = model.encoder(batch)
    mu, logvar = model.encoder(batch['graph'])
    z = mu #model.reparamterize(mu, logvar)
    y_train.append(true_acc.detach().numpy())
    latent_vectors.append(z.detach().cpu().numpy())

print(latent_vectors[0].shape)
print(len(latent_vectors))
pickle.dump(latent_vectors, open(os.path.join(model_dir,"train_latent_rep.pkl"), "wb"))
pickle.dump(y_train, open(os.path.join(model_dir,"train_accs.pkl"), "wb"))

y_test = []
latent_vectors = []
for batch in test_loader:
    true_acc = batch["acc"]
    batch = {k: v.to(device) for k, v in batch.items()}
    #_, mu, logvar = model.encoder(batch)
    mu, logvar = model.encoder(batch['graph'])
    z = mu #model.reparamterize(mu, logvar)
    y_test.append(true_acc.detach().numpy())
    latent_vectors.append(z.detach().cpu().numpy())

print(latent_vectors[0].shape)
print(len(latent_vectors))
pickle.dump(latent_vectors, open(os.path.join(model_dir,"test_latent_rep.pkl"), "wb"))
pickle.dump(y_test, open(os.path.join(model_dir,"test_accs.pkl"), "wb"))
