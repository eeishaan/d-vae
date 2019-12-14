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
    "dvae/checkpoints/lightning_logs/version_0/checkpoints/_ckpt_epoch_107.ckpt",
    map_location=torch.device("cpu"),
)

a = Namespace(**{"dataset_file": None, "mod": None, "bidirectional": True, "sgp": True})
model = Svae(a)
# model = Dvae(a)

model.load_state_dict(checkpoint["state_dict"])
model.eval()
model.freeze()

dataset_file = os.path.join(DATA_DIR, "final_structures6.txt")
train_loader, val_loader, test_loader = get_dataloaders(
    BATCH_SIZE, dataset_file, fmt="str"
)
# train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE, dataset_file)
acc = 0
# y_train = []
# latent_vectors = []
# for batch in val_loader[0]:
#     true_acc = batch["acc"]
#     _, mu, logvar = model.encoder(batch)
#     z = model.reparamterize(mu, logvar)
#     y_train.append(true_acc.detach().numpy())
#     latent_vectors.append(z.detach().numpy())

# print(latent_vectors[0].shape)
# print(len(latent_vectors))
# pickle.dump(latent_vectors, open(f"./val_latent_rep.pkl", "wb"))
# pickle.dump(y_train, open(f"./val_accs.pkl", "wb"))

for x in test_loader:
    res = model.model_eval(x, 10, 10)
    acc += res["recon_acc"]
acc / len(test_loader)

g = x["graph"]
n = x["node_encoding"]

g_0 = torch.cat([g[0].unsqueeze(0)] * 10, dim=0)
n_0 = torch.cat([n[0].unsqueeze(0)] * 10, dim=0)

inp_0 = {"graph": g_0, "node_encoding": n_0}
out_0 = model.predict(inp_0)
# out_0
# g_0
# n_0
