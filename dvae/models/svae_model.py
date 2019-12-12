import os
import sys

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dvae.utils.dataloader import get_dataloaders


class Encoder(nn.Module):

    def __init__(self, node_type=8, max_seq_len=8, hidden_size=501, latent_dim=56, bidir=False):
        super(Encoder, self).__init__()
        self.bidir = bidir
        self.node_type = node_type
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.inp_dim = node_type + max_seq_len
        # encoder GRU
        self.gru_layer = nn.GRU(self.inp_dim, hidden_size,
                                batch_first=True, bidirectional=self.bidir)
        # latent mean
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        # latent logvar
        self.fc_var = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        _, h_n = self.gru_layer(x)
        h_n = h_n.squeeze(0)

        mu = self.fc_mu(h_n)
        logvar = self.fc_var(h_n)
        return mu, logvar


class Decoder(nn.Module):

    def __init__(self, node_type=8, max_seq_len=8, hidden_size=501, latent_dim=56):
        super(Decoder, self).__init__()

        self.node_type = node_type
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        self.inp_dim = node_type + max_seq_len
        self.hidden_size = hidden_size
        # decoder GRU
        self.gru_layer = nn.GRU(hidden_size, hidden_size,
                                batch_first=True)

        self.fc_h0 = nn.Linear(latent_dim, hidden_size)
        self.add_vertex = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, node_type),
        )
        self.add_edges = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, max_seq_len),
        )
        self.relu = nn.ReLU()

    def forward(self, z):
        h_0 = self.relu(self.fc_h0(z))
        inp = h_0.unsqueeze(1).expand(-1, self.max_seq_len - 1, -1)
        h_out, _ = self.gru_layer(inp)
        type_scores = self.add_vertex(h_out)
        edge_scores = self.add_edges(h_out)
        return type_scores, edge_scores


class Svae(pl.LightningModule):

    def __init__(self, hparams):
        super(Svae, self).__init__()
        self.hparams = hparams
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            hparams.batch_size, hparams.dataset_file,  fmt='str') if hparams.dataset_file else [None]*3
        self.node_type = 8
        self.max_seq_len = 8
        self.encoder = Encoder(self.node_type, self.max_seq_len)
        self.decoder = Decoder(self.node_type, self.max_seq_len)
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='sum')
        self.beta = getattr(hparams, 'beta', 0.005)

    def reparamterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def _predict_from_logits(self, type_scores, edge_scores):
        gen_node_types = torch.argmax(type_scores, dim=-1)
        gen_node_encoding = F.one_hot(
            gen_node_types, num_classes=self.node_type).to(dtype=torch.float)

        gen_dep_graph = torch.zeros_like(edge_scores)
        gen_dep_graph[edge_scores > 0.5] = 1
        return gen_node_encoding, gen_dep_graph

    def predict(self, X):
        mu, logvar = self.encoder(X)
        z = self.reparamterize(mu, logvar)
        type_scores, edge_scores = self.decoder(z)
        gen_dep_graph, gen_node_encoding = self._predict_from_logits(
            type_scores, edge_scores)

        return gen_node_encoding, gen_dep_graph

    def forward(self, X):
        mu, logvar = self.encoder(X)
        z = self.reparamterize(mu, logvar)
        type_scores, edge_scores = self.decoder(z)

        return type_scores, edge_scores, mu, logvar

    def compute_accuracy(self, gen_dep_graph, gen_node_encoding, true_dep_graph, is_logits=False):
        if is_logits:
            gen_node_encoding, gen_dep_graph = self._predict_from_logits(
                gen_node_encoding, gen_dep_graph)
        pred_dep_graph = torch.cat(
            (gen_node_encoding, gen_dep_graph), dim=-1)
        is_equal = torch.eq(pred_dep_graph, true_dep_graph).all(
            dim=-1).all(dim=-1)
        acc = is_equal.sum() / is_equal.shape[0]
        return acc.float()

    def training_step(self, batch, batch_nb):
        # REQUIRED
        dep_graph = batch['graph']
        node_encoding = dep_graph[:, :, :self.node_type]
        dep_matrix = dep_graph[:, :, self.node_type:]

        gen_node_encoding, gen_dep_graph, mu, logvar = self.forward(dep_graph)

        edge_loss = self.bce_loss(gen_dep_graph, dep_matrix)

        vertex_loss = self.cross_entropy_loss(
            gen_node_encoding.view(-1, self.node_type),
            torch.argmax(
                node_encoding.view(-1, self.node_type).to(dtype=torch.float), dim=1))

        kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())

        loss = edge_loss + vertex_loss + self.beta * kl_loss

        with torch.no_grad():
            acc = self.compute_accuracy(
                gen_dep_graph.detach(), gen_node_encoding.detach(), dep_graph, is_logits=True)

        tensorboard_logs = {
            'train/edge_loss': edge_loss,
            'train/vertex_loss': vertex_loss,
            'train/kl_loss': kl_loss,
            'train/loss': loss,
            'train/acc': acc
        }
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        with torch.no_grad():
            dep_graph = batch['graph']
            node_encoding = dep_graph[:, :, :self.node_type]
            dep_matrix = dep_graph[:, :, self.node_type:]
            gen_node_encoding, gen_dep_graph, mu, logvar = self.forward(
                dep_graph)

            edge_loss = self.bce_loss(gen_dep_graph, dep_matrix)

            vertex_loss = self.cross_entropy_loss(
                gen_node_encoding.view(-1, self.node_type),
                torch.argmax(
                    node_encoding.view(-1, self.node_type).to(dtype=torch.float), dim=1))

            kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())

            loss = edge_loss + vertex_loss + self.beta * kl_loss
            acc = self.compute_accuracy(
                gen_dep_graph.detach(), gen_node_encoding.detach(), dep_graph, is_logits=True)

        return {
            'val/edge_loss': edge_loss,
            'val/vertex_loss': vertex_loss,
            'val/kl_loss': kl_loss,
            'val_loss': loss,
            'val/acc': acc
        }

    def validation_end(self, outputs):
        # OPTIONAL
        with torch.no_grad():
            loss_keys = ['val_loss', 'val/edge_loss',
                         'val/vertex_loss', 'val/kl_loss', 'val/acc']
            metrics = {
                key: torch.stack([x[key] for x in outputs]).mean() for key in loss_keys
            }
        return {'avg_val_loss': metrics['val_loss'], 'log': metrics}

    def configure_optimizers(self):
        # REQUIRED
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, min_lr=1e-5)
        return [optimizer], [lr_schedule]

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return self.train_loader

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return self.val_loader[0]

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return self.test_loader
