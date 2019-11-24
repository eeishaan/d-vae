import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from dvae.utils.dataloader import get_dataloaders


class Encoder(nn.Module):
    def __init__(self, num_classes=7, hidden_state_size=501):
        super(Encoder, self).__init__()
        self.hidden_state_size = hidden_state_size
        self.gating_network = nn.Sequential(
            nn.Linear(hidden_state_size, hidden_state_size),
            nn.Sigmoid()
        )
        self.mapping_network = nn.Linear(hidden_state_size, hidden_state_size)
        self.gru = nn.GRUCell(num_classes, hidden_state_size)

        self.lin11 = nn.Linear(hidden_state_size, hidden_state_size)
        self.lin12 = nn.Linear(hidden_state_size, hidden_state_size)

    def forward(self, X):
        dep_graph, node_encoding, node_order = X['graph'], X['node_encoding'], X['node_order']
        batch_size, seq_len, _ = dep_graph.shape
        device = dep_graph.device
        node_hidden_state = torch.zeros(
            batch_size, seq_len, self.hidden_state_size, device=device)

        for node_seq_index in node_order:
            # TODO: add modification for NAS and bayesian task here
            # compute h_in
            ancestors = dep_graph[:, node_seq_index]
            ancestor_hidden_state = node_hidden_state * \
                ancestors.view(batch_size, -1, 1)
            ancestor_hidden_state = ancestor_hidden_state.view(
                -1, self.hidden_state_size)
            h_in = torch.sum(
                self.gating_network(ancestor_hidden_state) *
                self.mapping_network(ancestor_hidden_state),
                dim=0)

            # comute hv
            hv = self.gru(node_encoding[:, node_seq_index], h_in)
            assert hv.shape == (batch_size, self.hidden_state_size), \
                'Shape of hv is wrong, desired {} got {}'.format(
                    (batch_size, self.hidden_state_size), hv.shape)
            node_hidden_state[:, node_seq_index] = hv

        mu = self.lin11(hv)
        logvar = self.lin12(hv)
        return hv, mu, logvar


class Decoder(nn.Module):
    def __init__(self, num_classes=7, hidden_state_size=501):
        super(Decoder, self).__init__()
        self.hidden_state_size = hidden_state_size

        self.lin1 = nn.Linear(hidden_state_size, hidden_state_size)
        self.add_vertex = nn.Sequential(
            nn.Linear(hidden_state_size, hidden_state_size),
            nn.Softmax()
        )
        self.add_edge = nn.Sequential(
            nn.Linear(2*hidden_state_size, 1),
            nn.Sigmoid()
        )
        self.gru = nn.GRUCell(num_classes, hidden_state_size)

    def forward(self, z, X):
        pass


class Dvae(pl.LightningModule):

    def __init__(self, batch_size, dataset_file):
        super(Dvae, self).__init__()
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            batch_size, dataset_file)
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparamterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, X):
        _, mu, logvar = self.encoder(X)
        z = self.reparamterize(mu, logvar)
        X_hat = self.decoder(z, X)
        return X_hat, mu, logvar

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=0.02)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return self.train_loader

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return self.val_loader

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return self.test_loader
