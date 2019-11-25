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
        dep_graph, node_encoding = X['graph'], X['node_encoding']
        batch_size, seq_len, _ = dep_graph.shape
        device = dep_graph.device
        node_hidden_state = torch.zeros(
            batch_size, seq_len, self.hidden_state_size, device=device)

        for index in range(seq_len):
            # TODO: add modification for NAS and bayesian task here
            # compute h_in
            ancestors = dep_graph[:, index]
            ancestor_hidden_state = node_hidden_state * \
                ancestors.view(batch_size, -1, 1)
            ancestor_hidden_state = ancestor_hidden_state.view(
                -1, self.hidden_state_size)
            h_in = self.gating_network(ancestor_hidden_state) * \
                self.mapping_network(ancestor_hidden_state)
            h_in = h_in.view(batch_size, -1, self.hidden_state_size)
            h_in = torch.sum(h_in, dim=1)

            # comute hv
            hv = self.gru(node_encoding[:, index], h_in)
            assert hv.shape == (batch_size, self.hidden_state_size), \
                'Shape of hv is wrong, desired {} got {}'.format(
                    (batch_size, self.hidden_state_size), hv.shape)
            node_hidden_state[:, index] = hv

        mu = self.lin11(hv)
        logvar = self.lin12(hv)
        return hv, mu, logvar


class Decoder(nn.Module):
    def __init__(self, num_classes=7, hidden_state_size=501):
        super(Decoder, self).__init__()
        self.hidden_state_size = hidden_state_size
        self.num_classes = num_classes
        self.lin1 = nn.Linear(hidden_state_size, hidden_state_size)
        self.add_vertex = nn.Sequential(
            nn.Linear(hidden_state_size, num_classes),
            nn.Softmax()
        )
        self.add_edge = nn.Sequential(
            nn.Linear(2*hidden_state_size, 1),
            nn.Sigmoid()
        )
        self.gating_network = nn.Sequential(
            nn.Linear(hidden_state_size, hidden_state_size),
            nn.Sigmoid()
        )
        self.mapping_network = nn.Linear(hidden_state_size, hidden_state_size)
        self.gru = nn.GRUCell(num_classes, hidden_state_size)

    def forward(self, z, X):
        dep_graph, node_encoding = X['graph'], X['node_encoding']

        batch_size, seq_len, _ = dep_graph.shape
        gen_dep_graph = torch.zeros_like(dep_graph)
        gen_node_encoding = torch.zeros_like(node_encoding)

        device = dep_graph.device
        node_hidden_state = torch.zeros(
            batch_size, seq_len, self.hidden_state_size, device=device)

        graph_state = self.lin1(z)

        for index in range(seq_len):

            # sample node type
            new_node_type_logits = self.add_vertex(graph_state)
            gen_node_encoding[:, index, :] = new_node_type_logits

            # compute initial hidden states
            if index == 0:
                h_in = graph_state
            else:
                pass
                # TODO: Compute initial h_v

            # sample edges
            for v_j in range(index-1, -1, -1):
                hidden_j = node_hidden_state[:, v_j]
                is_edge = self.add_edge(torch.cat([h_v, hidden_j], dim=0))
                is_edge[is_edge < 0.5] = 0
                is_edge[is_edge >= 0.5] = 1
                gen_dep_graph[:, index, v_j] = is_edge

                # TODO: add modification for NAS and bayesian task here
                # compute h_in
                ancestors = dep_graph[:, index]
                # warning verify if happening in-place
                # zero out all ancestors before v_j
                ancestors[:, :v_j] = 0
                ancestor_hidden_state = node_hidden_state * \
                    ancestors.view(batch_size, -1, 1)
                ancestor_hidden_state = ancestor_hidden_state.view(
                    -1, self.hidden_state_size)
                h_in = self.gating_network(ancestor_hidden_state) * \
                    self.mapping_network(ancestor_hidden_state)
                h_in = h_in.view(batch_size, -1, self.hidden_state_size)
                h_in = torch.sum(h_in, dim=1)

                # comute hv
                hv = self.gru(node_encoding[:, index], h_in)
                assert hv.shape == (batch_size, self.hidden_state_size), \
                    'Shape of hv is wrong, desired {} got {}'.format(
                        (batch_size, self.hidden_state_size), hv.shape)
                node_hidden_state[:, index] = hv
            node_hidden_state[:, index] = hv
            graph_state = hv
        return gen_dep_graph, gen_node_encoding


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
