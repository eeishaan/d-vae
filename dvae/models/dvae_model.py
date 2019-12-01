import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dvae.utils.dataloader import get_dataloaders


class Encoder(nn.Module):
    def __init__(self, num_classes=8, hidden_state_size=501, latent_space_dim=56, mod=None):
        super(Encoder, self).__init__()
        self.hidden_state_size = hidden_state_size
        self.gating_network = nn.Sequential(
            nn.Linear(hidden_state_size, hidden_state_size),
            nn.Sigmoid()
        )
        self.mapping_network = nn.Linear(hidden_state_size, hidden_state_size)
        self.gru = nn.GRUCell(num_classes, hidden_state_size)

        self.lin11 = nn.Linear(hidden_state_size, latent_space_dim)
        self.lin12 = nn.Linear(hidden_state_size, latent_space_dim)
        self.mod = mod

    def forward(self, X):
        dep_graph, node_encoding = X['graph'], X['node_encoding']
        batch_size, seq_len, _ = dep_graph.shape
        device = dep_graph.device
        node_hidden_state = torch.zeros(
            batch_size, seq_len, self.hidden_state_size, device=device)

        for index in range(seq_len):
            # TODO: add modification for NAS and bayesian task here

            ########## compute h_in ##########
            ancestor_mask = dep_graph[:, index].unsqueeze(-1)

            # broadcast mask here to zero out non-ancestor nodes
            h_in = self.gating_network(node_hidden_state) * \
                self.mapping_network(node_hidden_state) * \
                ancestor_mask
            h_in = torch.sum(h_in, dim=1)

            ########## comute hv ##########
            hv = self.gru(node_encoding[:, index], h_in)
            assert hv.shape == (batch_size, self.hidden_state_size), \
                'Shape of hv is wrong, desired {} got {}'.format(
                    (batch_size, self.hidden_state_size), hv.shape)
            node_hidden_state[:, index] = hv

        mu = self.lin11(hv)
        logvar = self.lin12(hv)
        return hv, mu, logvar

    def predict(self, X):
        return self.forward(X)


class Decoder(nn.Module):
    def __init__(self, num_classes=8, hidden_state_size=501, latent_space_dim=56, mod=None):
        super(Decoder, self).__init__()
        self.hidden_state_size = hidden_state_size
        self.num_classes = num_classes
        self.lin1 = nn.Linear(latent_space_dim, hidden_state_size)
        self.add_vertex = nn.Sequential(
            nn.Linear(hidden_state_size, 2*hidden_state_size),
            nn.ReLU(),
            nn.Linear(2*hidden_state_size, num_classes),
        )
        self.add_edge = nn.Sequential(
            nn.Linear(2*hidden_state_size, 4*hidden_state_size),
            nn.ReLU(),
            nn.Linear(4*hidden_state_size, 1),
        )
        self.gating_network = nn.Sequential(
            nn.Linear(hidden_state_size, hidden_state_size),
            nn.Sigmoid()
        )
        self.mapping_network = nn.Linear(hidden_state_size, hidden_state_size)
        self.gru = nn.GRUCell(num_classes, hidden_state_size)
        self.mod = mod

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
                hv = self.gru(node_encoding[:, index], h_in)
            else:
                hv = self.gru(node_encoding[:, index])

            # sample edges
            for v_j in range(index-1, -1, -1):
                hidden_j = node_hidden_state[:, v_j]
                is_edge = self.add_edge(torch.cat([hv, hidden_j], dim=1))
                gen_dep_graph[:, index, v_j] = is_edge.view(-1)

                # TODO: add modification for NAS and bayesian task here
                ######### compute h_in #########
                ancestors = dep_graph[:, index]
                ancestors_mask = torch.ones_like(ancestors)
                # zero out all ancestors before v_j
                ancestors_mask[:, :v_j] = 0
                ancestors_mask = ancestors * ancestors_mask
                ancestors_mask.unsqueeze_(-1)
                h_in = self.gating_network(node_hidden_state) * \
                    self.mapping_network(node_hidden_state) * \
                    ancestors_mask
                h_in = torch.sum(h_in, dim=1)

                ######### compute hv #########
                hv = self.gru(node_encoding[:, index], h_in)
                assert hv.shape == (batch_size, self.hidden_state_size), \
                    'Shape of hv is wrong, desired {} got {}'.format(
                        (batch_size, self.hidden_state_size), hv.shape)
                node_hidden_state[:, index] = hv
            node_hidden_state[:, index] = hv
            graph_state = hv
        return gen_dep_graph, gen_node_encoding

    def predict(self, z):
        gen_edges = []
        gen_nodes = []
        batch_size, _ = z.shape
        node_hidden_state = []
        graph_state = self.lin1(z)

        index = 0
        while True:
            if index == 8:
                break
            # sample node type
            new_node_type = F.softmax(
                self.add_vertex(graph_state), dim=1)
            new_node_type = new_node_type.argmax(dim=1)
            gen_nodes.append(new_node_type)

            new_node_encoding = F.one_hot(
                new_node_type, num_classes=self.num_classes).float()

            # compute initial hidden states
            if index == 0:
                h_in = graph_state
                hv = self.gru(new_node_encoding, h_in)
            else:
                hv = self.gru(new_node_encoding)

            possible_edges = torch.zeros(
                batch_size, max(0, index), dtype=torch.float)
            # sample edges
            for v_j in range(index-1, -1, -1):
                hidden_j = node_hidden_state[v_j]
                is_edge = self.add_edge(torch.cat([hv, hidden_j], dim=1))
                is_edge = F.sigmoid(is_edge)
                is_edge[is_edge >= 0.5] = 1
                is_edge[is_edge < 0.5] = 0
                possible_edges[:, v_j] = is_edge.view(-1)

                # TODO: add modification for NAS and bayesian task here
                ######### compute h_in #########

                c_node_hidden_state = torch.stack(
                    node_hidden_state).permute(1, 0, 2)
                h_in = self.gating_network(c_node_hidden_state) * \
                    self.mapping_network(c_node_hidden_state) * \
                    possible_edges.unsqueeze(-1)
                h_in = h_in.view(batch_size, -1, self.hidden_state_size)
                h_in = torch.sum(h_in, dim=1)

                ######### compute hv #########
                hv = self.gru(new_node_encoding, h_in)
                assert hv.shape == (batch_size, self.hidden_state_size), \
                    'Shape of hv is wrong, desired {} got {}'.format(
                        (batch_size, self.hidden_state_size), hv.shape)
            gen_edges.append(possible_edges)
            node_hidden_state.append(hv)
            graph_state = hv
            index += 1
        return gen_edges, gen_nodes


class Dvae(pl.LightningModule):

    def __init__(self, hparams):
        """
        batch_size: batch size 
        dataset_file: dataset file
        mod: Modification mode. Allowed values: None, 'nn' and 'bayesian'
        """
        super(Dvae, self).__init__()
        self.hparams = hparams
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            hparams.batch_size, hparams.dataset_file) if hparams.dataset_file else [None]*3
        self.encoder = Encoder(mod=hparams.mod)
        self.decoder = Decoder(mod=hparams.mod)
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='sum')
        self.num_classes = 8
        self.mod = hparams.mod

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
        dep_graph, node_encoding = batch['graph'], batch['node_encoding']
        (gen_dep_graph, gen_node_encoding), mu, logvar = self.forward(batch)
        edge_loss = self.bce_loss(gen_dep_graph, dep_graph)
        vertex_loss = self.cross_entropy_loss(
            gen_node_encoding.view(-1, self.num_classes),
            torch.argmax(
                node_encoding.view(-1, self.num_classes), dim=1))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
        loss = edge_loss + vertex_loss + 0.005 * kl_loss

        tensorboard_logs = {
            'edge_loss': edge_loss,
            'vertex_loss': vertex_loss,
            'kl_loss': kl_loss,
            'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        dep_graph, node_encoding = batch['graph'], batch['node_encoding']
        (gen_dep_graph, gen_node_encoding), mu, logvar = self.forward(batch)
        edge_loss = self.bce_loss(gen_dep_graph, dep_graph,)
        vertex_loss = self.cross_entropy_loss(
            gen_node_encoding.view(-1, self.num_classes),
            torch.argmax(
                node_encoding.view(-1, self.num_classes), dim=1))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
        loss = edge_loss + vertex_loss + 0.005 * kl_loss
        return {
            'val_edge_loss': edge_loss,
            'val_vertex_loss': vertex_loss,
            'val_kl_loss': kl_loss,
            'val_loss': loss}

    def validation_end(self, outputs):
        # OPTIONAL
        loss_keys = ['val_loss', 'val_edge_loss',
                     'val_vertex_loss', 'val_kl_loss']
        metrics = {
            key: torch.stack([x[key] for x in outputs]).mean() for key in loss_keys
        }
        return {'avg_val_loss': metrics['val_loss'], 'log': metrics}

    def configure_optimizers(self):
        # REQUIRED
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
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

    def predict(self, X):
        _, mu, logvar = self.encoder.predict(X)
        z = self.reparamterize(mu, logvar)
        X_hat = self.decoder.predict(z)
        return X_hat
