import os
import sys
import torch
import igraph
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader
sys.path.append('../')
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h_0 = self.relu(self.fc_h0(z))
        inp = z.new_zeros((z.shape[0], self.max_seq_len - 1, self.hidden_size))
        h_out, _ = self.gru_layer(inp, h_0.unsqueeze(0))
        type_scores = self.add_vertex(h_out)
        edge_scores = self.sigmoid(self.add_edges(h_out))
        return type_scores, edge_scores


class Svae(pl.LightningModule):

    def __init__(self, batch_size, dataset_file):
        super(Svae, self).__init__()
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            batch_size, dataset_file, fmt='str')
        self.node_type = 8
        self.max_seq_len = 8
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='sum')

    # def construct_igraph(self, type_scores, edge_scores, stochastic=True):

    #     assert(type_scores.shape[:2] == edge_scores.shape[:2])
    #     if stochastic:
    #         type_probs = F.softmax(type_scores, dim=-1).detach()

    #     G = []
    #     adj_lists = []
    #     gen_node_encoding = []
    #     start_type = 0
    #     end_type = self.node_type - 1
    #     for g_idx in range(len(type_scores)):
    #         g = igraph.Graph(directed=True)
    #         g.add_vertex(type=start_type)
    #         for i in range(1, self.max_seq_len):
    #             if i == self.max_seq_len - 1:
    #                 node_type = end_type
    #             else:
    #                 if stochastic:
    #                     node_type = torch.multinomial(type_scores[g_idx, i - 1], 1).item()
    #                 else:
    #                     node_type = torch.argmax(type_scores[g_idx, i - 1]).item()
    #             if node_type == end_type:
    #                 end_vertices = g.vs.select(_outdegree_eq=0)
    #                 g.add_vertex(type=node_type)
    #                 for v in end_vertices:
    #                     g.add_edge(v, i)
    #                 break
    #             else:
    #                 g.add_vertex(type=node_type)
    #                 for ek in range(i):
    #                     ek_score = edge_scores[g_idx, i - 1, ek].item()
    #                     if stochastic:
    #                         if np.random.random_sample() < ek_score:
    #                             g.add_edge(ek, i)
    #                     else:
    #                         if ek_score > 0.5:
    #                             g.add_edge(ek, i)
    #         G.append(g)
    #         adj_lists.append(g.get_adjlist(igraph.IN))
    #     return G, adj_lists

    def reparamterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)

    def forward(self, X):
        mu, logvar = self.encoder(X)
        z = self.reparamterize(mu, logvar)
        type_scores, edge_scores = self.decoder(z)
        gen_node_types = torch.argmax(type_scores, dim=-1)
        gen_node_encoding = F.one_hot(
            gen_node_types, num_classes=self.node_type).to(dtype=torch.float)

        gen_dep_graph = torch.zeros_like(edge_scores)
        gen_dep_graph[edge_scores > 0.5] = 1

        return gen_dep_graph, gen_node_encoding, mu, logvar

    def training_step(self, batch, batch_nb):
        # REQUIRED
        dep_graph = batch['graph']
        node_encoding = dep_graph[:, :, :self.node_type]
        dep_matrix = dep_graph[:, :, self.node_type:]
        gen_dep_graph, gen_node_encoding, mu, logvar = self.forward(dep_graph)

        edge_loss = self.bce_loss(gen_dep_graph, dep_matrix)
        vertex_loss = self.cross_entropy_loss(
            gen_node_encoding.view(-1, self.node_type),
            torch.argmax(
                node_encoding.view(-1, self.node_type).to(dtype=torch.float), dim=1))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
        loss = edge_loss + vertex_loss + 0.005 * kl_loss

        tensorboard_logs = {
            'edge_loss': edge_loss,
            'vertex_loss': vertex_loss,
            'kl_loss': kl_loss,
            'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    # def validation_step(self, batch, batch_nb):
    #     # OPTIONAL
    #     x, y = batch
    #     y_hat = self.forward(x)
    #     return {'val_loss': F.cross_entropy(y_hat, y)}

    # def validation_end(self, outputs):
    #     # OPTIONAL
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'val_loss': avg_loss}
    #     return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

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
        return self.val_loader

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return self.test_loader
