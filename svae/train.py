import os
from argparse import Namespace
import argparse

import torch
from pytorch_lightning import Trainer

from constants import BATCH_SIZE, DATA_DIR, MODEL_DIR
from svae_model import Svae
from dvae.utils.dataloader import get_dataloaders


def train(epoch, data_loader, model, optimizer, device):
    model.train()
    total_loss = 0.
    count = 0.
    node_type = 8

    for i, batch in enumerate(data_loader):

        optimizer.zero_grad()

        dep_graph = batch['graph'].to(device)
        node_encoding = dep_graph[:, :, :node_type]
        dep_matrix = dep_graph[:, :, node_type:]

        gen_dep_graph, gen_node_encoding, mu, logvar = model.forward(dep_graph)

        edge_loss = model.bce_loss(gen_dep_graph, dep_matrix)
        vertex_loss = model.cross_entropy_loss(
            gen_node_encoding.view(-1, node_type),
            torch.argmax(
                node_encoding.view(-1, node_type).to(dtype=torch.float), dim=1))

        kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())

        loss = edge_loss + vertex_loss + 0.005 * kl_loss
        loss.backward()
        optimizer.step()

        count += model.compute_accuracy(gen_dep_graph, gen_node_encoding, dep_graph)

        total_loss += loss.item()

        if i % 10 == 0:
            print('| epoch {:3d} | loss: {}'.format(epoch, total_loss / (i + 1)))

    accuracy = 100 * count.item() / len(data_loader)
    print('|Train | Epoch: {:3d} | accuracy: {}'.format(epoch, accuracy))
    return total_loss, accuracy


def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.
    count = 0.
    node_type = 8
    with torch.no_grad():
        for i, batch in enumerate(data_loader):

            dep_graph = batch['graph'].to(device)
            node_encoding = dep_graph[:, :, :node_type]
            dep_matrix = dep_graph[:, :, node_type:]

            gen_dep_graph, gen_node_encoding, mu, logvar = model.forward(dep_graph)

            edge_loss = model.bce_loss(gen_dep_graph, dep_matrix)
            vertex_loss = model.cross_entropy_loss(
                gen_node_encoding.view(-1, node_type),
                torch.argmax(
                    node_encoding.view(-1, node_type).to(dtype=torch.float), dim=1))

            kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())

            loss = edge_loss + vertex_loss + 0.005 * kl_loss

            count += model.compute_accuracy(gen_dep_graph, gen_node_encoding, dep_graph)

            total_loss += loss.item()

    accuracy = 100 * count.item() / len(data_loader)
    avg_loss = total_loss / (i * len(data_loader))
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', default=300, type=int, help='no. of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-4, type=float)
    args = parser.parse_args()
    print(args)

    dataset_file = os.path.join(DATA_DIR, 'final_structures6.txt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Svae(BATCH_SIZE, dataset_file)
    model.to(device)

    train_loader, val_loader, test_loader = get_dataloaders(
        BATCH_SIZE, dataset_file, fmt='str')
    val_loader = val_loader[0]
    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs = args.max_epochs
    best_model = None
    best_epoch = 0
    lr = args.lr

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', min_lr=1e-5)

    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    best_model = None

    for epoch in range(1, epochs + 1):

        train_loss, train_acc = train(epoch, train_loader, model, optimizer, device)
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        val_loss, val_acc = evaluate(model, val_loader, device)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        print('-' * 89)
        print('| End of epoch {:3d} | valid loss: {} | valid accuracy: {}'.format(
            epoch, val_loss, val_acc))
        print('-' * 89)

        lr_schedule.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model = model
            best_epoch = epoch
            torch.save(model.state_dict(), "./best_model_lstm.pt")

    print('Best val loss: {} | acc: {} at epoch {}'.format(
        best_val_loss, best_val_acc, best_epoch))
    test_loss, test_acc = evaluate(best_model, test_loader)
    print('Test | loss: {} | acc: {}'.format(
        test_loss, test_acc))

    log_results = {'train_acc': train_accs,
                   'train_loss': train_losses,
                   'val_acc': val_accs,
                   'val_loss': val_losses,
                   'best_loss': best_val_loss,
                   'best_acc': best_val_acc,
                   'test_loss': test_loss,
                   'test_acc': test_acc
                   }
    pickle.dump(log_results, open(f'log_results.pkl', 'wb'))

if __name__ == '__main__':
    main()
