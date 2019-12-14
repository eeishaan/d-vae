import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.decomposition import PCA
import logging
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from time import gmtime, strftime
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from random import shuffle
import pickle
from tensorboard_logger import configure, log_value
import scipy.misc
import time as tm

from utils import *
from model import *
from data import *
from args import Args
import create_graphs



def train_mlp_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        x = Variable(x)
        y = Variable(y)
        h = rnn(x, pack=True, input_len=y_len)
        y_pred = output(h)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, y)
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()
        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs,loss.data.item(), args.graph_type, args.num_layers, args.hidden_size_rnn))
        # logging
        log_value('loss_'+args.fname, loss.data.item(), epoch*args.batch_ratio+batch_idx)
        loss_sum += loss.data.item()
    return loss_sum/(batch_idx+1)

def test_mlp_epoch(epoch, args, rnn, output, test_batch_size=16, save_histogram=False,sample_time=1):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()
    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)) # normalized prediction score
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)) # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node))
    for i in range(max_num_node):
        h = rnn(x_step)
        y_pred_step = output(h)
        y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
        x_step = sample_sigmoid(y_pred_step, sample=True, sample_time=sample_time)
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data)
    y_pred_data = y_pred.data
    y_pred_long_data = y_pred_long.data.long()
    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)
    # # save prediction histograms, plot histogram over each time step
    # if save_histogram:
    #     save_prediction_histogram(y_pred_data.cpu().numpy(),
    #                           fname_pred=args.figure_prediction_save_path+args.fname_pred+str(epoch)+'.jpg',
    #                           max_num_node=max_num_node)
    return G_pred_list

def test_mlp_partial_epoch(epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):
    rnn.eval()
    output.eval()
    G_pred_list = []
    for batch_idx, data in enumerate(data_loader):
        x = data['x'].float()
        y = data['y'].float()
        y_len = data['len']
        test_batch_size = x.size(0)
        rnn.hidden = rnn.init_hidden(test_batch_size)
        # generate graphs
        max_num_node = int(args.max_num_node)
        y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)) # normalized prediction score
        y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)) # discrete prediction
        x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node))
        for i in range(max_num_node):
            print('finish node',i)
            h = rnn(x_step)
            y_pred_step = output(h)
            y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
            x_step = sample_sigmoid_supervised(y_pred_step, y[:,i:i+1,:], current=i, y_len=y_len, sample_time=sample_time)
            y_pred_long[:, i:i + 1, :] = x_step
            rnn.hidden = Variable(rnn.hidden.data)
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()
        # save graphs as pickle
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
    return G_pred_list

def test_mlp_partial_simple_epoch(epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):
    rnn.eval()
    output.eval()
    G_pred_list = []
    for batch_idx, data in enumerate(data_loader):
        x = data['x'].float()
        y = data['y'].float()
        y_len = data['len']
        test_batch_size = x.size(0)
        rnn.hidden = rnn.init_hidden(test_batch_size)
        # generate graphs
        max_num_node = int(args.max_num_node)
        y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)) # normalized prediction score
        y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)) # discrete prediction
        x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node))
        for i in range(max_num_node):
            print('finish node',i)
            h = rnn(x_step)
            y_pred_step = output(h)
            y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
            x_step = sample_sigmoid_supervised_simple(y_pred_step, y[:,i:i+1,:], current=i, y_len=y_len, sample_time=sample_time)
            y_pred_long[:, i:i + 1, :] = x_step
            rnn.hidden = Variable(rnn.hidden.data)
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()
        # save graphs as pickle
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
    return G_pred_list

def train_mlp_forward_epoch(epoch, args, rnn, output, data_loader):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        x = Variable(x)
        y = Variable(y)
        h = rnn(x, pack=True, input_len=y_len)
        y_pred = output(h)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        # use cross entropy loss
        loss = 0
        for j in range(y.size(1)):
            # print('y_pred',y_pred[0,j,:],'y',y[0,j,:])
            end_idx = min(j+1,y.size(2))
            loss += binary_cross_entropy_weight(y_pred[:,j,0:end_idx], y[:,j,0:end_idx])*end_idx
        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs,loss.data.item(), args.graph_type, args.num_layers, args.hidden_size_rnn))
        # logging
        log_value('loss_'+args.fname, loss.data.item(), epoch*args.batch_ratio+batch_idx)
        loss_sum += loss.data.item()
    return loss_sum/(batch_idx+1)


########### train function for LSTM + VAE
def train(args, dataset_train, rnn, output):
    # check if load existing model
    if args.load:
        fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
        rnn.load_state_dict(torch.load(fname))
        fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
        output.load_state_dict(torch.load(fname))

        args.lr = 0.00001
        epoch = args.load_epoch
        print('model loaded!, lr: {}'.format(args.lr))
    else:
        epoch = 1
    # initialize optimizer
    optimizer_rnn = optim.Adam(list(rnn.parameters()), lr=args.lr)
    optimizer_output = optim.Adam(list(output.parameters()), lr=args.lr)
    scheduler_rnn = MultiStepLR(optimizer_rnn, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_output = MultiStepLR(optimizer_output, milestones=args.milestones, gamma=args.lr_rate)
    # start main loop
    time_all = np.zeros(args.epochs)
    while epoch<=args.epochs:
        time_start = tm.time()
        # train
        if 'GraphRNN_VAE' in args.note:
            train_vae_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        elif 'GraphRNN_MLP' in args.note:
            train_mlp_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        elif 'GraphRNN_RNN' in args.note:
            train_rnn_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start
        # test
        if epoch % args.epochs_test == 0 and epoch>=args.epochs_test_start:
            for sample_time in range(1,4):
                G_pred = []
                while len(G_pred)<args.test_total_size:
                    if 'GraphRNN_VAE' in args.note:
                        G_pred_step = test_vae_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,sample_time=sample_time)
                    elif 'GraphRNN_MLP' in args.note:
                        G_pred_step = test_mlp_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,sample_time=sample_time)
                    elif 'GraphRNN_RNN' in args.note:
                        G_pred_step = test_rnn_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size)
                    G_pred.extend(G_pred_step)
                # save graphs
                fname = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + '.dat'
                save_graph_list(G_pred, fname)
                if 'GraphRNN_RNN' in args.note:
                    break
            print('test done, graphs saved')
        # save model checkpoint
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + args.fname + 'lstm_' + str(epoch) + '.dat'
                torch.save(rnn.state_dict(), fname)
                fname = args.model_save_path + args.fname + 'output_' + str(epoch) + '.dat'
                torch.save(output.state_dict(), fname)
        epoch += 1
    np.save(args.timing_save_path+args.fname,time_all)


########### for graph completion task
def train_graph_completion(args, dataset_test, rnn, output):
    fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
    rnn.load_state_dict(torch.load(fname))
    fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
    output.load_state_dict(torch.load(fname))
    epoch = args.load_epoch
    print('model loaded!, epoch: {}'.format(args.load_epoch))
    for sample_time in range(1,4):
        if 'GraphRNN_MLP' in args.note:
            G_pred = test_mlp_partial_simple_epoch(epoch, args, rnn, output, dataset_test,sample_time=sample_time)
        if 'GraphRNN_VAE' in args.note:
            G_pred = test_vae_partial_epoch(epoch, args, rnn, output, dataset_test,sample_time=sample_time)
        # save graphs
        fname = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + 'graph_completion.dat'
        save_graph_list(G_pred, fname)
    print('graph completion done, graphs saved')


########### for NLL evaluation
def train_nll(args, dataset_train, dataset_test, rnn, output,graph_validate_len,graph_test_len, max_iter = 1000):
    fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
    rnn.load_state_dict(torch.load(fname))
    fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
    output.load_state_dict(torch.load(fname))
    epoch = args.load_epoch
    print('model loaded!, epoch: {}'.format(args.load_epoch))
    fname_output = args.nll_save_path + args.note + '_' + args.graph_type + '.csv'
    with open(fname_output, 'w+') as f:
        f.write(str(graph_validate_len)+','+str(graph_test_len)+'\n')
        f.write('train,test\n')
        for iter in range(max_iter):
            if 'GraphRNN_MLP' in args.note:
                nll_train = train_mlp_forward_epoch(epoch, args, rnn, output, dataset_train)
                nll_test = train_mlp_forward_epoch(epoch, args, rnn, output, dataset_test)
            if 'GraphRNN_RNN' in args.note:
                nll_train = train_rnn_forward_epoch(epoch, args, rnn, output, dataset_train)
                nll_test = train_rnn_forward_epoch(epoch, args, rnn, output, dataset_test)
            print('train',nll_train,'test',nll_test)
            f.write(str(nll_train)+','+str(nll_test)+'\n')
    print('NLL evaluation done')