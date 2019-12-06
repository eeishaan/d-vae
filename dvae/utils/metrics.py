import torch


def reconstruction_accuracy(true_encodings, true_deps, pred_encodings, pred_deps):
    """
    true_encodings: Tensors containing  true node encoding of graph.
                    Shape: (batch_size, node_types)
    pred_encodings: Tensors containing  predicted node encoding of graph
                    (batch_size, sample_num, node_types). sample_num = 
                    decoded graphs for the same input graph.
    true_deps: Tensor containing true dependency graph.
                Shape: (batch_size, seq_len, seq_len)
    pred_deps: Tensor containing predicted dependency graph.
                Shape: (batch_size, sample_num, seq_len, seq_len)
    """
    # first check if node order is correct
    batch_size, sample_num, _ = pred_encodings.shape
    node_equality = (true_encodings.unsqueeze(1) == pred_encodings)
    node_equality = node_equality.all(dim=-1)

    # check if correct nodes have correct edges
    node_equality.unsqueeze_(-1)
    true_deps = true_deps.view(batch_size, -1).unsqueeze(1)
    pred_deps = pred_deps.view(batch_size, sample_num, -1)
    edge_equality = (true_deps == pred_deps) * node_equality
    edge_equality = edge_equality.all(dim=-1)

    correct_graphs = edge_equality.sum()
    return correct_graphs / (batch_size * sample_num)
