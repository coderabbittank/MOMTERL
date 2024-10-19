import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention

import rdkit
import sys

import numpy as np
from gnn_model import GNN, GNNDecoder

from copy import deepcopy

sys.path.append('./util/')

from datautils import DataLoaderMaskingPred
from loader import MoleculeDataset
import pandas as pd

criterion = nn.CrossEntropyLoss()


def gen_ran_output(data, model, args, device):
    vice_model = deepcopy(model)
    for (name, vice_param), (name, param) in zip(vice_model.named_parameters(), model.named_parameters()):
        if name.split('.')[0] == 'projection_head':
            vice_param.data = param.data
        else:
            vice_param.data = param.data + args.eta * torch.normal(0,
                                                                   torch.ones_like(param.data) * param.data.std()).to(
                device)
    z2 = vice_model.forward_cl2(data.x, data.edge_index, data.edge_attr)
    return z2


lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


def loss_cl(x1, x2):
    T = 0.1
    batch_size, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss


class graphcl(nn.Module):
    def __init__(self, gnn):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))

    def forward_cl(self, x, edge_index, edge_attr, batch):
        x_node = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x_node, batch)
        x = self.projection_head(x)
        return x_node, x

    def forward_cl2(self, x, edge_index, edge_attr):
        x_node = self.gnn(x, edge_index, edge_attr)
        return x_node

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss


def group_node_rep(node_rep, batch_size, num_part):
    group = []
    motif_group = []
    super_group = []

    count = 0
    for i in range(batch_size):
        num_atom = num_part[i * 3]
        num_motif = num_part[i * 3 + 1]
        num_all = num_atom + num_motif + 1
        group.append(node_rep[count:count + num_atom])
        motif_group.append(node_rep[count + num_atom:count + num_atom + num_motif])
        super_group.append(node_rep[count + num_all - 1])
        count += num_all
    return group, motif_group, super_group


def train(model_list, loader1, optimizer, device, args):
    model, atom_pred_decoder_model, projection_decoder = model_list

    model.train()
    atom_pred_decoder_model.train()
    projection_decoder.train()

    train_loss_accum = 0
    for step, batch in enumerate(loader1):

        batch = batch.to(device)

        
        node_rep, _ = model.forward_cl(batch.x_nosuper, batch.edge_index_nosuper, batch.edge_attr_nosuper, batch.batch1)

        with torch.no_grad():
            masked_node_indices_atom = batch.masked_atom_indices_atom
            label_atom = batch.node_attr_label

        if args.decoder == 'linear':
            pred_atom = atom_pred_decoder_model(node_rep[masked_node_indices_atom])
            pred_node = pred_atom
        else:
            pred_atom = atom_pred_decoder_model(node_rep, batch)
            pred_node = pred_atom[masked_node_indices_atom]

        node_loss_type = criterion(pred_node, label_atom)
        loss_mask = node_loss_type

        loss = loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())

    return train_loss_accum / (step + 1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=1,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default='zinc_standard_agent',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--output_model_file', type=str, default='./saved_model/MOMCenter_0.3.pth',
                        help='filename to output the pre-trained model')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')
    parser.add_argument("--hidden_size", type=int, default=300, help='hidden size')
    parser.add_argument('--motif_to_mask_percent', type=float, default='0.30')
    parser.add_argument('--node_to_mask_percent', type=float, default='1')
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--decoder', type=str, default='linear')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    dataset1 = MoleculeDataset('./dataset/' + args.dataset, dataset=args.dataset)
    smiles_list1 = pd.read_csv('./dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()

    loader1 = DataLoaderMaskingPred(dataset1, smiles_list1, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, motif_mask_rate=args.motif_to_mask_percent,
                                    intermotif_mask_rate=args.node_to_mask_percent, mask_edge=0)

    NUM_NODE_ATTR = 119

    model = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(
        device)
    model = graphcl(model).to(device)

    projection_decoder = nn.Sequential(nn.Linear(args.emb_dim, args.emb_dim), nn.ReLU(inplace=True),
                                       nn.Linear(args.emb_dim, args.emb_dim)).to(device)

    if args.decoder == 'linear':
        atom_pred_decoder_model = torch.nn.Linear(args.emb_dim, NUM_NODE_ATTR).to(device)
    else:
        atom_pred_decoder_model = GNNDecoder(args.emb_dim, NUM_NODE_ATTR, JK=args.JK, gnn_type=args.gnn_type).to(device)

    model_list = [model, atom_pred_decoder_model, projection_decoder]
    optimizer = optim.Adam([{"params": model.parameters()}, {"params": atom_pred_decoder_model.parameters()},
                            {"params": projection_decoder.parameters()}], lr=args.lr, weight_decay=args.decay)

    for epoch in range(1, args.epochs + 1):
        print('====epoch', epoch)
        train_loss = train(model_list, loader1, optimizer, device, args)
        print(train_loss)
        if not args.output_model_file == "":
            torch.save(model.gnn.state_dict(), args.output_model_file)


if __name__ == "__main__":
    main()
