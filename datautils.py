from itertools import count
from re import S
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import math
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from torch_geometric.data import Batch
from torch_geometric.data import Data
from loader import motif_decomp
import copy

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ],
    'possible_bond_inring': [None, False, True]
}


class DataLoaderMasking(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderMasking, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchMasking.from_data_list(data_list),
            **kwargs)


class DataLoaderMaskingPred(torch.utils.data.DataLoader):

    def __init__(self, dataset, smiles_list, batch_size=1, shuffle=True, motif_mask_rate=0.25, intermotif_mask_rate=1,
                 mask_edge=0.0, **kwargs):
        self._transform = MaskAtom(num_atom_type=119, num_edge_type=5, mask_rate=motif_mask_rate,
                                   inter_mask_rate=intermotif_mask_rate, mask_edge=mask_edge)
        self.smiles_list = smiles_list
        super(DataLoaderMaskingPred, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collate_fn,
            **kwargs)

    def collate_fn(self, batches):
        batchs = [self._transform(x, self.smiles_list[x.id]) for x in batches]
        return BatchMasking.from_data_list(batchs)


class DataLoaderMaskingPred2(torch.utils.data.DataLoader):

    def __init__(self, dataset, smiles_list, batch_size=1, shuffle=True, motif_mask_rate=0.25, intermotif_mask_rate=1,
                 mask_edge=0.0, **kwargs):
        self._transform = MaskAtom2(num_atom_type=119, num_edge_type=5, mask_rate=motif_mask_rate,
                                    inter_mask_rate=intermotif_mask_rate, mask_edge=mask_edge)
        self.smiles_list = smiles_list
        super(DataLoaderMaskingPred2, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collate_fn,
            **kwargs)

    def collate_fn(self, batches):
        batchs = [self._transform(x) for x in batches]
        return BatchMasking2.from_data_list(batchs)


class BatchMasking(Data):

    def __init__(self, batch=None, **kwargs):
        super(BatchMasking, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchMasking()

        for key in keys:
            batch[key] = []
        batch.batch1 = []
        batch.batch2 = []
        batch.batchmotifs = []
        batch.batch_len = len(data_list)
        cumsum_node_nosuper = 0
        cumsum_edge_nosuper = 0
        cumsum_node = 0
        for i, data in enumerate(data_list):

            num_nodes_nosuper = data.x_nosuper.size()[0]
            num_nodes = data.x.size()[0]
            batch.batch1.append(torch.full((num_nodes_nosuper,), i, dtype=torch.long))
            batch.batch2.append(torch.full((num_nodes,), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index_nosuper', 'masked_atom_indices','masked_atom_indices_atom']:
                    item = item + cumsum_node_nosuper
                elif key == 'connected_edge_indices':
                    item = item + cumsum_edge_nosuper
                elif key == 'edge_index':
                    item = item + cumsum_node
                batch[key].append(item)

            cumsum_node_nosuper += num_nodes_nosuper
            cumsum_edge_nosuper += data.edge_index_nosuper.shape[1]
            cumsum_node += num_nodes

        for key in keys:
            if key not in ['batchmotifs']:
                batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
        batch.batch1 = torch.cat(batch.batch1, dim=-1)
        batch.batch2 = torch.cat(batch.batch2, dim=-1)

        return batch.contiguous()

    def cumsum(self, key, item):
        return key in ['edge_index_nosuper', 'face', 'masked_atom_indices', 'connected_edge_indices', 'edge_index','masked_atom_indices_atom']

    @property
    def num_graphs(self):
        return self.batch[-1].item() + 1


class BatchMasking2(Data):

    def __init__(self, batch=None, **kwargs):
        super(BatchMasking2, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchMasking()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0
        cumsum_edge = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'masked_atom_indices']:
                    item = item + cumsum_node
                elif key == 'connected_edge_indices':
                    item = item + cumsum_edge
                batch[key].append(item)

            cumsum_node += num_nodes
            cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    def cumsum(self, key, item):
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices']

    @property
    def num_graphs(self):
        return self.batch[-1].item() + 1


class MaskAtom:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, inter_mask_rate, mask_edge):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms/motifs to be masked
        :param inter_mask_rate: % of atoms within motif to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge

        self.num_chirality_tag = 3
        self.num_bond_direction = 3

        self.offset = 0

        self.inter_mask_rate = inter_mask_rate

    def __call__(self, data, smiles, masked_atom_indices=None):
        # Operating on a single mol
        mol = Chem.MolFromSmiles(smiles)
        motifs = motif_decomp(mol)

        num_atoms = data.x_nosuper.size()[0]
        sample_size = int(num_atoms * self.mask_rate + 1)
        valid_motifs = []
        if len(motifs) != 1:
            for motif in motifs:
                valid_motifs.append(motif)
        copy_motifs = copy.deepcopy(valid_motifs)
        # print(len(copy_motifs))

        # masked_atom_indices = None
        masked_atom_indices = []

        # Select motifs 
        while len(masked_atom_indices) < sample_size:

            if len(valid_motifs) < 1:
                index_list = random.sample(range(num_atoms), sample_size)
                for index in index_list:
                    if index not in masked_atom_indices:
                        masked_atom_indices.append(index)
            else:
                candidate = valid_motifs[random.sample(range(0, len(valid_motifs)), 1)[0]]
                valid_motifs.remove(candidate)
                for atom_idx in candidate:
                    for i, edge in enumerate(data.edge_index_nosuper[0]):
                        if atom_idx == edge:
                            for motif in valid_motifs:
                                if data.edge_index_nosuper[1][i].item() in motif:
                                    valid_motifs.remove(motif)

                if len(masked_atom_indices) + len(candidate) > sample_size + 0.1 * num_atoms:
                    continue

                for index in candidate:
                    masked_atom_indices.append(index)

        if masked_atom_indices == None:
            masked_atom_indices = []

            # randon masking
            num_atoms = data.x_nosuper.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)

            # 随机集中掩码原子(连续的一串索引)
            # num_atoms = data.x_nosuper.size()[0]
            # sample_size = int(num_atoms * self.mask_rate + 1)
            # start_index = random.randint(0, num_atoms - sample_size)  # 随机选择起始索引
            # masked_atom_indices = list(range(start_index, start_index + sample_size))  # 连续的一串索引

            # #随机motif集中掩码  
            # while len(masked_atom_indices) < sample_size:
            #     if(len(valid_motifs)<1 and len(masked_atom_indices)==0):
            #         if(len(copy_motifs)<=1):
            #             num_atoms = data.x_nosuper.size()[0]
            #             sample_size = int(num_atoms * self.mask_rate + 1)
            #             masked_atom_indices = random.sample(range(num_atoms), sample_size)
            #         else:
            #             candidate = copy_motifs[random.sample(range(0, len(copy_motifs)), 1)[0]]
            #             copy_motifs.remove(candidate)
            #             for index in candidate:
            #                 if(len(masked_atom_indices)<sample_size):
            #                     masked_atom_indices.append(index)

            #     elif(len(valid_motifs)<1 and len(masked_atom_indices)>0):
            #         for atom_idx in masked_atom_indices:
            #             for i, edge in enumerate(data.edge_index_nosuper[0]):
            #                 if atom_idx == edge:
            #                     connected_atom = data.edge_index_nosuper[1][i].item()
            #                     if connected_atom not in masked_atom_indices and len(masked_atom_indices)<sample_size:
            #                         masked_atom_indices.append(connected_atom)

            #     elif(len(valid_motifs)>=1):
            #         candidate = valid_motifs[random.sample(range(0, len(valid_motifs)), 1)[0]]
            #         valid_motifs.remove(candidate)
            #         if(len(masked_atom_indices)+len(candidate)<=sample_size+0.1*num_atoms):
            #             for index in candidate:
            #                 if index not in masked_atom_indices:
            #                     masked_atom_indices.append(index)
            #             for atom_idx in candidate:
            #                 for i, edge in enumerate(data.edge_index_nosuper[0]):
            #                     if atom_idx == edge:
            #                         for motif in valid_motifs:
            #                             connected_atom = data.edge_index_nosuper[1][i].item()
            #                             if connected_atom in motif and connected_atom not in masked_atom_indices and len(masked_atom_indices)<sample_size:
            #                                 masked_atom_indices.append(connected_atom)
            #         else:
            #             continue

        # l = math.ceil(len(masked_atom_indices) * self.inter_mask_rate)
        # masked_atom_indices_atom = random.sample(masked_atom_indices, l)
        # masked_atom_indices_chi = random.sample(masked_atom_indices, l)

        # node masking

        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x_nosuper[atom_idx].view(1, -1))
        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.node_attr_label = data.mask_node_label[:, 0]
        data.masked_atom_indices_atom = torch.tensor(masked_atom_indices)

        for atom_idx in masked_atom_indices:
            data.x_nosuper[atom_idx] = torch.tensor([self.num_atom_type, data.x_nosuper[atom_idx][1]])

        if self.mask_edge:

            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(data.edge_index_nosuper.cpu().numpy().T):
                for atom_idx in masked_atom_indices:
                    if atom_idx in set((u, v)) and \
                            bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::2]:  # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(
                        data.edge_attr_nosuper[bond_idx].view(1, -1))

                data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                # modify the original bond features of the bonds connected to the mask atoms
                for bond_idx in connected_edge_indices:
                    data.edge_attr_nosuper[bond_idx] = torch.tensor(
                        [self.num_edge_type, 0])

                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2])
            else:
                data.mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices).to(torch.int64)

            edge_type = F.one_hot(data.mask_edge_label[:, 0], num_classes=self.num_edge_type).float()
            bond_direction = F.one_hot(data.mask_edge_label[:, 1], num_classes=self.num_bond_direction).float()
            data.edge_attr_label = torch.cat((edge_type, bond_direction), dim=1)
            # data.edge_attr_label = edge_type

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)


class MaskAtom2:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, inter_mask_rate, mask_edge):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms/motifs to be masked
        :param inter_mask_rate: % of atoms within motif to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge

        self.num_chirality_tag = 3
        self.num_bond_direction = 3

        self.offset = 0

        self.inter_mask_rate = inter_mask_rate

    def __call__(self, data, masked_atom_indices=None):
        """
        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """

        # mol = Chem.MolFromSmiles(smiles)

        num_atoms = data.x.size()[0]
        sample_size = int(num_atoms * self.mask_rate + 1)

        masked_atom_indices = []

        masked_atom_indices = random.sample(range(num_atoms), sample_size)

        # create mask node label by copying atom feature of mask atom
        # node masking

        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))

        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices_atom = torch.tensor(masked_atom_indices)
        # print(data.mask_node_label[:,0])
        data.node_attr_label = data.mask_node_label[:, 0]

        for atom_idx in masked_atom_indices:
            data.x[atom_idx] = torch.tensor([self.num_atom_type, data.x[atom_idx][1]])

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)


def molgraph_to_graph_data(batch):
    graph_data_batch = []
    for mol in batch:
        data = Data(x=mol.x, edge_index=mol.edge_index, edge_attr=mol.edge_attr, num_part=mol.num_part)
        graph_data_batch.append(data)
    new_batch = Batch().from_data_list(graph_data_batch)
    return new_batch
