import csv
import os
import sys
import networkx as nx
import torch
import pandas as pd
import numpy as np
from itertools import islice
from rdkit import Chem
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA


def creat_data(datafile, cellfile):
    file2 = "data/" + cellfile + ".csv"
    cell_features = []
    with open(file2) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            cell_features.append(row)
    cell_features = np.array(cell_features)
    # print('cell_features', cell_features)

    compound_iso_smiles = []
    df = pd.read_csv("data/smiles.csv")
    compound_iso_smiles += list(df["smile"])
    compound_iso_smiles = set(compound_iso_smiles)
    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g

    # convert to PyTorch data format
    df = pd.read_csv("data/" + datafile + ".csv")
    drug1, drug2, cell, label = (
        list(df["drug1"]),
        list(df["drug2"]),
        list(df["cell"]),
        list(df["label"]),
    )
    drug1, drug2, cell, label = (
        np.asarray(drug1),
        np.asarray(drug2),
        np.asarray(cell),
        np.asarray(label),
    )
    # make data PyTorch Geometric ready

    print("Start creating data")
    TestbedDataset(
        root="data",
        dataset=datafile + "_drug1",
        xd=drug1,
        xt=cell,
        xt_featrue=cell_features,
        y=label,
        smile_graph=smile_graph,
    )
    TestbedDataset(
        root="data",
        dataset=datafile + "_drug2",
        xd=drug2,
        xt=cell,
        xt_featrue=cell_features,
        y=label,
        smile_graph=smile_graph,
    )
    print("Data created successfully")
    print("preparing ", datafile + " in pytorch format!")


class TestbedDataset(InMemoryDataset):
    def __init__(
        self,
        root="/tmp",
        dataset="_drug1",
        xd=None,
        xt=None,
        y=None,
        xt_featrue=None,
        transform=None,
        pre_transform=None,
        smile_graph=None,
    ):
        # root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print(
                "Pre-processed data found: {}, loading ...".format(
                    self.processed_paths[0]
                )
            )
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print(
                "Pre-processed data {} not found, doing pre-processing...".format(
                    self.processed_paths[0]
                )
            )
            self.process(xd, xt, xt_featrue, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + ".pt"]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def get_cell_feature(self, cellId, cell_features):
        for row in islice(cell_features, 0, None):
            if cellId in row[0]:
                return row[1:]
        return None

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, xt_featrue, y, smile_graph):
        assert len(xd) == len(xt) and len(xt) == len(y), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        print("number of data", data_len)
        for i in range(data_len):
            # print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(
                x=torch.Tensor(features),
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                y=torch.Tensor([labels]),
            )
            cell = self.get_cell_feature(target, xt_featrue)

            if cell is None:
                print("cell", cell)
                sys.exit()

            new_cell = []
            # print('cell_feature', cell_feature)
            for n in cell:
                new_cell.append(float(n))
            GCNData.cell = torch.FloatTensor([new_cell])
            GCNData.__setitem__("c_size", torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print("Graph construction done. Saving to file.")
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])


def atom_features(atom):
    return np.array(
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                "C",
                "N",
                "O",
                "S",
                "F",
                "Si",
                "P",
                "Cl",
                "Br",
                "Mg",
                "Na",
                "Ca",
                "Fe",
                "As",
                "Al",
                "I",
                "B",
                "V",
                "K",
                "Tl",
                "Yb",
                "Sb",
                "Sn",
                "Ag",
                "Pd",
                "Co",
                "Se",
                "Ti",
                "Zn",
                "H",
                "Li",
                "Ge",
                "Cu",
                "Au",
                "Ni",
                "Cd",
                "In",
                "Mn",
                "Zr",
                "Cr",
                "Pt",
                "Hg",
                "Pb",
                "Unknown",
            ],
        )
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(
            atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )
        + one_of_k_encoding_unk(
            atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )
        + [atom.GetIsAromatic()]
    )


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index