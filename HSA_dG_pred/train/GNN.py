import os
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from rdkit.Chem import rdmolops

def one_hot(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def bond_to_features(bond):
    bond_type = bond.GetBondType()
    bond_stereo = bond.GetStereo()
    bond_conjugation = bond.GetIsConjugated()
    bond_is_in_ring = bond.IsInRing()

    bond_type_one_hot = one_hot(bond_type, [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ])

    bond_stereo_one_hot = one_hot(bond_stereo, [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS
    ])

    return torch.tensor(bond_type_one_hot + bond_stereo_one_hot + [bond_conjugation, bond_is_in_ring], dtype=torch.float)


def smiles_to_graph(smiles, T):
    mol = Chem.MolFromSmiles(smiles)
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    atom_features = torch.tensor(atoms, dtype=torch.float).view(-1,1)

    AllChem.ComputeGasteigerCharges(mol)
    Chem.AssignStereochemistry(mol)

    acceptor_smarts_one = '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]'
    acceptor_smarts_two = "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]"
    donor_smarts_one = "[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]"
    donor_smarts_two = "[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]"

    hydrogen_donor_one = Chem.MolFromSmarts(donor_smarts_one)
    hydrogen_donor_two = Chem.MolFromSmarts(donor_smarts_two)
    hydrogen_acceptor_one = Chem.MolFromSmarts(acceptor_smarts_one)
    hydrogen_acceptor_two = Chem.MolFromSmarts(acceptor_smarts_two)

    hydrogen_donor_match_one = mol.GetSubstructMatches(hydrogen_donor_one)
    hydrogen_donor_match_two = mol.GetSubstructMatches(hydrogen_donor_two)
    hydrogen_donor_match = []
    hydrogen_donor_match.extend(hydrogen_donor_match_one)
    hydrogen_donor_match.extend(hydrogen_donor_match_two)
    hydrogen_donor_match = list(set(hydrogen_donor_match))

    hydrogen_acceptor_match_one = mol.GetSubstructMatches(hydrogen_acceptor_one)
    hydrogen_acceptor_match_two = mol.GetSubstructMatches(hydrogen_acceptor_two)
    hydrogen_acceptor_match = []
    hydrogen_acceptor_match.extend(hydrogen_acceptor_match_one)
    hydrogen_acceptor_match.extend(hydrogen_acceptor_match_two)
    hydrogen_acceptor_match = list(set(hydrogen_acceptor_match))

    ring = mol.GetRingInfo()

    m = []
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)

        o = []
        o += one_hot(atom.GetSymbol(), ['C', 'H', 'O', 'N', 'S', 'Cl', 'F', 'Br', 'P',
                                        'I'])
        o += [atom.GetDegree()]
        o += one_hot(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                               Chem.rdchem.HybridizationType.SP2,
                                               Chem.rdchem.HybridizationType.SP3,
                                               Chem.rdchem.HybridizationType.SP3D,
                                               Chem.rdchem.HybridizationType.SP3D2])
        o += [atom.GetImplicitValence()]
        o += [atom.GetIsAromatic()]
        o += [ring.IsAtomInRingOfSize(atom_idx, 3),
              ring.IsAtomInRingOfSize(atom_idx, 4),
              ring.IsAtomInRingOfSize(atom_idx, 5),
              ring.IsAtomInRingOfSize(atom_idx, 6),
              ring.IsAtomInRingOfSize(atom_idx, 7),
              ring.IsAtomInRingOfSize(atom_idx, 8)]

        o += [atom_idx in hydrogen_donor_match]
        o += [atom_idx in hydrogen_acceptor_match]
        o += [atom.GetFormalCharge()]

        temperature_features = torch.tensor([T], dtype=torch.float).view(-1)
        o_features = torch.tensor(o, dtype=torch.float).view(-1)
        merged_features = torch.cat((atom_features[atom_idx], o_features, temperature_features))
        m.append(merged_features)

#    node_features = torch.cat((temperature_features, atom_features), dim=1)
    node_features = torch.stack(m)

    edges = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])
    
        bond_features = bond_to_features(bond)
        edge_features.append(bond_features)
        edge_features.append(bond_features)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    if mol.GetNumAtoms() == 1:
        edge_attr = torch.tensor([], dtype=torch.float)
    else:
        edge_attr = torch.stack(edge_features)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
#    return Data(x=atom_features, edge_index=edge_index)

def load_data(filename):
    solutes = pd.read_csv(filename, header=None, sep='\s+')
    data_list = []
    
    for index, row in solutes.iterrows():
        smiles = row[0]
        T = row[1]
        dG = row[2]
        graph_data = smiles_to_graph(smiles, T)
        graph_data.dG = torch.tensor([dG], dtype=torch.float)
        data_list.append(graph_data)
    
    return data_list

data_list = load_data('../train')
torch.save(data_list, 'molecular_graphs.pt')
pd.DataFrame(data_list).to_csv("molecular_graphs.csv", index=False, header=False)


import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm, GATConv, GlobalAttention, global_add_pool, global_mean_pool

from scipy.stats import gmean, pearsonr, kendalltau
from matplotlib import pyplot as plt
import os
import pandas as pd


class GATNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_dim):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(in_dim, hidden_dim*4, heads=1, edge_dim=edge_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim*4)
        self.dropout1 = torch.nn.Dropout(p=dp)
#        self.conv2 = GATConv(hidden_dim*4, hidden_dim*4, heads=1, edge_dim=edge_dim)
#        self.bn2 = torch.nn.BatchNorm1d(hidden_dim*4)
#        self.dropout2 = torch.nn.Dropout(p=dp)
#        self.conv3 = GATConv(hidden_dim*4, hidden_dim*4, heads=1, edge_dim=edge_dim)
#        self.bn3 = torch.nn.BatchNorm1d(hidden_dim*4)

        self.conv2 = GATConv(hidden_dim*4, hidden_dim*2, heads=1, edge_dim=edge_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim*2)
        self.dropout2 = torch.nn.Dropout(p=dp)
        self.conv3 = GATConv(hidden_dim*2, hidden_dim*1, heads=1, edge_dim=edge_dim)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim*1)
        self.dropout3 = torch.nn.Dropout(p=dp)
        self.conv4 = GATConv(hidden_dim*1, hidden_dim*2, heads=1, edge_dim=edge_dim)
        self.bn4 = torch.nn.BatchNorm1d(hidden_dim*2)
        self.dropout4 = torch.nn.Dropout(p=dp)
        self.conv5 = GATConv(hidden_dim*2, hidden_dim*4, heads=1, edge_dim=edge_dim)
        self.bn5 = torch.nn.BatchNorm1d(hidden_dim*4)

        self.fc2 = torch.nn.Linear(hidden_dim*4, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 16)
        self.fc4 = torch.nn.Linear(16, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = F.leaky_relu(self.conv2(x, edge_index, edge_attr))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.conv3(x, edge_index, edge_attr))
        x = self.bn3(x)
        x = self.dropout3(x)
        x = F.leaky_relu(self.conv4(x, edge_index, edge_attr))
        x = self.bn4(x)
        x = self.dropout4(x)
        x = F.leaky_relu(self.conv5(x, edge_index, edge_attr))
        x = self.bn5(x)
        x = global_add_pool(x, batch)

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x.view(-1)

class GCNNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_dim):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim*4, edge_dim=edge_dim)
        self.bn1 = BatchNorm(hidden_dim*4)
        self.dropout1 = torch.nn.Dropout(p=dp)
        self.conv2 = GCNConv(hidden_dim*4, hidden_dim*2, edge_dim=edge_dim)
        self.bn2 = BatchNorm(hidden_dim*2)
        self.dropout2 = torch.nn.Dropout(p=dp)
        self.conv3 = GCNConv(hidden_dim*2, hidden_dim*1, edge_dim=edge_dim)
        self.bn3 = BatchNorm(hidden_dim*1)
        self.dropout3 = torch.nn.Dropout(p=dp)
        self.conv4 = GCNConv(hidden_dim*1, hidden_dim*2, edge_dim=edge_dim)
        self.bn4 = BatchNorm(hidden_dim*2)
        self.dropout4 = torch.nn.Dropout(p=dp)
        self.conv5 = GCNConv(hidden_dim*2, hidden_dim*4, edge_dim=edge_dim)
        self.bn5 = BatchNorm(hidden_dim*4)
        self.att = GlobalAttention(torch.nn.Linear(hidden_dim*4, 1))

        self.fc1 = torch.nn.Linear(hidden_dim*4, hidden_dim*2)
        self.fc2 = torch.nn.Linear(hidden_dim*2, hidden_dim*1)
        self.fc3 = torch.nn.Linear(hidden_dim*1, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = x.float()
        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = F.leaky_relu(self.conv2(x, edge_index, edge_attr))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.conv3(x, edge_index, edge_attr))
        x = self.bn3(x)
        x = self.dropout3(x)
        x = F.leaky_relu(self.conv4(x, edge_index, edge_attr))
        x = self.bn4(x)
        x = self.dropout4(x)
        x = F.leaky_relu(self.conv5(x, edge_index, edge_attr))
        x = self.bn5(x)
        x = self.att(x, batch)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)



data_list = torch.load('molecular_graphs.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for i, data in enumerate(data_list):
    a=0
#    data.x = torch.cat([data.x, fingerprint_repeated], dim=1)

cv_cycles = 10
hidden_dim = 128
train_epoch = 500
dp = 0.2
lr = 1e-3
wd = 0

dataset=2
RMSE_list = np.zeros((cv_cycles, dataset))
MAE_list = np.zeros((cv_cycles, dataset))
pearson_list = np.zeros((cv_cycles, dataset))
kendall_list = np.zeros((cv_cycles, dataset))

kf = KFold(n_splits=cv_cycles, shuffle=True)

for fold in range(0,cv_cycles):
    train_data, test_data = train_test_split(data_list, test_size=0.1)

#for fold, (train_index, test_index) in enumerate(kf.split(data_list)):
#    train_data = [data_list[i] for i in train_index]
#    test_data = [data_list[i] for i in test_index]

    train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=512, shuffle=False)

#    model = GCNNet(data.x.shape[1], hidden_dim, data.edge_attr.shape[1]).to(device)
    model = GATNet(data.x.shape[1], hidden_dim, data.edge_attr.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.9)
    criterion = torch.nn.MSELoss(reduction='mean')

    with tqdm(range(train_epoch)) as epochs:
        for epoch in epochs:
            model.train()
            train_loss = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.edge_attr, data.batch) #data.T, data.batch)
                loss = criterion(out, data.dG)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        
            train_loss /= len(train_loader)
            scheduler.step()

            model.eval()
            test_loss = 0
            pred_list = []
            true_list = []
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    out = model(data.x, data.edge_index, data.edge_attr, data.batch) #data.T, data.batch)
                    loss = criterion(out, data.dG)
                    test_loss += loss.item()
                    pred_list.append(out.cpu().numpy())
                    true_list.append(data.dG.cpu().numpy())
        
            test_loss /= len(test_loader)
        
            pred_list = np.concatenate(pred_list)
            true_list = np.concatenate(true_list)


            infos = {
                'Epoch': epoch,
                'TrainLoss': '{:.3f}'.format(np.sqrt(train_loss)),
                'TestLoss': '{:.3f}'.format(np.sqrt(test_loss)),
            }
            epochs.set_postfix(infos)

        RMSE_list[fold, 1] = np.sqrt(np.mean((pred_list - true_list) ** 2))
        MAE_list[fold, 1] = np.mean(np.abs(pred_list - true_list))
        pearson_list[fold, 1] = pearsonr(pred_list.flatten(), true_list.flatten())[0]
        kendall_list[fold, 1] = kendalltau(pred_list.flatten(), true_list.flatten())[0]

#        torch.save(model.state_dict(), f'GNN.{fold}.pt')


for metrics_list,metrics_name in zip([RMSE_list,MAE_list,pearson_list,kendall_list],['RMSE','MAE','Pearson_r','Kendall_Tau']):
    tmp_list = metrics_list
    tmp = np.stack( (np.mean(tmp_list, axis=0), np.std(tmp_list, axis=0)/np.sqrt(cv_cycles)*1.96), axis=1 )
    np.savetxt("metrics_{:s}.dat".format(metrics_name), (tmp[:,0]))
    np.savetxt("metrics_{:s}.dat.err".format(metrics_name), (tmp[:,1]))

