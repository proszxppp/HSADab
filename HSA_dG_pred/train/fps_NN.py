
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem

solutes=pd.read_csv('../train',header=None,sep='\s+')

feature = []

for smiles,T,dG in zip(solutes[0],solutes[1],solutes[2]):
    mol = Chem.MolFromSmiles(smiles)

    form = rdMolDescriptors.CalcMolFormula(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)

    rtb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    psa = rdMolDescriptors.CalcTPSA(mol)
    stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)

    logp, mr = rdMolDescriptors.CalcCrippenDescriptors(mol)
    csp3 = rdMolDescriptors.CalcFractionCSP3(mol)

    nrings = rdMolDescriptors.CalcNumRings(mol)
    nrings_h = rdMolDescriptors.CalcNumHeterocycles(mol)
    nrings_ar = rdMolDescriptors.CalcNumAromaticRings(mol)
    nrings_ar_h = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)

    spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)

    mw = rdMolDescriptors._CalcMolWt(mol)

    atm_hetero = rdMolDescriptors.CalcNumHeteroatoms(mol)
    atm_heavy = mol.GetNumHeavyAtoms()
    atm_all = mol.GetNumAtoms()

# sometimes C is recognized as sp3 carbon and thus csp3=1.0 and more hydrogen added

# 1024-bit Morgan fingerprint with a radius parameter 2 
    fp_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    fp_MorganBits = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp_vec, fp_MorganBits)

# MACCS fingerprint
    fp_vec = MACCSkeys.GenMACCSKeys(mol)
    fp_MACCS = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp_vec, fp_MACCS)

    feature.append( np.concatenate((np.array([hbd, hba, rtb, psa, stereo, logp, mr, csp3, nrings, nrings_h, nrings_ar, nrings_ar_h, spiro, mw, atm_hetero, atm_heavy, atm_all]), fp_MorganBits, fp_MACCS, np.array([T,dG])), axis=0) )

pd.DataFrame(feature).to_csv("exp.dat", index=False, header=False)



import numpy as np

from sklearn.model_selection import KFold, train_test_split
import os
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.init as init

from scipy.stats import gmean, pearsonr, kendalltau

cv_cycles=20
dataset=2

method_name='MLP'
RMSE_list = np.zeros((cv_cycles, dataset))   # ML_model, K-fold, n_dataset 3 = train, test, new system
MAE_list = np.zeros((cv_cycles, dataset))   # 3 = train, test, new system
pearson_list = np.zeros((cv_cycles, dataset))   # 3 = train, test, new system
kendall_list = np.zeros((cv_cycles, dataset))   # 3 = train, test, new system

train_epoch = 300
hidden_dim = 200    
dp = 0.2
lr = 1e-2  
wd = 0  
K_fold = 20
folds = KFold(n_splits=K_fold, shuffle=True)
multistep_lr = [7*_ for _ in range(1, 93)]  #+ [600*_ for _ in range(9, 33)]

device = "cuda" if torch.cuda.is_available() else "cpu"

class Regressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, dp: float, out_dim=1):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nets = nn.Sequential(
                                  nn.Linear(in_dim, hidden_dim),
                                  nn.BatchNorm1d(hidden_dim),
                                  nn.LeakyReLU(),
                                  nn.Dropout(p=dp),
                                  nn.Linear(hidden_dim, hidden_dim),
                                  nn.BatchNorm1d(hidden_dim),
                                  nn.LeakyReLU(),
                                  nn.Dropout(p=dp),
                                  nn.Linear(hidden_dim, hidden_dim),
                                  nn.BatchNorm1d(hidden_dim),
                                  nn.LeakyReLU(),
                                  nn.Dropout(p=dp),
                                  nn.Linear(hidden_dim, hidden_dim),
                                  nn.BatchNorm1d(hidden_dim),
                                  nn.LeakyReLU(),
                                  nn.Dropout(p=dp),
                                  nn.Linear(hidden_dim, hidden_dim),
                                  nn.BatchNorm1d(hidden_dim),
                                  nn.LeakyReLU(),
                                  nn.Dropout(p=dp),
                                  nn.Linear(hidden_dim, out_dim),
                                  ).to(self.device)

    def forward(self, x):
        out = self.nets(x)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                   init.constant_(m.bias, 0)


if __name__ == '__main__':

    data = pd.read_csv('exp.dat', header=None)

    dG=data.pop(data.shape[1]-1)
    feature=data

    in_dim = feature.shape[1]
    out_dim = 1

    trail = 0
    train_result_loss = 0
    test_result_loss = 0
#    for train_idx, test_idx in folds.split(feature, dG):       # K-fold CV
#        train_X, train_y = np.array(feature.iloc[train_idx, :]), np.array(dG[train_idx])
#        test_X, test_y = np.array(feature.iloc[test_idx, :]), np.array(dG[test_idx])
    for i in range(0,K_fold):   # MCCV
        train_X, test_X, train_y, test_y = train_test_split(feature,dG, test_size=0.1)    #(dataset)
        print("Trail {}...".format(trail))
        train_X, train_y, test_X, test_y = \
            torch.tensor(np.array(train_X)).float(), torch.tensor(np.array(train_y)).float().unsqueeze(1), \
            torch.tensor(np.array(test_X)).float(), torch.tensor(np.array(test_y)).float().unsqueeze(1),
#            torch.tensor(np.array(train_X)).float().to(device), torch.tensor(np.array(train_y)).float().unsqueeze(1).to(device), \
#            torch.tensor(np.array(test_X)).float().to(device), torch.tensor(np.array(test_y)).float().unsqueeze(1).to(device),
#            torch.tensor(train_X).float().to(device), torch.tensor(train_y).float().unsqueeze(1).to(device), \
#            torch.tensor(test_X).float().to(device), torch.tensor(test_y).float().unsqueeze(1).to(device),

        model = Regressor(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, dp=dp)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, multistep_lr, gamma=0.9, last_epoch=-1)
        loss_R = nn.MSELoss(reduction='mean')

        epoch_train_loss = []
        epoch_test_loss = []
        train_data=TensorDataset(train_X,train_y)
        test_data=TensorDataset(test_X,test_y)
        train_loader = DataLoader(train_data, batch_size=1024, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=1024, shuffle=False, num_workers=4)

        with tqdm(range(train_epoch)) as epochs:
            for epoch in epochs:
                model.train()
                loss_all = 0
                train_pred_all = []
                for i,(inputs,labels) in enumerate(train_loader):
                    inputs,labels = inputs.to(device), labels.to(device)
                    train_pred = model(inputs)
                    train_pred_all.append(train_pred.detach().cpu().numpy().squeeze())
                    loss = loss_R(train_pred, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    loss_all += loss
                    optimizer.step()

                scheduler.step()
                train_loss=loss_all/(i+1)
                epoch_train_loss.append(float(torch.sqrt(train_loss)))

                model.eval()
                loss_all = 0
                test_pred_all = []
                for i,(inputs,labels) in enumerate(test_loader):
                    inputs,labels = inputs.to(device), labels.to(device)
                    test_pred = model(inputs)
                    test_pred_all.append(test_pred.detach().cpu().numpy().squeeze())
                    loss = loss_R(test_pred, labels)
                    loss_all += loss

                test_loss=loss_all/(i+1)
                epoch_test_loss.append(float(torch.sqrt(test_loss)))

                infos = {
                    'Epoch': epoch,
                    'TrainLoss': '{:.3f}'.format(torch.sqrt(train_loss)),
                    'TestLoss': '{:.3f}'.format(torch.sqrt(test_loss)),
                }
                epochs.set_postfix(infos)

        ref_train=train_y.detach().cpu().numpy().squeeze() 
        pred_train=np.concatenate(train_pred_all)      
        ref_test=test_y.detach().cpu().numpy().squeeze() 
        pred_test=np.concatenate(test_pred_all)       

        xlen=22
        ylen=22
        oo=plt.figure(1, figsize=(xlen,ylen))
        ax=oo.gca()
        plt.plot(np.arange(-200,200), np.arange(-200,200), color="cyan", label="y=x")
        for n_dataset, (ref,pred) in enumerate(zip([ref_train,ref_test],[pred_train,pred_test])):
#            np.savetxt("ref_pred_{:s}_{:d}_{:d}.dat".format(method_name,trail,n_dataset).replace('_0.dat','_train.dat').replace('_1.dat','_test.dat').replace('_2.dat','_CB7.dat'), (ref, pred))
            RMSE_list[trail,n_dataset] = np.sqrt(np.mean( (ref - pred)**2 ))
            MAE_list[trail,n_dataset] = np.mean(abs( ref - pred ))
            pearson_list[trail,n_dataset] = pearsonr(ref, pred)[0]
            kendall_list[trail,n_dataset] = kendalltau(ref, pred)[0]
            plt.scatter(ref, pred, label="dataset{:d} RMSE {:.1f} $\AA$ Pearson r {:.2f} Kendall {:.2f}".format(n_dataset, RMSE_list[trail,n_dataset], pearson_list[trail,n_dataset], kendall_list[trail,n_dataset]).replace('dataset0','Training').replace('dataset1','Testing').replace('dataset2','CB7'))
            print(method_name,RMSE_list[trail,n_dataset], pearson_list[trail,n_dataset], kendall_list[trail,n_dataset])

        plt.title("Performance {:s} Trail {:d}th ".format(method_name,trail),fontsize=35)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.legend(loc='best',fontsize=30)
        ax.set_xlim(-100,0)
        ax.set_ylim(-100,0)
        plt.xlabel("Experimental Reference ($\AA$)",fontsize=40)
        plt.ylabel("Predicted Results ($\AA$)",fontsize=40)
#        plt.savefig("exp_theo_{:s}_{:d}.png".format(method_name,trail),dpi=250)
        plt.close()

        train_result_loss += epoch_train_loss[-1]
        test_result_loss += epoch_test_loss[-1]
#        draw_train_curve(epoch_train_loss, epoch_test_loss, trail)
        model_scripted = torch.jit.script(model)
#        model_scripted.save('MLP.{:d}.pt'.format(trail))
        trail += 1

    for metrics_list,metrics_name in zip([RMSE_list,MAE_list,pearson_list,kendall_list],['RMSE','MAE','Pearson_r','Kendall_Tau']):
        tmp_list = metrics_list
        tmp = np.stack( (np.mean(tmp_list, axis=0), np.std(tmp_list, axis=0)/np.sqrt(cv_cycles)*1.96), axis=1 )
        np.savetxt("metrics_{:s}.dat".format(metrics_name), (tmp[:,0]))
        np.savetxt("metrics_{:s}.dat.err".format(metrics_name), (tmp[:,1]))

