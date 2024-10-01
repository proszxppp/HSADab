import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
import joblib
import torch

solutes = pd.read_csv('test_exp', header=None, sep='\s+')

def extract_features(solutes):
    feature_list = []
    for smiles, T, dG in zip(solutes[0], solutes[1], solutes[2]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES: {smiles}")
            continue

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

        fp_Morgan = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024), fp_Morgan)

        fp_MACCS = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), fp_MACCS)

        features = np.concatenate((np.array([hbd, hba, rtb, psa, stereo, logp, mr, csp3, nrings, nrings_h, nrings_ar, nrings_ar_h, spiro, mw, atm_hetero, atm_heavy, atm_all]), 
                                   fp_Morgan, fp_MACCS, np.array([T, dG])), axis=0)

        feature_list.append(features)
    
    return np.array(feature_list)

sys_feature = extract_features(solutes)
ref_dG = sys_feature[:, -1]  
sys_feature = sys_feature[:, :-1]  

def load_and_predict(estimator_names, feature_data, model_path='model/'):
    predictions = {}
    for estimator in estimator_names:
        if 'MLP' in estimator:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            estimator_model = torch.jit.load(f'{model_path}{estimator}.pt', map_location=device)
            estimator_model.eval()
            
            pred_dG = estimator_model(torch.tensor(feature_data).float().to(device)).detach().cpu().numpy().squeeze()
        else:
            estimator_model = joblib.load(f'{model_path}{estimator}.joblib')
            pred_dG = estimator_model.predict(feature_data)

        predictions[estimator] = pred_dG
        np.savetxt(f'{estimator}.dat', pred_dG)

    return predictions

kn_estimators = ["KN.0", "KN.1", "KN.2"]
rf_estimators = ["RF.0", "RF.1", "RF.2"]
lgbm_estimators = ["LGBM.0", "LGBM.1", "LGBM.2"]
mlp_estimators = ["MLP.0", "MLP.1", "MLP.2"]

load_and_predict(kn_estimators, sys_feature)
load_and_predict(rf_estimators, sys_feature)
load_and_predict(lgbm_estimators, sys_feature)
load_and_predict(mlp_estimators, sys_feature)

