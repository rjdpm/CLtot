#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu June 20, 2025 at 15:56:00 AM IST
"""
import re
import base64
from transformers import AutoTokenizer, utils
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')
from rdkit import Chem
import sys, os

from mol2vec.features import mol2alt_sentence, sentences2vec
from gensim.models import word2vec
from typing import List
from concurrent.futures import ProcessPoolExecutor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from inference_utils_cltot.data_utils_smiles_cltot import compute_descriptors_from_name, plot_smiles_grid
from inference_utils_cltot.graph_utils_smiles_cltot import Multirelational_GraphDataset, GraphData_from_pickle
from inference_utils_cltot.data_utils_cltot import load_from_pickle, fix_seed, load_json
from inference_utils_cltot.train_module_cltot import Trainer_functions
from inference_utils_cltot.models_module_cltot import ChemBERTaRegressor, MRGCN, SMILESLinearNet, FusionNet


fix_seed(np_seed=20, torch_seed=30, random_seed=10)

__all__ = ['PredMol2Vec',
           'PredDescriptors',
           'PredGCN',
           'PredChemBert',
           'PredFusionNet',
           
           'predict_cltot_mol2vec',
           'predict_cltot_feat',
           'predict_cltot_gcn',
           'predict_cltot_bert',
           'predict_cltot_FusionNet',
           
           'predict_cltot',
           ]

path_1 = os.path.dirname(os.path.abspath(__file__))
GLOBAL_PATH_INIT = f'{path_1}/Best_models'

def _round(x, decimal):
    if isinstance(x, (str, float, int)):
        return float(f"{x:.{decimal}f}")
    elif isinstance(x, (list, np.ndarray, tuple, torch.Tensor)):
        return [float(f"{_x:.{decimal}f}") for _x in x]

def _format_prediction(pred):

    pred = np.asarray(pred).reshape(-1)
    if len(pred) == 1:
        return _round(pred[0].item(), 4)
   
    return _round(pred, 4)

class PredMol2Vec:
    
    mol2vec_model_path  = f'{GLOBAL_PATH_INIT}/Mol2Vec/repr_model/model_300dim.pkl'
    rf_model_path       = f'{GLOBAL_PATH_INIT}/Mol2Vec/regressors/mol2vec_rf_best_model.pkl'
    svm_model_path      = f'{GLOBAL_PATH_INIT}/Mol2Vec/regressors/mol2vec_svm_best_model.pkl'
    xgb_model_path      = f'{GLOBAL_PATH_INIT}/Mol2Vec/regressors/mol2vec_xgb_best_model.pkl'
    nn_model_path       = f'{GLOBAL_PATH_INIT}/Mol2Vec/regressors/mol2vec_nn_best_model.pt'
    scaler_path_feats   = f'{GLOBAL_PATH_INIT}/Mol2Vec/regressors/StandardScaler_feats.pkl'
    scaler_path_labels  = f'{GLOBAL_PATH_INIT}/Mol2Vec/regressors/StandardScaler_labels.pkl'

    def __init__(self,
                 smiles: str,
                 regressor: str='None',
                 mol2vec_model_path: str = mol2vec_model_path,
                 ):
        
        self.smiles = smiles
        self.regressor = regressor.lower().strip()
        # Load Mol2Vec model
        self.mol2vec_model_path = mol2vec_model_path
        self.mol2vec_model = word2vec.Word2Vec.load(self.mol2vec_model_path)
        self.scaler_feats = load_from_pickle(self.scaler_path_feats)
        self.scaler_labels = load_from_pickle(self.scaler_path_labels)
        if isinstance(self.smiles, str):
            self.preprocessing()
        if isinstance(self.smiles, (list, tuple)):
            self.preprocessing_list()
        
    def preprocessing(self):   
         
        # Convert SMILES → RDKit mol
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {self.smiles}")
        # Convert mol → sentence → vector
        sentence = mol2alt_sentence(mol, 1)
        vec = sentences2vec([sentence], self.mol2vec_model, unseen='UNK')
        # Scale representation
        self.X = self.scaler_feats.transform(vec)
        
        return self.X
    
    def preprocessing_list(self):   
         
        # Convert SMILES → RDKit mol
        mol_sentances = []
        imvalid_smiles = []
        for smi in self.smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mol_sentances.append(mol2alt_sentence(mol, 1))
            else:
                imvalid_smiles.append(smi)
        vec = sentences2vec(mol_sentances, self.mol2vec_model, unseen='UNK')
        # Scale representation
        self.X = self.scaler_feats.transform(vec)
        
        return self.X
        
    def predict_mol2vec_nonNN(self, model_path):
        
        model = load_from_pickle(model_path)
        pred_scaled = model.predict(self.X).reshape(-1, 1)
        pred = self.scaler_labels.inverse_transform(pred_scaled)
        pred = _format_prediction(pred)
        
        return pred

    def predict_mol2vec_nn(self, model_path=nn_model_path):
        nn_model = SMILESLinearNet(list_dims=[300, 256, 128, 1],
                                   dropout=0.4,
                                   act_func='relu',
                                   norm_type='layer')
        
        Trainer_functions.load_test_network(network=nn_model, temp_network_path=model_path, verbose=False)
        nn_model.eval()

        with torch.no_grad():
            pred_scaled = nn_model(torch.from_numpy(self.X).float()).reshape(-1, 1)
        
        pred = self.scaler_labels.inverse_transform(pred_scaled)
        pred = _format_prediction(pred)
        
        return pred

    # -------------------- Dispatcher -------------------- #

    def predict(self):
        if self.regressor == 'nn':
            return self.predict_mol2vec_nn(self.nn_model_path)
        elif self.regressor == 'xgb':
            return self.predict_mol2vec_nonNN(self.xgb_model_path)
        elif self.regressor == 'rf':
            return self.predict_mol2vec_nonNN(self.rf_model_path)
        elif self.regressor == 'svm':
            return self.predict_mol2vec_nonNN(self.svm_model_path)
        elif self.regressor == 'all':
            y_pred_nn = self.predict_mol2vec_nn(self.nn_model_path)
            y_pred_rf = self.predict_mol2vec_nonNN(self.rf_model_path)
            y_pred_svm = self.predict_mol2vec_nonNN(self.svm_model_path)
            y_pred_xgb = self.predict_mol2vec_nonNN(self.xgb_model_path)
            return {'NN':y_pred_nn, 'RF':y_pred_rf, 'SVM':y_pred_svm, 'XGB':y_pred_xgb}
        else:
            raise ValueError(
                f"Unknown regressor: {self.regressor}. "
                f"Choose from 'nn', 'xgb', 'rf', 'svm'."
            )
def predict_cltot_mol2vec(smiles: str, regressor: str):
    
    obj = PredMol2Vec(smiles, regressor)
    prediction = obj.predict()
    
    return prediction


class PredDescriptors:
    
    features_path       = f'{GLOBAL_PATH_INIT}/Descriptors/repr_model/optimal_features.json'
    scaler_path_feats   = f'{GLOBAL_PATH_INIT}/Descriptors/regressors/StandardScaler_feats.pkl'
    scaler_path_labels  = f'{GLOBAL_PATH_INIT}/Descriptors/regressors/StandardScaler_labels.pkl'
    rf_model_path       = f'{GLOBAL_PATH_INIT}/Descriptors/regressors/feat_rf_best_model.pkl'
    svm_model_path      = f'{GLOBAL_PATH_INIT}/Descriptors/regressors/feat_svm_best_model.pkl'
    xgb_model_path      = f'{GLOBAL_PATH_INIT}/Descriptors/regressors/feat_xgb_best_model.pkl'
    nn_model_path       = f'{GLOBAL_PATH_INIT}/Descriptors/regressors/feat_nn_best_model.pt'
    
    def __init__(self,
                 smiles: str,
                 regressor: str='None'
                 ):
        self.smiles = smiles
        self.regressor = regressor
        self.scaler_feats = load_from_pickle(self.scaler_path_feats)
        self.scaler_labels = load_from_pickle(self.scaler_path_labels)
        self.features = load_json(self.features_path)
        if isinstance(self.smiles, str):
            self.preprocessing()
        if isinstance(self.smiles, (list, tuple)):
            self.preprocessing_list()
        
    def preprocessing(self):
        
        X = compute_descriptors_from_name(self.smiles, self.features)
        X = np.array([X[k] for k in self.features]).reshape(1, -1)
        self.X = self.scaler_feats.transform(X)
        
        return self.X
    
    def preprocessing_list(self):
        
        X_init = []
        for smi in self.smiles:
            X = compute_descriptors_from_name(smi, self.features)
            X_init.append([X[k] for k in self.features])
        X = np.array(X_init)
        self.X = self.scaler_feats.transform(X)
        
        return self.X
        
    def predict_feat_nonNN(self, model_path):
        
        model = load_from_pickle(model_path)
        pred_scaled = model.predict(self.X).reshape(-1, 1)
        pred = self.scaler_labels.inverse_transform(pred_scaled)
        pred = _format_prediction(pred)
        
        return pred
        

    def predict_feat_nn(self, model_path=nn_model_path):
        
        nn_model = SMILESLinearNet(list_dims=[91, 512, 512, 1],
                                   dropout=0.2,
                                   act_func='relu',
                                   norm_type='layer')
        Trainer_functions.load_test_network(network=nn_model, temp_network_path=model_path, verbose=False)
        nn_model.eval()

        with torch.no_grad():
            pred_scaled = nn_model(torch.from_numpy(self.X).float()).reshape(-1, 1)
        pred = self.scaler_labels.inverse_transform(pred_scaled)
        pred = _format_prediction(pred)
        
        return pred
    
    def predict(self):
        
        if self.regressor == 'nn':
            return self.predict_feat_nn(self.nn_model_path)
        elif self.regressor == 'xgb':
            return self.predict_feat_nonNN(self.xgb_model_path)
        elif self.regressor == 'rf':
            return self.predict_feat_nonNN(self.rf_model_path)
        elif self.regressor == 'svm':
            return self.predict_feat_nonNN(self.svm_model_path)
        elif self.regressor == 'all':
            y_pred_nn = self.predict_feat_nn(self.nn_model_path)
            y_pred_rf = self.predict_feat_nonNN(self.rf_model_path)
            y_pred_svm = self.predict_feat_nonNN(self.svm_model_path)
            y_pred_xgb = self.predict_feat_nonNN(self.xgb_model_path)
            return {'NN':y_pred_nn, 'RF':y_pred_rf, 'SVM':y_pred_svm, 'XGB':y_pred_xgb}
        else:
            raise ValueError(
                f"Unknown regressor: {self.regressor}. "
                f"Choose from 'nn', 'xgb', 'rf', 'svm'."
            )
    
def predict_cltot_feat(smiles: str, regressor: str):
    
    obj = PredDescriptors(smiles, regressor)
    prediction = obj.predict()
    
    return prediction

class PredGCN:
    
    scaler_path_feats   = f'{GLOBAL_PATH_INIT}/GCN/regressors/StandardScaler_feats.pkl'
    scaler_path_labels  = f'{GLOBAL_PATH_INIT}/GCN/regressors/StandardScaler_labels.pkl'
    
    rf_model_path       = f'{GLOBAL_PATH_INIT}/GCN/regressors/gcn_rf_best_model.pkl'
    svm_model_path      = f'{GLOBAL_PATH_INIT}/GCN/regressors/gcn_svm_best_model.pkl'
    xgb_model_path      = f'{GLOBAL_PATH_INIT}/GCN/regressors/gcn_xgb_best_model.pkl'
    nn_model_path       = f'{GLOBAL_PATH_INIT}/GCN/regressors/gcn_nn_best_model.pt'
    
    mol_details_path    = f'{GLOBAL_PATH_INIT}/GCN/repr_model/Dataset_atoms_maxatoms_hybds_chityp_bndlst.json'
    mean_path           = f'{GLOBAL_PATH_INIT}/GCN/repr_model/dataset_mean.pkl'
    std_path            = f'{GLOBAL_PATH_INIT}/GCN/repr_model/dataset_std.pkl'
    regr_model_path     = f'{GLOBAL_PATH_INIT}/GCN/repr_model/gcn_repr_best_model.pt'
    features_list       = ['atom_properties', 'logP_values']
    
    def __init__(self,
                 smiles: str,
                 regressor: str='None'
                 ):
        self.smiles = smiles
        self.regressor = regressor
        self.scaler_feats = load_from_pickle(self.scaler_path_feats)
        self.scaler_labels = load_from_pickle(self.scaler_path_labels)
        self.mol_details = load_json(self.mol_details_path)
        self.dataset_mean = load_from_pickle(self.mean_path)
        self.dataset_std = load_from_pickle(self.std_path)
        
        self.gcn_model = MRGCN(list_dims_gcn=[16, 76, 136, 196, 256],
                               list_dims_fc=[192, 128, 64, 1],
                               max_num_atom=72,
                               dropout=0.3,
                               num_relations=4,
                               act_func='relu',
                               act_func_gcn='relu',
                               norm_type='layer',
                               alpha=1.0,
                               negative_slope=0.01,
                               use_attention=True,
                               attn_type='self attention',
                               attn_dim=16,
                               num_heads=4
                               )
        Trainer_functions.load_test_network(network=self.gcn_model, temp_network_path=self.regr_model_path, verbose=False)
        self.gcn_model.eval()
        
        if isinstance(self.smiles, str):
            self.preprocessing()
        if isinstance(self.smiles, (list, tuple)):
            self.preprocessing_list()
        

    def preprocessing_init(self, smiles):
        
        X = Multirelational_GraphDataset.smiles2GcnDictData(smiles, **self.mol_details)
        temp = GraphData_from_pickle.GcnDictData2GcnInput(X, self.dataset_mean, self.dataset_std,
                                                          max_num_atoms=self.mol_details['max_num_atoms'],
                                                          features_list=self.features_list)
        feature_vector, adjacency_tensor, degree_tensor = temp
        input_vector = self.gcn_model.get_features(feature_vector=feature_vector.unsqueeze(0),
                                                    adjacency_tensor=adjacency_tensor.unsqueeze(0),
                                                    degree_tensor=degree_tensor.unsqueeze(0)
                                                    )
    
        return input_vector.detach().cpu().numpy()
    
    def preprocessing(self):
        
        input_vector = self.preprocessing_init(self.smiles)
        self.X = self.scaler_feats.transform(input_vector)
        
        return self.X
    
    def preprocessing_list(self):
        
        input_vector_list = []
        for smi in self.smiles:
            input_vector_list.append(self.preprocessing_init(smi))
        input_vector_list = np.array(input_vector_list)
        self.X = self.scaler_feats.transform(np.squeeze(input_vector_list, axis=1))
         
        return self.X
        
    def predict_gcn_nonNN(self, model_path):
        
        model = load_from_pickle(model_path)
        pred_scaled = model.predict(self.X).reshape(-1, 1)
        pred = self.scaler_labels.inverse_transform(pred_scaled)
        pred = _format_prediction(pred)
        
        return pred

    def predict_gcn_nn(self, model_path):
        
        nn_model = SMILESLinearNet(list_dims=[256, 171, 86, 1],
                                   dropout=0.3,
                                   act_func='relu',
                                   norm_type='layer'
                                   )
        Trainer_functions.load_test_network(network=nn_model, temp_network_path=model_path, verbose=False)
        nn_model.eval()

        with torch.no_grad():
            pred_scaled = nn_model(torch.from_numpy(self.X).float()).reshape(-1, 1)
        pred = self.scaler_labels.inverse_transform(pred_scaled)
        pred = _format_prediction(pred)
        
        return pred
    
    def predict(self):
        
        if self.regressor == 'nn':
            return self.predict_gcn_nn(self.nn_model_path)
        elif self.regressor == 'xgb':
            return self.predict_gcn_nonNN(self.xgb_model_path)
        elif self.regressor == 'rf':
            return self.predict_gcn_nonNN(self.rf_model_path)
        elif self.regressor == 'svm':
            return self.predict_gcn_nonNN(self.svm_model_path)
        elif self.regressor == 'all':
            y_pred_nn = self.predict_gcn_nn(self.nn_model_path)
            y_pred_rf = self.predict_gcn_nonNN(self.rf_model_path)
            y_pred_svm = self.predict_gcn_nonNN(self.svm_model_path)
            y_pred_xgb = self.predict_gcn_nonNN(self.xgb_model_path)
            return {'NN':y_pred_nn, 'RF':y_pred_rf, 'SVM':y_pred_svm, 'XGB':y_pred_xgb}
        else:
            raise ValueError(
                f"Unknown regressor: {self.regressor}. "
                f"Choose from 'nn', 'xgb', 'rf', 'svm'."
            )
    
def predict_cltot_gcn(smiles: str, regressor: str):
    
    obj = PredGCN(smiles, regressor)
    prediction = obj.predict()
    
    return prediction


class PredChemBert:
    
    repr_model_path     = f'{GLOBAL_PATH_INIT}/ChemBert/repr_model/bert_repr_best_model.pt'
    scaler_path_feats   = f'{GLOBAL_PATH_INIT}/ChemBert/regressors/StandardScaler_feats.pkl'
    scaler_path_labels  = f'{GLOBAL_PATH_INIT}/ChemBert/regressors/StandardScaler_labels.pkl'

    rf_model_path       = f'{GLOBAL_PATH_INIT}/ChemBert/regressors/bert_rf_best_model.pkl'
    svm_model_path      = f'{GLOBAL_PATH_INIT}/ChemBert/regressors/bert_svm_best_model.pkl'
    xgb_model_path      = f'{GLOBAL_PATH_INIT}/ChemBert/regressors/bert_xgb_best_model.pkl'
    nn_model_path       = f'{GLOBAL_PATH_INIT}/ChemBert/regressors/bert_nn_best_model.pt'
    
    def __init__(self,
                 smiles: str,
                 regressor: str='None'
                 ):
        self.smiles = smiles
        self.regressor = regressor
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.scaler_feats = load_from_pickle(self.scaler_path_feats)
        self.scaler_labels = load_from_pickle(self.scaler_path_labels)
        self.MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"
        self.TOKENIZER = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.repr_model = ChemBERTaRegressor(model_name='DeepChem/ChemBERTa-77M-MLM',
                                             list_dims=[384, 192, 1],
                                             dropout=0.3,
                                             act_func='relu',
                                             norm_type='layer'
                                             ).to(self.device)
        Trainer_functions.load_test_network(network=self.repr_model, temp_network_path=self.repr_model_path, verbose=False)
        self.repr_model.eval()
        
        if isinstance(self.smiles, str):
            self.preprocessing()
        if isinstance(self.smiles, (list, tuple)):
            self.preprocessing_list()
        
    def preprocessing(self):
        
        tokens = self.TOKENIZER(self.smiles, padding="max_length", return_tensors="pt").to(self.device)
        repr_smi = self.repr_model.embeddings(**tokens)[0].cpu().detach().numpy()
        repr_smi = repr_smi.reshape(1, -1)
        self.X = self.scaler_feats.transform(repr_smi)
        tokens = None
        
        return self.X
    
    def preprocessing_list(self, batch_size=128):
        
        reprs = []
        for start in range(0, len(self.smiles), batch_size):
            temp_smiles = self.smiles[start:start+batch_size]
            tokens = self.TOKENIZER(temp_smiles, padding="max_length", return_tensors="pt").to(self.device)
            _repr = self.repr_model.embeddings(**tokens).cpu().detach().numpy()
            reprs.append(_repr)
        reprs = np.concatenate(reprs)
        self.X = self.scaler_feats.transform(reprs)
        tokens, reprs = None, None
        
        return self.X
        
    def predict_bert_nonNN(self, model_path):
        
        model = load_from_pickle(model_path)
        pred_scaled = model.predict(self.X).reshape(-1, 1)
        pred = self.scaler_labels.inverse_transform(pred_scaled)
        pred = _format_prediction(pred)
        
        return pred

    def predict_bert_nn(self, model_path=nn_model_path):
        
        nn_model = SMILESLinearNet(list_dims=[384, 256, 128, 1],
                                   dropout=0.5,
                                   act_func='relu',
                                   norm_type='layer'
                                   )
        Trainer_functions.load_test_network(network=nn_model, temp_network_path=model_path, verbose=False)
        nn_model.eval()

        with torch.no_grad():
            pred_scaled = nn_model(torch.from_numpy(self.X).float()).reshape(-1, 1)
        pred = self.scaler_labels.inverse_transform(pred_scaled)
        pred = _format_prediction(pred)
        
        return pred
    
    def predict(self):
        
        if self.regressor == 'nn':
            return self.predict_bert_nn(self.nn_model_path)
        elif self.regressor == 'xgb':
            return self.predict_bert_nonNN(self.xgb_model_path)
        elif self.regressor == 'rf':
            return self.predict_bert_nonNN(self.rf_model_path)
        elif self.regressor == 'svm':
            return self.predict_bert_nonNN(self.svm_model_path)
        elif self.regressor == 'all':
            y_pred_nn = self.predict_bert_nn(self.nn_model_path)
            y_pred_rf = self.predict_bert_nonNN(self.rf_model_path)
            y_pred_svm = self.predict_bert_nonNN(self.svm_model_path)
            y_pred_xgb = self.predict_bert_nonNN(self.xgb_model_path)
            return {'NN':y_pred_nn, 'RF':y_pred_rf, 'SVM':y_pred_svm, 'XGB':y_pred_xgb}
        else:
            raise ValueError(
                f"Unknown regressor: {self.regressor}. "
                f"Choose from 'nn', 'xgb', 'rf', 'svm'."
            )
    
def predict_cltot_bert(smiles: str, regressor: str):
    
    obj = PredChemBert(smiles, regressor)
    prediction = obj.predict()
    
    return prediction


class PredFusionNet:
    
    scaler_path_feats   = f'{GLOBAL_PATH_INIT}/Fused_model/regressors/StandardScaler_feats.pkl'
    scaler_path_labels  = f'{GLOBAL_PATH_INIT}/Fused_model/regressors/StandardScaler_labels.pkl'
    repr_model_path     = f'{GLOBAL_PATH_INIT}/Fused_model/repr_model/fnet_repr_best_model.pt'

    rf_model_path       = f'{GLOBAL_PATH_INIT}/Fused_model/regressors/fnet_rf_best_model.pkl'
    svm_model_path      = f'{GLOBAL_PATH_INIT}/Fused_model/regressors/fnet_svm_best_model.pkl'
    xgb_model_path      = f'{GLOBAL_PATH_INIT}/Fused_model/regressors/fnet_xgb_best_model.pkl'
    nn_model_path       = f'{GLOBAL_PATH_INIT}/Fused_model/regressors/fnet_nn_best_model.pt'

    def __init__(self,
                 smiles: str,
                 regressor: str='None',
                 ):
        
        self.smiles = smiles
        if isinstance(regressor, (list, tuple)):
            self.regressor = [regr.lower().strip() for regr in regressor]
        elif isinstance(regressor, str):
            self.regressor = regressor.lower().strip()
        self.scaler_feats = load_from_pickle(self.scaler_path_feats)
        self.scaler_labels = load_from_pickle(self.scaler_path_labels)
        
        self.repr_model = FusionNet(input_dims=[91, 300, 256, 384],
                                    fusion_dim=512,
                                    mlp_params={'list_dims': [512, 256, 1],
                                                'dropout': 0.2,
                                                'act_func': 'relu',
                                                'norm_type': 'layer',
                                                'alpha': 1,
                                                'negative_slope': 0.01})
        Trainer_functions.load_test_network(network=self.repr_model, temp_network_path=self.repr_model_path, verbose=False)
        self.repr_model.eval()
        
        if isinstance(self.smiles, str):
            self.preprocessing()
        if isinstance(self.smiles, (list, tuple)):
            self.preprocessing_list()
        
    def preprocessing(self):   
         
        Mol2Vec = PredMol2Vec(self.smiles)
        Descriptors = PredDescriptors(self.smiles)
        GCN = PredGCN(self.smiles)
        Bert = PredChemBert(self.smiles)
        
        self.d_repr = torch.from_numpy(Descriptors.X).type(torch.float)
        self.b_repr = torch.from_numpy(Bert.X).type(torch.float)
        self.m_repr = torch.from_numpy(Mol2Vec.X).type(torch.float)
        self.g_repr = torch.from_numpy(GCN.X).type(torch.float)
        
        with torch.no_grad():
            vec = self.repr_model.feature_repr((self.d_repr, self.m_repr,  self.g_repr, self.b_repr))
        self.X = self.scaler_feats.transform(vec.detach().numpy())
        
        return self.X
    
    def preprocessing_list(self):   
         
        Mol2Vec = PredMol2Vec(self.smiles)
        Descriptors = PredDescriptors(self.smiles)
        GCN = PredGCN(self.smiles)
        Bert = PredChemBert(self.smiles)
        
        self.d_repr = torch.from_numpy(Descriptors.X).type(torch.float)
        self.b_repr = torch.from_numpy(Bert.X).type(torch.float)
        self.m_repr = torch.from_numpy(Mol2Vec.X).type(torch.float)
        self.g_repr = torch.from_numpy(GCN.X).type(torch.float)
        
        with torch.no_grad():
            vec = self.repr_model.feature_repr((self.d_repr, self.m_repr,  self.g_repr, self.b_repr))
        self.X = self.scaler_feats.transform(vec.detach().numpy())
        
        return self.X
        
    def predict_FusionNet_nonNN(self, model_path):
        
        model = load_from_pickle(model_path)
        pred_scaled = model.predict(self.X).reshape(-1, 1)
        pred = self.scaler_labels.inverse_transform(pred_scaled)
        pred = _format_prediction(pred)
        
        return pred

    def predict_FusionNet_nn(self, model_path=nn_model_path):
        
        nn_model = SMILESLinearNet(list_dims=[512, 341, 171, 1],
                                   dropout=0.5,
                                   act_func='leakyrelu',
                                   norm_type='layer'
                                   )
        
        Trainer_functions.load_test_network(network=nn_model, temp_network_path=model_path, verbose=False)
        nn_model.eval()

        with torch.no_grad():
            pred_scaled = nn_model(torch.from_numpy(self.X).float()).reshape(-1, 1)
        
        pred = self.scaler_labels.inverse_transform(pred_scaled)
        pred = _format_prediction(pred)
        
        return pred

    def predict(self):

        if isinstance(self.regressor, str):
            regressors = [self.regressor.lower()]
        elif isinstance(self.regressor, (list, tuple)):
            regressors = [r.lower() for r in self.regressor]
        else:
            raise TypeError("regressor must be a string, list, or tuple.")

        if 'all' in regressors:
            regressors = ['nn', 'rf', 'svm', 'xgb']

        supported = {'nn', 'rf', 'svm', 'xgb'}
        invalid = set(regressors) - supported
        if invalid:
            raise ValueError(f"Unknown regressor(s): {sorted(invalid)}. Choose from 'nn', 'rf', 'svm', 'xgb', or 'all'.")

        predictions = {}
        for regressor in regressors:
            if regressor == 'nn':
                predictions['NN'] = self.predict_FusionNet_nn(self.nn_model_path)
            elif regressor == 'xgb':
                predictions['XGB'] = self.predict_FusionNet_nonNN(self.xgb_model_path)
            elif regressor == 'rf':
                predictions['RF'] = self.predict_FusionNet_nonNN(self.rf_model_path)
            elif regressor == 'svm':
                predictions['SVM'] = self.predict_FusionNet_nonNN(self.svm_model_path)

        return predictions


def predict_cltot_FusionNet(smiles: str, regressor: str):
    
    obj = PredFusionNet(smiles, regressor)
    prediction = obj.predict()
    
    return prediction

def predict_cltot(smiles, regressor='all', representation='FusionNet'):

    if isinstance(smiles, pd.Series):
        smiles = smiles.tolist()
    elif isinstance(smiles, str):
        smiles = [smiles]
    elif isinstance(smiles, (list, tuple)):
        smiles = list(smiles)
    else:
        raise TypeError("smiles must be a string, pandas Series, list, or tuple")

    predict_funcs = {'mol2vec': predict_cltot_mol2vec,
                    'gcn': predict_cltot_gcn,
                    'chembert': predict_cltot_bert,
                    'descriptors': predict_cltot_feat,
                    'FusionNet': predict_cltot_FusionNet
                    }
    representation = representation.strip()
    if representation not in predict_funcs:
        raise ValueError(f"Unknown representation: {representation}. Choose from {list(predict_funcs)}")
    predict_func = predict_funcs[representation]
    y_pred_temp = predict_func(smiles=smiles, regressor=regressor)

    svg_strings = [plot_smiles_grid(smiles_list=[smi], 
                                    cols=1, 
                                    image_size=(600, 600), 
                                    figsize=(12, 8), 
                                    atom_font_size=25, 
                                    fmt='svg', 
                                    show=False
                                    ) for smi in smiles]

    if isinstance(y_pred_temp, dict):
        n = len(smiles)
        predictions = []
        for i in range(n):
            pred_i = {key: np.asarray(val).reshape(-1)[i].item() for key, val in y_pred_temp.items()}
            predictions.append(pred_i)
    else:
        pred_array = np.asarray(y_pred_temp).reshape(-1)
        predictions = [pred_array[i].item() for i in range(len(smiles))]

    output = [{'smiles': smi, **pred, 'image': svg} 
              for smi, pred, svg in zip(smiles, predictions, svg_strings)
              ]

    return output