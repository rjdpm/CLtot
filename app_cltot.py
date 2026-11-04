#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Total Clearance Molecular Prediction API

Run:
    uvicorn app_cltot:app --host 0.0.0.0 --port 8000

Swagger:
    http://localhost:8000/docs
"""

import os
import sys
import warnings
from contextlib import asynccontextmanager
from typing import List, Union
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rdkit import Chem
from transformers import AutoTokenizer, utils
from mol2vec.features import mol2alt_sentence, sentences2vec
from gensim.models import word2vec

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "../")))

from inference_utils_cltot.data_utils_smiles_cltot import compute_descriptors_from_name, plot_smiles_grid
from inference_utils_cltot.graph_utils_smiles_cltot import Multirelational_GraphDataset, GraphData_from_pickle
from inference_utils_cltot.data_utils_cltot import load_from_pickle, fix_seed, load_json
from inference_utils_cltot.train_module_cltot import Trainer_functions
from inference_utils_cltot.models_module_cltot import ChemBERTaRegressor, MRGCN, SMILESLinearNet, FusionNet

# ============================================================
# CONFIGURATION
# ============================================================

GLOBAL_PATH_INIT = os.path.join(BASE_DIR, "Best_models")
CHEMBERT_LOCAL_PATH = os.path.join(GLOBAL_PATH_INIT, "ChemBert", "repr_model")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

utils.logging.disable_progress_bar()
fix_seed(np_seed=20, torch_seed=30, random_seed=10)

print("=" * 70)
print("CLTOT Prediction API")
print("=" * 70)
print(f"Base directory : {BASE_DIR}")
print(f"Models         : {GLOBAL_PATH_INIT}")
print(f"Device         : {DEVICE}")
print("=" * 70)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def _round(x, decimal=4):

    if isinstance(x, (str, float, int)):
        return float(f"{x:.{decimal}f}")
    if isinstance(x, (list, np.ndarray, tuple, torch.Tensor)):
        return [float(f"{_x:.{decimal}f}") for _x in x]

    return x


def _format_prediction(pred):

    pred = np.asarray(pred).reshape(-1)
    if len(pred) == 1:
        return _round(pred[0].item(), 4)

    return _round(pred.tolist(), 4)


# ============================================================
# MODEL LOADING HELPERS
# ============================================================

def load_sklearn_model(path):

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    print(f"Loading sklearn model: {path}")

    return load_from_pickle(path)


def load_xgb_model(path):

    if not os.path.exists(path):
        raise FileNotFoundError(f"XGBoost model not found: {path}")

    print(f"Loading XGBoost model: {path}")
    model = xgb.XGBRegressor()
    model.load_model(path)

    return model


def load_nn_model(input_dim, hidden_dims, output_dim, dropout, act_func, norm_type, model_path):

    model = SMILESLinearNet(list_dims=[input_dim, *hidden_dims, output_dim],
                            dropout=dropout,
                            act_func=act_func,
                            norm_type=norm_type
                            )

    Trainer_functions.load_test_network(network=model, temp_network_path=model_path, verbose=False)
    model.eval()

    return model


# ============================================================
# MODEL STORE
# ============================================================

class ModelStore:

    def __init__(self):
        self.device = DEVICE

        self.mol2vec = {}
        self.descriptors = {}
        self.gcn = {}
        self.chembert = {}
        self.fusion = {}

        self.load_mol2vec()
        self.load_descriptors()
        self.load_gcn()
        self.load_chembert()
        self.load_fusion()

        print("=" * 70)
        print("ALL MODELS LOADED")
        print("=" * 70)

    # ========================================================
    # MOL2VEC
    # ========================================================
    def load_mol2vec(self):

        print("\nLoading Mol2Vec models...")
        base = os.path.join(GLOBAL_PATH_INIT, "Mol2Vec")
        self.mol2vec["representation"] = word2vec.Word2Vec.load(os.path.join(base, "repr_model", "model_300dim.pkl"))
        self.mol2vec["scaler_feats"] = load_from_pickle(os.path.join(base, "regressors", "StandardScaler_feats.pkl"))
        self.mol2vec["scaler_labels"] = load_from_pickle(os.path.join(base, "regressors", "StandardScaler_labels.pkl"))
        self.mol2vec["rf"] = load_sklearn_model(os.path.join(base, "regressors", "mol2vec_rf_best_model.pkl"))
        self.mol2vec["svm"] = load_sklearn_model(os.path.join(base, "regressors", "mol2vec_svm_best_model.pkl"))
        self.mol2vec["xgb"] = load_xgb_model(os.path.join(base, "regressors", "mol2vec_xgb_best_model.json"))

        self.mol2vec["nn"] = load_nn_model(input_dim=300,
                                           hidden_dims=[256, 128],
                                           output_dim=1,
                                           dropout=0.4,
                                           act_func="relu",
                                           norm_type="layer",
                                           model_path=os.path.join(base, "regressors", "mol2vec_nn_best_model.pt")
                                           )
        print("Mol2Vec loaded.")

    # ========================================================
    # DESCRIPTORS
    # ========================================================
    def load_descriptors(self):

        print("\nLoading Descriptor models...")
        base = os.path.join(GLOBAL_PATH_INIT, "Descriptors")
        self.descriptors["features"] = load_json(os.path.join(base, "repr_model", "optimal_features.json"))
        self.descriptors["scaler_feats"] = load_from_pickle(os.path.join(base, "regressors", "StandardScaler_feats.pkl"))
        self.descriptors["scaler_labels"] = load_from_pickle(os.path.join(base, "regressors", "StandardScaler_labels.pkl"))
        self.descriptors["rf"] = load_sklearn_model(os.path.join(base, "regressors", "feat_rf_best_model.pkl"))
        self.descriptors["svm"] = load_sklearn_model(os.path.join(base, "regressors", "feat_svm_best_model.pkl"))
        self.descriptors["xgb"] = load_xgb_model(os.path.join(base, "regressors", "feat_xgb_best_model.json"))
        self.descriptors["nn"] = load_nn_model(input_dim=91,
                                               hidden_dims=[512, 512],
                                               output_dim=1,
                                               dropout=0.2,
                                               act_func="relu",
                                               norm_type="layer",
                                               model_path=os.path.join(base, "regressors", "feat_nn_best_model.pt")
                                               )
        print("Descriptors loaded.")
    # ========================================================
    # GCN
    # ========================================================
    def load_gcn(self):

        print("\nLoading GCN models...")
        base = os.path.join(GLOBAL_PATH_INIT, "GCN")
        self.gcn["scaler_feats"] = load_from_pickle(os.path.join(base, "regressors", "StandardScaler_feats.pkl"))
        self.gcn["scaler_labels"] = load_from_pickle(os.path.join(base, "regressors", "StandardScaler_labels.pkl"))
        self.gcn["mol_details"] = load_json(os.path.join(base, "repr_model", "Dataset_atoms_maxatoms_hybds_chityp_bndlst.json"))
        self.gcn["dataset_mean"] = load_from_pickle(os.path.join(base, "repr_model", "dataset_mean.pkl"))
        self.gcn["dataset_std"] = load_from_pickle(os.path.join(base, "repr_model", "dataset_std.pkl"))
        self.gcn["features_list"] = ["atom_properties", "logP_values"]
        self.gcn["representation"] = MRGCN(list_dims_gcn=[16, 76, 136, 196, 256],
                                           list_dims_fc=[192, 128, 64, 1],
                                           max_num_atom=72,
                                           dropout=0.3,
                                           num_relations=4,
                                           act_func="relu",
                                           act_func_gcn="relu",
                                           norm_type="layer",
                                           alpha=1.0,
                                           negative_slope=0.01,
                                           use_attention=True,
                                           attn_type="self attention",
                                           attn_dim=16,
                                           num_heads=4
                                           )

        Trainer_functions.load_test_network(network=self.gcn["representation"],
                                            temp_network_path=os.path.join(base, "repr_model", "gcn_repr_best_model.pt"),
                                            verbose=False
                                            )
        self.gcn["representation"].eval()
        self.gcn["rf"] = load_sklearn_model(os.path.join(base, "regressors", "gcn_rf_best_model.pkl"))
        self.gcn["svm"] = load_sklearn_model(os.path.join(base, "regressors", "gcn_svm_best_model.pkl"))
        self.gcn["xgb"] = load_xgb_model(os.path.join(base, "regressors", "gcn_xgb_best_model.json"))
        self.gcn["nn"] = load_nn_model(input_dim=256,
                                       hidden_dims=[171, 86],
                                       output_dim=1,
                                       dropout=0.3,
                                       act_func="relu",
                                       norm_type="layer",
                                       model_path=os.path.join(base, "regressors", "gcn_nn_best_model.pt")
                                       )
        print("GCN loaded.")

    # ========================================================
    # CHEMBERT
    # ========================================================

    def load_chembert(self):

        print("\nLoading ChemBERT...")

        base = os.path.join(GLOBAL_PATH_INIT, "ChemBert")
        if not os.path.isdir(CHEMBERT_LOCAL_PATH):
            raise FileNotFoundError("\nLocal ChemBERT directory not found:\n {CHEMBERT_LOCAL_PATH}\n\n")
        print("Loading ChemBERT tokenizer locally from:\n{CHEMBERT_LOCAL_PATH}")

        self.chembert["tokenizer"] = AutoTokenizer.from_pretrained(CHEMBERT_LOCAL_PATH, local_files_only=True)
        self.chembert["scaler_feats"] = load_from_pickle(os.path.join(base, "regressors", "StandardScaler_feats.pkl"))
        self.chembert["scaler_labels"] = load_from_pickle(os.path.join(base, "regressors", "StandardScaler_labels.pkl"))

        self.chembert["representation"] = ChemBERTaRegressor(model_name=CHEMBERT_LOCAL_PATH,
                                                             list_dims=[384, 192, 1],
                                                             dropout=0.3,
                                                             act_func="relu",
                                                             norm_type="layer",
                                                             ).to(self.device)
        Trainer_functions.load_test_network(network=self.chembert["representation"],
                                            temp_network_path=os.path.join(base, "repr_model", "bert_repr_best_model.pt"),
                                            verbose=False
                                            )
        self.chembert["representation"].eval()
        self.chembert["rf"] = load_sklearn_model(os.path.join(base, "regressors", "bert_rf_best_model.pkl"))
        self.chembert["svm"] = load_sklearn_model(os.path.join(base, "regressors", "bert_svm_best_model.pkl"))
        self.chembert["xgb"] = load_xgb_model(os.path.join(base, "regressors", "bert_xgb_best_model.json"))
        self.chembert["nn"] = load_nn_model(input_dim=384,
                                            hidden_dims=[256, 128],
                                            output_dim=1,
                                            dropout=0.5,
                                            act_func="relu",
                                            norm_type="layer",
                                            model_path=os.path.join(base, "regressors", "bert_nn_best_model.pt")
                                            )
        print("ChemBERT loaded locally.")

    # ========================================================
    # FUSION NET
    # ========================================================

    def load_fusion(self):

        print("\nLoading FusionNet...")
        base = os.path.join(GLOBAL_PATH_INIT, "Fused_model")
        self.fusion["scaler_feats"] = load_from_pickle(os.path.join(base, "regressors", "StandardScaler_feats.pkl"))
        self.fusion["scaler_labels"] = load_from_pickle(os.path.join(base, "regressors", "StandardScaler_labels.pkl"))
        self.fusion["representation"] = FusionNet(input_dims=[91, 300, 256, 384],
                                                  fusion_dim=512,
                                                  mlp_params={"list_dims": [512, 256, 1],
                                                              "dropout": 0.2,
                                                              "act_func": "relu",
                                                              "norm_type": "layer",
                                                              "alpha": 1,
                                                              "negative_slope": 0.01,
                                                              }
                                                )

        Trainer_functions.load_test_network(network=self.fusion["representation"], 
                                            temp_network_path=os.path.join(base, "repr_model", "fnet_repr_best_model.pt"),
                                            verbose=False
                                            )
        self.fusion["representation"].eval()
        self.fusion["rf"] = load_sklearn_model(os.path.join(base, "regressors", "fnet_rf_best_model.pkl"))
        self.fusion["svm"] = load_sklearn_model(os.path.join(base, "regressors", "fnet_svm_best_model.pkl"))
        self.fusion["xgb"] = load_xgb_model(os.path.join(base, "regressors", "fnet_xgb_best_model.json"))
        self.fusion["nn"] = load_nn_model(input_dim=512,
                                          hidden_dims=[341, 171],
                                          output_dim=1,
                                          dropout=0.5,
                                          act_func="leakyrelu",
                                          norm_type="layer",
                                          model_path=os.path.join(base, "regressors", "fnet_nn_best_model.pt")
                                          )
        print("FusionNet loaded.")

# ============================================================
# GLOBAL MODEL STORE
# ============================================================
MODEL_STORE = None
# ============================================================
# MOL2VEC
# ============================================================

def mol2vec_features(smiles_list):

    model = MODEL_STORE.mol2vec
    sentences = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smi}")

        sentences.append(mol2alt_sentence(mol, 1))

    vec = sentences2vec(sentences, model["representation"], unseen="UNK")

    return model["scaler_feats"].transform(vec)


def predict_mol2vec(smiles_list, regressor):

    model = MODEL_STORE.mol2vec
    X = mol2vec_features(smiles_list)

    if regressor == "nn":
        with torch.no_grad():
            pred_scaled = model["nn"](torch.from_numpy(X).float()).reshape(-1, 1).numpy()
    else:
        pred_scaled = model[regressor].predict(X).reshape(-1, 1)
    pred = model["scaler_labels"].inverse_transform(pred_scaled)

    return _format_prediction(pred)

# ============================================================
# DESCRIPTORS
# ============================================================
def descriptor_features(smiles_list):

    model = MODEL_STORE.descriptors
    X_list = []

    for smi in smiles_list:
        X = compute_descriptors_from_name(smi, model["features"])
        X_list.append([X[k] for k in model["features"]])
    X = np.asarray(X_list)

    return model["scaler_feats"].transform(X)


def predict_descriptors(smiles_list, regressor):

    model = MODEL_STORE.descriptors
    X = descriptor_features(smiles_list)

    if regressor == "nn":
        with torch.no_grad():
            pred_scaled = model["nn"](torch.from_numpy(X).float()).reshape(-1, 1).numpy()
    else:
        pred_scaled = model[regressor].predict(X).reshape(-1, 1)
    pred = model["scaler_labels"].inverse_transform(pred_scaled)

    return _format_prediction(pred)

# ============================================================
# GCN
# ============================================================
def gcn_single_feature(smi):

    model = MODEL_STORE.gcn

    X = Multirelational_GraphDataset.smiles2GcnDictData(smi, **model["mol_details"])
    temp = GraphData_from_pickle.GcnDictData2GcnInput(data=X, 
                                                      dataset_mean=model["dataset_mean"], 
                                                      dataset_std=model["dataset_std"],
                                                      max_num_atoms=model["mol_details"]["max_num_atoms"],
                                                      features_list=model["features_list"],
                                                      )
    feature_vector, adjacency_tensor, degree_tensor = temp

    with torch.no_grad():
        input_vector = model["representation"].get_features(feature_vector=feature_vector.unsqueeze(0),
                                                            adjacency_tensor=adjacency_tensor.unsqueeze(0),
                                                            degree_tensor=degree_tensor.unsqueeze(0)
                                                            )

    return input_vector.detach().cpu().numpy()


def gcn_features(smiles_list):

    vectors = []
    for smi in smiles_list:
        vectors.append(gcn_single_feature(smi))

    X = np.array(vectors)
    X = np.squeeze(X, axis=1)

    return MODEL_STORE.gcn["scaler_feats"].transform(X)


def predict_gcn(smiles_list, regressor):

    model = MODEL_STORE.gcn
    X = gcn_features(smiles_list)

    if regressor == "nn":
        with torch.no_grad():
            pred_scaled = model["nn"](torch.from_numpy(X).float()).reshape(-1, 1).numpy()
    else:
        pred_scaled = model[regressor].predict(X).reshape(-1, 1)
    pred = model["scaler_labels"].inverse_transform(pred_scaled)

    return _format_prediction(pred)

# ============================================================
# CHEMBERT
# ============================================================
def chembert_features(smiles_list):

    model = MODEL_STORE.chembert
    tokenizer = model["tokenizer"]
    bert = model["representation"]
    all_repr = []
    batch_size = 128

    for start in range(0, len(smiles_list), batch_size):

        batch = smiles_list[start:start + batch_size]
        tokens = tokenizer(batch, padding="max_length", return_tensors="pt")
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

        with torch.no_grad():
            repr_smi = bert.embeddings(**tokens).detach().cpu().numpy()
        all_repr.append(repr_smi)

    reprs = np.concatenate(all_repr, axis=0)

    return model["scaler_feats"].transform(reprs)


def predict_chembert(smiles_list, regressor):

    model = MODEL_STORE.chembert
    X = chembert_features(smiles_list)
    if regressor == "nn":
        with torch.no_grad():
            pred_scaled = model["nn"](torch.from_numpy(X).float()).reshape(-1, 1).numpy()
    else:
        pred_scaled = model[regressor].predict(X).reshape(-1, 1)

    pred = model["scaler_labels"].inverse_transform(pred_scaled)

    return _format_prediction(pred)

# ============================================================
# FUSION
# ============================================================
def fusion_features(smiles_list):

    model = MODEL_STORE.fusion
    descriptors = descriptor_features(smiles_list)
    mol2vec = mol2vec_features(smiles_list)
    gcn = gcn_features(smiles_list)
    bert = chembert_features(smiles_list)
    d_repr = torch.from_numpy(descriptors).float()
    m_repr = torch.from_numpy(mol2vec).float()
    g_repr = torch.from_numpy(gcn).float()
    b_repr = torch.from_numpy(bert).float()

    with torch.no_grad():
        vec = model["representation"].feature_repr((d_repr, m_repr, g_repr, b_repr))

    return model["scaler_feats"].transform(vec.detach().numpy())


def predict_fusion(smiles_list, regressor):

    model = MODEL_STORE.fusion
    X = fusion_features(smiles_list)
    if regressor == "nn":
        with torch.no_grad():
            pred_scaled = model["nn"](torch.from_numpy(X).float()).reshape(-1, 1).numpy()
    else:
        pred_scaled = model[regressor].predict(X).reshape(-1, 1)
    pred = model["scaler_labels"].inverse_transform(pred_scaled)

    return _format_prediction(pred)

# ============================================================
# GENERAL PREDICTION DISPATCHER
# ============================================================
def run_prediction(smiles_list, representation, regressors):

    representation_map = {"mol2vec": "mol2vec",
                          "descriptors": "descriptors",
                          "descriptor": "descriptors",
                          "gcn": "gcn",
                          "chembert": "chembert",
                          "bert": "chembert",
                          "fusionnet": "FusionNet",
                          "fusion": "FusionNet"
                          }
    key = representation_map.get(representation.strip().lower())

    if key is None:
        raise ValueError("Unknown representation. Choose from: mol2vec, descriptors, gcn, chembert, FusionNet.")

    if isinstance(regressors, str):
        regressors = [regressors.lower().strip()]
    else:
        regressors = [r.lower().strip() for r in regressors]

    if "all" in regressors:
        regressors = ["nn", "rf", "svm", "xgb"]

    supported = {"nn", "rf", "svm", "xgb"}
    invalid = set(regressors) - supported

    if invalid:
        raise ValueError(f"Unknown regressor(s): {sorted(invalid)}. Choose from nn, rf, svm, xgb, all.")

    prediction_functions = {"mol2vec": predict_mol2vec,
                            "descriptors": predict_descriptors,
                            "gcn": predict_gcn,
                            "chembert": predict_chembert,
                            "FusionNet": predict_fusion,
                            }
    predict_function = prediction_functions[key]
    raw_predictions = {}

    for regressor in regressors:
        raw_predictions[regressor.upper()] = predict_function(smiles_list, regressor)

    results = []
    for i, smi in enumerate(smiles_list):
        item = {"smiles": smi}
        for name, values in raw_predictions.items():
            values_array = np.asarray(values).reshape(-1)
            item[name] = float(values_array[i])

        results.append(item)

    return results
# ============================================================
# IMAGE
# ============================================================

def create_smiles_image(smi):

    try:
        return plot_smiles_grid(smiles_list=[smi],
                                cols=1,
                                image_size=(600, 600),
                                figsize=(12, 8),
                                atom_font_size=25,
                                fmt="svg",
                                show=False,
                                )
    except Exception as e:
        print(f"Could not create image for {smi}: {e}")

        return None
# ============================================================
# API SCHEMA
# ============================================================

class PredictionRequest(BaseModel):

    smiles: Union[str, List[str]] = Field(..., description="One SMILES or a list of SMILES.")
    representation: str = Field(default="FusionNet", description=("mol2vec, descriptors, gcn, chembert, or FusionNet."))
    regressor: Union[str, List[str]] = Field(default="all", description=("nn, rf, svm, xgb, all, or a list."))
    include_image: bool = Field(default=True, description="Include molecular SVG image.")
# ============================================================
# FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global MODEL_STORE
    print("\nStarting CLTOT API...")
    print("Loading models once at startup...")
    MODEL_STORE = ModelStore()
    print("\nServer is ready.")
    print("Models will be reused for all requests.")
    yield

    print("\nShutting down CLTOT API.")
    MODEL_STORE = None
# ============================================================
# FASTAPI APPLICATION
# ============================================================
app = FastAPI(title="CLTOT Molecular Prediction API",
              description="Molecular property prediction using Mol2Vec, Descriptors, GCN, ChemBERT and FusionNet.",
              version="1.0.0",
              lifespan=lifespan
              )
# ============================================================
# ROOT
# ============================================================
@app.get("/")
def root():
    return {"name": "CLTOT Molecular Prediction API",
            "status": "running",
            "device": str(DEVICE),
            "docs": "/docs"
            }
# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {"status": "healthy",
            "models_loaded": MODEL_STORE is not None,
            "device": str(DEVICE)
            }
# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/models")
def models():
    return {"representations": ["mol2vec", "descriptors", "gcn", "chembert", "FusionNet"],
            "regressors": ["nn", "rf", "svm", "xgb", "all"],
            "chembert_local": True,
            "device": str(DEVICE)
            }
# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    if MODEL_STORE is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet.")

    if isinstance(request.smiles, str):
        smiles_list = [request.smiles]
    else:
        smiles_list = list(request.smiles)

    if not smiles_list:
        raise HTTPException(status_code=400, detail="No SMILES provided.")
    # --------------------------------------------------------
    # Validate SMILES
    # --------------------------------------------------------
    invalid_smiles = []
    for smi in smiles_list:
        if not isinstance(smi, str):
            invalid_smiles.append(str(smi))
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid_smiles.append(smi)

    if invalid_smiles:
        raise HTTPException(status_code=400, 
                            detail={"message": "Invalid SMILES string(s).",
                                    "invalid_smiles": invalid_smiles,
                                    },
                            )
    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------
    try:
        results = run_prediction(smiles_list=smiles_list, 
                                 representation=request.representation,
                                 regressors=request.regressor
                                 )
    except Exception as e:
        print("Prediction error:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))
    # --------------------------------------------------------
    # Molecular images
    # --------------------------------------------------------
    if request.include_image:
        for result in results:
            result["image"] = (create_smiles_image(result["smiles"]))

    out = {"success": True,
           "representation": request.representation,
           "regressor": request.regressor,
           "count": len(results),
           "results": results,
           }

    return out
# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_cltot:app", host="0.0.0.0", port=8000, reload=False,)
