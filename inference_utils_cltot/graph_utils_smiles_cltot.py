import torch
import numpy as np
import pickle, gzip
from torch.utils.data import Dataset
from collections import Counter
from rdkit import Chem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from logP_values_cltot import *

__all__ = [
    'list_to_onehot',
    'get_element_properties',
    'get_multirelational_bond_matrix',
    'Multirelational_GraphDataset',
    'GraphData_from_pickle',
]

path_1 = os.path.dirname(os.path.abspath(__file__))
GLOBAL_PATH_INIT = f'{path_1}/../Best_models_CLtot/metadata'
dict_path = f'{GLOBAL_PATH_INIT}/All_possible_atoms_X.pkl'

with open(dict_path, 'rb') as fp:
    logP_dict = pickle.load(fp)
    
def list_to_onehot(element, elements_list, val=1.0) -> list:
    
    onehot = [0.]*len(elements_list)
    idx = elements_list.index(element)
    onehot[idx] = val
    
    return onehot

def get_element_properties(symbol):
    
    data_elec = {
        "H": {"VdW Radius": 120, "Sanderson Electronegativity": 2.592, "Polarizability": 0.667, "Pauling Electronegativity": 2.2},
        "B": {"VdW Radius": 191, "Sanderson Electronegativity": 2.275, "Polarizability": 3.030, "Pauling Electronegativity": 2.04},
        "C": {"VdW Radius": 177, "Sanderson Electronegativity": 2.746, "Polarizability": 1.760, "Pauling Electronegativity": 2.55},
        "N": {"VdW Radius": 166, "Sanderson Electronegativity": 3.194, "Polarizability": 1.100, "Pauling Electronegativity": 3.04},
        "O": {"VdW Radius": 150, "Sanderson Electronegativity": 3.654, "Polarizability": 0.802, "Pauling Electronegativity": 3.44},
        "F": {"VdW Radius": 146, "Sanderson Electronegativity": 4.000, "Polarizability": 0.557, "Pauling Electronegativity": 3.98},
        "Al": {"VdW Radius": 225, "Sanderson Electronegativity": 1.714, "Polarizability": 6.800, "Pauling Electronegativity": 1.61},
        "Si": {"VdW Radius": 219, "Sanderson Electronegativity": 2.138, "Polarizability": 5.380, "Pauling Electronegativity": 1.9},
        "P": {"VdW Radius": 190, "Sanderson Electronegativity": 2.515, "Polarizability": 3.630, "Pauling Electronegativity": 2.19},
        "S": {"VdW Radius": 189, "Sanderson Electronegativity": 2.957, "Polarizability": 2.900, "Pauling Electronegativity": 2.58},
        "Cl": {"VdW Radius": 182, "Sanderson Electronegativity": 3.475, "Polarizability": 2.180, "Pauling Electronegativity": 3.16},
        "Fe": {"VdW Radius": 244, "Sanderson Electronegativity": 2.200, "Polarizability": 8.400, "Pauling Electronegativity": 1.83},
        "Co": {"VdW Radius": 240, "Sanderson Electronegativity": 2.560, "Polarizability": 7.500, "Pauling Electronegativity": 1.88},
        "Ni": {"VdW Radius": 240, "Sanderson Electronegativity": 1.940, "Polarizability": 6.800, "Pauling Electronegativity": 1.91},
        "Cu": {"VdW Radius": 238, "Sanderson Electronegativity": 1.980, "Polarizability": 6.100, "Pauling Electronegativity": 1.9},
        "Zn": {"VdW Radius": 239, "Sanderson Electronegativity": 2.223, "Polarizability": 7.100, "Pauling Electronegativity": 1.65},
        "Br": {"VdW Radius": 186, "Sanderson Electronegativity": 3.219, "Polarizability": 3.050, "Pauling Electronegativity": 2.96},
        "Sn": {"VdW Radius": 242, "Sanderson Electronegativity": 1.490, "Polarizability": 7.700, "Pauling Electronegativity": 1.96},
        "I": {"VdW Radius": 204, "Sanderson Electronegativity": 2.778, "Polarizability": 5.350, "Pauling Electronegativity": 2.66},
        "Na": {"VdW Radius": 250, "Sanderson Electronegativity": 0.56, "Polarizability": 24.11, "Pauling Electronegativity": 0.93},
        "Ca": {"VdW Radius": 262, "Sanderson Electronegativity": 0.950, "Polarizability": 23.70, "Pauling Electronegativity": 1.00},
        "Se": {"VdW Radius": 182, "Sanderson Electronegativity": 3.01, "Polarizability": 4.45, "Pauling Electronegativity": 2.55},
        "K": {"VdW Radius": 273, "Sanderson Electronegativity": 0.45, "Polarizability": 43.40, "Pauling Electronegativity": 0.82},
        
    }
    
    if symbol in data_elec:
        return data_elec[symbol]
    else:
        print(f"Element '{symbol}' not found in the data.")
        
        
def get_atom_properties(atom):
    
    ## Atom Properties from RDKit
    atom_properties = [
        atom.GetAtomicNum(),               
        atom.GetDegree(), 
        atom.GetFormalCharge(),            
        atom.GetTotalNumHs(),  
        atom.GetExplicitValence(),         
        atom.GetImplicitValence(),         
        atom.GetNumExplicitHs(),           
        atom.GetNumImplicitHs(),           
        atom.GetMass(),
        Chem.rdchem.GetNumPiElectrons(atom)
    ]   
    atom_properties = [float(prop) for prop in atom_properties]
    
    ## Some other molecular properties including Electronegetivity
    atom_properties.extend(list(get_element_properties(atom.GetSymbol()).values()))
    
    return atom_properties


def get_properties_smiles(smiles):
    
    '''
        Given a single SMILES it gives the feature representation of that smiles
    '''
    if isinstance(smiles, str):
        mol = Chem.MolFromSmiles(smiles)
    else:
        mol=smiles
    ComputeGasteigerCharges(mol) 
    all_properties = [None]*mol.GetNumAtoms()
    if mol is None:
        print("Invalid SMILES string")
        return []

    # Iterate over atoms in the molecule and get properties
    for i, atom in enumerate(mol.GetAtoms()):
        atom_properties = get_atom_properties(atom)
        GasteigerCharge = float(atom.GetProp("_GasteigerCharge")) if atom.HasProp("_GasteigerCharge") else 0.
        atom_properties.append(GasteigerCharge)
        all_properties[i] = atom_properties
         
    return all_properties


def get_atom_info_vector(smiles,
                         atom_symbols,
                         hybridization_list,
                         chiraltypes,
                         bonds_list
                         ):

    mol = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()
    atom_properties_mol = [[]]*num_atoms
    hybridization_atoms_list = [None]*num_atoms
    symbol_list = [None]*num_atoms
    aromaticity_list = [None]*num_atoms
    ring_list = [None]*num_atoms
    chirality_list = [None]*num_atoms
    bond_info = [None]*num_atoms
    neigbor_symbol_info = [None]*num_atoms
    neighbour_info_list = [None]*num_atoms
    logP_list = [None]*num_atoms
    
    
    if mol is None:
        print("Invalid SMILES string")
        return []
    
    # Iterate over atoms in the molecule and get properties
    for i, atom in enumerate(mol.GetAtoms()):
        aromatic = [0., 0.]
        ring = [0., 0.]
        bonds_onehot = [0.]*len(bonds_list)
        neighbor_onehot = [0.]*len(atom_symbols)
        
        ## Atom Properties from RDKit
        onehot_symbol = list_to_onehot(atom.GetSymbol(), atom_symbols)
        symbol_list[i] = onehot_symbol
        
        ## Chirality Encoding
        chirality = list_to_onehot(str(atom.GetChiralTag()), chiraltypes)
        chirality_list[i] = chirality
        
        ## Hybridization Encoding
        hybridization = list_to_onehot(str(atom.GetHybridization()), hybridization_list)
        hybridization_atoms_list[i] = hybridization
        
        ## Aromaticity & Ring Information Encoding
        aromatic[int(atom.GetIsAromatic())] = 1.0 
        aromaticity_list[i] = aromatic
            
        ring[int(atom.IsInRing())] = 1.0
        ring_list[i] = ring
        
        logP_list[i] = [get_logP_from_csv(logp_dict=logP_dict, mol=mol, atom_idx=i)]
            
        ##Neighbour Symbols Encoding
        all_neighbours = [neighbor.GetSymbol() for neighbor in atom.GetNeighbors()]
        all_bonds = [str(bond.GetBondType()) for bond in atom.GetBonds()]
        bond_value_dict = dict({'SINGLE':1, 'DOUBLE':2, 'TRIPLE':3, 'AROMATIC':4})
        all_neighbours_counts = Counter(all_neighbours)
        temp_matrix = np.zeros((len(atom_symbols), len(bonds_list)))
        for j in range(len(all_neighbours)):
            temp_matrix[(atom_symbols.index(all_neighbours[j]), bonds_list.index(all_bonds[j]))] = bond_value_dict[all_bonds[j]]*float(all_neighbours_counts.get(all_neighbours[j]))
        neighbour_info_list[i] = list(temp_matrix.flatten())
        
        ## Bond Information Encoding
        all_bonds = [str(bond.GetBondType()) for bond in atom.GetBonds()]
        all_bonds_counts = Counter(all_bonds)
        bond_dict = {bond:all_bonds_counts.get(bond) for bond in all_bonds}
        for bond in all_bonds:
            bonds_onehot[bonds_list.index(bond)] = float(bond_dict[bond])
        bond_info[i] = bonds_onehot
            
        ##Neighbour Symbols Encoding
        all_neighbours = [neighbor.GetSymbol() for neighbor in atom.GetNeighbors()]
        all_neighbours_counts = Counter(all_neighbours)
        for neigh in all_neighbours:
            neighbor_onehot[atom_symbols.index(neigh)] = float(all_neighbours_counts.get(neigh))#1.0
        neigbor_symbol_info[i] = neighbor_onehot
            
   
    ## Atom 2D Properties 
    atom_properties_mol = get_properties_smiles(smiles)
     
    atom_feature_vector = {
        'symbol':np.array(symbol_list),
        'rdkit_2d_atom_prop':np.array(atom_properties_mol)[:, :10],
        'electronegetivity_infos':np.array(atom_properties_mol)[:, 10:-1],
        'gasteiger_charge':np.array(atom_properties_mol)[:, -1:],
        'atom_properties':np.array(atom_properties_mol),
        'hybridization':np.array(hybridization_atoms_list),
        'aromaticity':np.array(aromaticity_list),
        'ring':np.array(ring_list),
        'logP_values':np.array(logP_list),
        'chirality':np.array(chirality_list),
        'bond_info':np.array(bond_info),
        'neigbor_symbol_info':np.array(neigbor_symbol_info),
        'neighbour_info':np.array(neighbour_info_list)
    }
    
    return atom_feature_vector

def get_multirelational_bond_matrix(smiles, bonds_list, weighted_flag=True, include_self_loop = True):
    """
    Generates a multi-relational bond matrix for a given molecule.
    
    Args:
        smiles (str): SMILES representation of the molecule.
        bonds_list (list): List of bond types as strings, e.g., ["SINGLE", "DOUBLE"].
        weighted_flag (bool): Whether to assign numeric weights based on bond order.

    Returns:
        np.ndarray: A [num_relations, num_atoms, num_atoms] bond matrix.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    num_atoms = mol.GetNumAtoms()
    num_relations = len(bonds_list)

    if include_self_loop:
        bond_matrix = np.array([np.eye(num_atoms, num_atoms) for _ in range(num_relations)], dtype=float)
    else:
        bond_matrix = np.zeros((num_relations, num_atoms, num_atoms), dtype=float)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_type_str = str(bond.GetBondType())  # E.g., "SINGLE", "DOUBLE"
        if bond_type_str not in bonds_list:
            continue

        relation_idx = bonds_list.index(bond_type_str)
        
        # Use numeric weight (1.0, 2.0, etc.) or binary
        weight = float(bond.GetBondTypeAsDouble()) if weighted_flag else 1.0

        bond_matrix[relation_idx, i, j] = weight
        bond_matrix[relation_idx, j, i] = weight

    return torch.tensor(bond_matrix)


class Multirelational_GraphDataset(Dataset):
    def __init__(self, smi_list, labels, max_num_atoms, atom_symbols, hybridization_list, chiraltypes, bonds_list, bond_weight_flag=False, include_self_loop = True, **kwargs):
        self.smi_list = smi_list
        self.labels = labels
        self.max_num_atoms = max_num_atoms
        self.atom_symbols = atom_symbols
        self.hybridization_list = hybridization_list
        self.chiraltypes = chiraltypes
        self.bonds_list = bonds_list
        self.num_relations = len(bonds_list)
        self.bond_weight_flag = bond_weight_flag
        self.include_self_loop = include_self_loop
    
    def __len__(self):
        return len(self.smi_list)

    def _get_one(self, idx):
        
        smi = self.smi_list[idx]
        atom_feature_vector, adjacency_tensor, degree_tensor = self.smi2feature(smi)
        y  = self.labels[idx]
        adjacency_tensor = adjacency_tensor.cpu().numpy()
        degree_tensor = degree_tensor.cpu().numpy()
        
        return smi, atom_feature_vector, adjacency_tensor, degree_tensor, y
    
    def __getitem__(self, idx):
        if isinstance(idx, (slice, list, tuple)):  
            if isinstance(idx, slice):
                indices = range(*idx.indices(len(self)))
                print(len(self))
            else:
                indices = idx
            items = [self._get_one(i) for i in indices]
            return self.collate_fn(items)
        return self._get_one(idx)
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate a list of (smi, atom_feature_dict, adjacency_matrix, degree_matrix, y)
        into numpy arrays (safe for multiprocessing).
        """
        smi_list, atom_feature_dicts, adjacency_tensors, degree_tensors, ys = zip(*batch)

        smi_list = list(smi_list)

        feature_keys = atom_feature_dicts[0].keys()
        atom_feature_vectors = {}
        for key in feature_keys:
            try:
                atom_feature_vectors[key] = np.stack([afd[key] for afd in atom_feature_dicts], axis=0).astype(np.float32)
            except ValueError:
                atom_feature_vectors[key] = [afd[key].astype(np.float32) for afd in atom_feature_dicts]

        adjacency_tensors = np.array(adjacency_tensors, dtype=np.float32)
        degree_tensors    = np.array(degree_tensors, dtype=np.float32)
        ys                = np.array(ys, dtype=np.float32)
        return_vals = {'smiles':smi_list,
                       'atom_feature_vectors':atom_feature_vectors,
                       'adjacency_tensors':adjacency_tensors,
                       'degree_tensors':degree_tensors,
                       'labels':ys,
                       }
        
        return return_vals
    
    def smi2feature(self, smi):
        
        atom_feature_vector = get_atom_info_vector(smi,
                                                atom_symbols=self.atom_symbols,
                                                hybridization_list=self.hybridization_list,
                                                chiraltypes=self.chiraltypes,
                                                bonds_list=self.bonds_list
                                                )
        adj_matrix_temp = get_multirelational_bond_matrix(smi,
                                                        self.bonds_list,
                                                        weighted_flag=self.bond_weight_flag,
                                                        include_self_loop = self.include_self_loop
                                                        )
        adjacency_tensor = torch.zeros((self.num_relations, self.max_num_atoms, self.max_num_atoms))
        adjacency_tensor[:adj_matrix_temp.shape[0], :adj_matrix_temp.shape[1], :adj_matrix_temp.shape[2]] = adj_matrix_temp # A = A + I
        degree_tensor = torch.stack([torch.diag(vector) for vector in adjacency_tensor.sum(axis=1)]) # D = D + I
        
        return atom_feature_vector, adjacency_tensor, degree_tensor
    
    @staticmethod
    def smiles2GcnDictData(
                    smi,
                    atom_symbols,
                    hybridization_list,
                    chiraltypes,
                    bonds_list,
                    max_num_atoms,
                    bond_weight_flag=False,
                    include_self_loop = True,
                    *args, **kwargs
                    ):
        num_relations = len(bonds_list)
        atom_feature_vector = get_atom_info_vector(smi,
                                                    atom_symbols=atom_symbols,
                                                    hybridization_list=hybridization_list,
                                                    chiraltypes=chiraltypes,
                                                    bonds_list=bonds_list
                                                    )

        adj_matrix_temp = get_multirelational_bond_matrix(
                                                        smi,
                                                        bonds_list,
                                                        weighted_flag=bond_weight_flag,
                                                        include_self_loop=include_self_loop
                                                        )
        adjacency_tensor = torch.zeros((num_relations, max_num_atoms, max_num_atoms))
        adjacency_tensor[ :adj_matrix_temp.shape[0], :adj_matrix_temp.shape[1], :adj_matrix_temp.shape[2]] = adj_matrix_temp
        degree_tensor = torch.stack([torch.diag(vector) for vector in adjacency_tensor.sum(axis=1)])
        
        return_vals = {'SMILES': smi,
                       'atom_feature_vector': atom_feature_vector,
                       'adjacency_matrix': adjacency_tensor.numpy(),
                       'degree_matrix': degree_tensor.numpy(),
                       }

        return return_vals


class GraphData_from_pickle(Dataset):
    
    def __init__(self,
                 dataset_path,
                 features_list = ['symbol', 'atom_properties', 'hybridization', 'aromaticity', 'ring', 'chirality', '3d_descriptors', 'neighbour_info'],
                 max_num_atoms=65,
                 padding=True
                 ):
        self.features_list = features_list
        self.max_num_atoms = max_num_atoms
        self.padding = padding
        if dataset_path.endswith('.pkl'):
            with open(dataset_path, 'rb') as fp:
                self.dataset = pickle.load(fp)
        if dataset_path.endswith('.pkl.gz'):
            with gzip.open(dataset_path, 'rb') as fp:
                self.dataset = pickle.load(fp)
        
        filename = '/'.join(dataset_path.split('/')[:-1]+['dataset_mean.pkl'])
        with open(filename, 'rb') as fp:
            self.dataset_mean = pickle.load(fp)
        filename = '/'.join(dataset_path.split('/')[:-1]+['dataset_std.pkl'])
        with open(filename, 'rb') as fp:
            self.dataset_std = pickle.load(fp)
            
    def __len__(self):
        return len(self.dataset)

    def _get_one(self, idx):
        
        data=self.dataset[idx]
        feature_vector, adjacency_matrix, degree_matrix, y = self.smi2feature(data)
        
        return feature_vector, adjacency_matrix, degree_matrix, y
    
    def __getitem__(self, idx):
        if isinstance(idx, (slice, list, tuple)):  
            if isinstance(idx, slice):
                indices = range(*idx.indices(len(self)))
            else:
                indices = idx
            items = [self._get_one(i) for i in indices]
            return self.collate_fn(items)
        return self._get_one(idx)
    
    def collate_fn(self, batch):
        """
        Collate a list of (feature_vector, adjacency_matrix, degree_matrix, y)
        into batched tensors.
        """
        feature_vectors, adjacency_matrices, degree_matrices, ys = zip(*batch)

        feature_vectors   = [torch.as_tensor(fv, dtype=torch.float32) for fv in feature_vectors]
        adjacency_matrices = [torch.as_tensor(adj, dtype=torch.float32) for adj in adjacency_matrices]
        degree_matrices    = [torch.as_tensor(deg, dtype=torch.float32) for deg in degree_matrices]
        ys                 = [torch.as_tensor(y, dtype=torch.float32) for y in ys]

        try:
            feature_vectors = torch.stack(feature_vectors, dim=0)
        except RuntimeError:
            pass  

        try:
            adjacency_matrices = torch.stack(adjacency_matrices, dim=0)
        except RuntimeError:
            pass

        try:
            degree_matrices = torch.stack(degree_matrices, dim=0)
        except RuntimeError:
            pass

        ys = torch.stack(ys, dim=0)

        return feature_vectors, adjacency_matrices, degree_matrices, ys
    
    def smi2feature(self, data):
        
        atom_feature_vector = data['atom_feature_vector']
        try:
            features = [atom_feature_vector[k] for k in self.features_list]
            mean = [self.dataset_mean[k] for k in self.features_list]
            std = [self.dataset_std[k] for k in self.features_list]
        except:
            raise AttributeError(f'Features should be in: {list(atom_feature_vector.keys())}')
        
        try:
            atom_feature_vector = np.concatenate(features, axis=1)
        except:
            raise ValueError(f'Error found in: SMILES - {data['SMILES']}, Features - {atom_feature_vector}')
        
         # Convert to tensor
        atom_feature_vector = torch.from_numpy(atom_feature_vector).float()
        dataset_mean = torch.from_numpy(np.concatenate(mean)).float()
        dataset_std  = torch.from_numpy(np.concatenate(std)).float()
        if self.padding:
            feature_vector = torch.zeros((self.max_num_atoms, atom_feature_vector.shape[1]), dtype=torch.float32)
            feature_vector[:atom_feature_vector.shape[0], :] = (atom_feature_vector - dataset_mean) / dataset_std
        else:
            feature_vector = (atom_feature_vector - dataset_mean) / dataset_std
        
        adjacency_matrix = torch.from_numpy(data['adjacency_matrix']).float()
        degree_matrix    = torch.from_numpy(data['degree_matrix']).float()
        y                = torch.tensor(data['label'], dtype=torch.float32)
        
        return feature_vector, adjacency_matrix, degree_matrix, y
    
    def get_smiles(self, idx):
        smi = self.dataset[idx]['SMILES']
        return smi
    
    def get_atom_info_vector(self, idx):
        
        data=self.dataset[idx]
        atom_feature_vector = data['atom_feature_vector']
        
        return atom_feature_vector
    
    @staticmethod
    def GcnDictData2GcnInput(data,
                            dataset_mean,
                            dataset_std,
                            max_num_atoms,
                            features_list=['symbol', 'atom_properties', 'hybridization', 'aromaticity', 'ring', 'chirality', 'neighbour_info'], #'3d_descriptors'
                            padding=True
                            ):
        atom_feature_vector = data['atom_feature_vector']

        try:
            features = [atom_feature_vector[k] for k in features_list]
            mean     = [dataset_mean[k] for k in features_list]
            std      = [dataset_std[k] for k in features_list]
        except KeyError:
            raise AttributeError(f'Features should be in: {list(atom_feature_vector.keys())}')

        try:
            atom_feature_vector = np.concatenate(features, axis=1)
        except Exception as e:
            raise ValueError(f"Error found in: SMILES - {data.get('SMILES', 'UNKNOWN')}, "f"Features - {atom_feature_vector}") from e

        atom_feature_vector = torch.from_numpy(atom_feature_vector).float()
        dataset_mean = torch.from_numpy(np.concatenate(mean)).float()
        dataset_std  = torch.from_numpy(np.concatenate(std)).float()

        if padding:
            feature_vector = torch.zeros((max_num_atoms, atom_feature_vector.shape[1]), dtype=torch.float32)
            feature_vector[:atom_feature_vector.shape[0], :] = ((atom_feature_vector - dataset_mean) / dataset_std)
        else:
            feature_vector = (atom_feature_vector - dataset_mean) / dataset_std

        adjacency_matrix = torch.from_numpy(data['adjacency_matrix']).float()
        degree_matrix    = torch.from_numpy(data['degree_matrix']).float()

        return feature_vector, adjacency_matrix, degree_matrix