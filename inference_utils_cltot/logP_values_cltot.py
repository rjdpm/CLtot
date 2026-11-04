from rdkit import Chem
from rdkit.Chem import rdmolops
from collections import Counter

__all__ = [
    'Atom_logP_Value',
    'smiles2logP',
    'get_logP_from_csv'
]

zero_func = lambda x, y: 0

def is_atom_lipo(mol, atom_idx):
    """
    Check if an atom is separated by at least three single bonds from any unsaturated carbon 
    or hetero-atom in the molecule.
    
    Args:
        mol (rdkit.Chem.Mol): The molecule.
        atom_idx (int): Index of the atom to check.
        
    Returns:
        bool: True if the condition is met, False otherwise.
    """
    # Get molecular graph and bond information
    unsaturated_carbons = set()
    hetero_atoms = set()
    distance_matrix = rdmolops.GetDistanceMatrix(mol)
    
    for bond in mol.GetBonds():
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        
        if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
            # Identify unsaturated carbons
            for idx in [start_idx, end_idx]:
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetAtomicNum() == 6:  # Carbon
                    unsaturated_carbons.add(idx)
        
        for atom in [mol.GetAtomWithIdx(start_idx), mol.GetAtomWithIdx(end_idx)]:
            if atom.GetAtomicNum() != 6 and atom.GetAtomicNum() != 1:  # Hetero-atom
                hetero_atoms.add(atom.GetIdx())
    
    # Combine unsaturated carbons and hetero-atoms
    target_atoms = unsaturated_carbons.union(hetero_atoms)
    
    # Compute shortest path distances using GetDistanceMatrix
    for target_idx in target_atoms:
        if distance_matrix[atom_idx, target_idx] <= 3:
            return False  
    
    return True 


def is_nitrogen_in_amide(mol, atom_idx):
    
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetSymbol()!= 'N':
        return False
    
    # Define the amide substructure pattern
    amide_pattern = Chem.MolFromSmarts("[NX3][CX3](=[OX1])[#6]")
    matches = mol.GetSubstructMatches(amide_pattern)
    for match in matches:
        if atom_idx in match:
            return True
    return False


def has_heteroatom_bond(atom):
    
    if atom.GetSymbol()!= 'C' and atom.GetSymbol()!= 'H':
        return True
    else:
        neighbours = [x.GetSymbol() for x in atom.GetNeighbors()]
        if set(neighbours)-{'C', 'H'}:
            return True
        else:
            return False


def neighbor_atoms_with_specific_bonds(neighbors, corresponding_bonds, bond_type = 'SINGLE'):
    
    indices = [i for i, x in enumerate(corresponding_bonds) if x == bond_type]
    bond_atoms = [neighbors[i] for i in indices]
    if bond_type == 'TRIPLE':
        bond_atoms = ['X' if y != 'C' and y != 'H' and y != 'N' else y for y in bond_atoms]
    else:
        bond_atoms = ['X' if y != 'C' and y != 'H' else y for y in bond_atoms]
    bond_atoms = tuple(bond_atoms)
    
    return bond_atoms


def sp3_carbon_logP(mol, atom_idx):
    
    atom = mol.GetAtomWithIdx(atom_idx)
    nH = atom.GetTotalNumHs(includeNeighbors=True)
    lipo = int(is_atom_lipo(mol, atom_idx))
    pi = 1 if Chem.rdchem.GetNumPiElectrons(atom) != 0 else 0
    has_heteroatom = int(has_heteroatom_bond(atom))
    
    # Dictionary mapping (nH, lipo, pi, has_heteroatom) to logP values
    logP_values = {
        (3, 1, 0, 0):  0.7896, # C.3.3h.lipo
        (3, 0, 1, 1): -0.0753, # C.3.3h.X.pi
        (3, 0, 0, 1):  0.0402, # C.3.3h.X
        (3, 0, 1, 0):  0.5018, # C.3.3h.pi
        (3, 0, 0, 0):  0.5240, # C.3.3h
        (2, 1, 0, 0):  0.5201, # C.3.2h.lipo
        (2, 0, 1, 1): -0.2441, # C.3.2h.X.pi
        (2, 0, 0, 1): -0.0821, # C.3.2h.X
        (2, 0, 1, 0):  0.2718, # C.3.2h.pi
        (2, 0, 0, 0):  0.3436, # C.3.2h
        (1, 0, 1, 1): -0.3711, # C.3.h.X.pi
        (1, 0, 0, 1): -0.1426, # C.3.h.X
        (1, 1, 0, 0):  0.1485, # C.3.h.lipo
        (1, 0, 1, 0):  0.0841, # C.3.h.pi
        (1, 0, 0, 0):  0.1485, # C.3.h
        (0, 0, 1, 1): -0.5475, # C.3.X.pi
        (0, 0, 0, 1): -0.4447, # C.3.X
        (0, 0, 1, 0):  0.0885, # C.3.pi
        (0, 1, 0, 0):  0.0596, # C.3.lipo
        (0, 0, 0, 0):  0.0596, # C.3
        (4, 1, 0, 0):  0.0596, # C.3.4h.lipo
    }
    # print((nH, lipo, pi, has_heteroatom))
    key = (nH, lipo, pi, has_heteroatom)
    
    # Return the logP value based on the key
    return logP_values.get(key, key)  # Default to 0.0 if no match is found


def aromatic_carbon_logP(mol, atom_idx):
    
    mol = Chem.AddHs(mol)
    atom = mol.GetAtomWithIdx(atom_idx)
    neighbors = [x.GetSymbol() for x in atom.GetNeighbors()]
    bonds = [str(bond.GetBondType()) for bond in atom.GetBonds()]
    
    aromatic_bond_atoms = neighbor_atoms_with_specific_bonds(neighbors=neighbors, corresponding_bonds=bonds, bond_type='AROMATIC')
    bond_counter = Counter(bonds)
    num_double_bond = bond_counter.get('AROMATIC')
    if num_double_bond == 3:   # A≈C*(≈A)≈A
        return 0.3158
    
    single_bond_atoms = neighbor_atoms_with_specific_bonds(neighbors=neighbors, corresponding_bonds=bonds, bond_type='SINGLE')
    logP_values = {
        (frozenset({'H'}), frozenset({'C', 'X'})) : -0.1039,  # R≈C*(-H)≈X
        (frozenset({'H'}), frozenset({'X', 'X'})) : -0.1039,  # X≈C*(-H)≈X
        (frozenset({'H'}), frozenset({'C', 'C'})) :  0.3157,  # R≈C*(-H)≈R
        (frozenset({'X'}), frozenset({'C', 'X'})) : -0.1003,  # R≈C*(-X)≈X
        (frozenset({'X'}), frozenset({'X', 'X'})) : -0.1003,  # X≈C*(-X)≈X
        (frozenset({'X'}), frozenset({'C', 'C'})) : -0.0112,  # R≈C*(-X)≈R
        (frozenset({'C'}), frozenset({'C', 'X'})) : -0.1874,  # R≈C*(-R)≈X
        (frozenset({'C'}), frozenset({'X', 'X'})) : -0.1874,  # X≈C*(-R)≈X
        (frozenset({'C'}), frozenset({'C', 'C'})) :  0.1911,  # R≈C*(-R)≈R
    }
    key = (single_bond_atoms, aromatic_bond_atoms)
    
    # Return the logP value based on the key
    return logP_values.get(key, key)  # Default to 0.0 if no match is found


def sp2_carbon_logP(mol, atom_idx):
    
    mol = Chem.AddHs(mol)
    atom = mol.GetAtomWithIdx(atom_idx)
    ring = int(atom.IsInRing())
    neighbors = [x.GetSymbol() for x in atom.GetNeighbors()]
    bonds = [str(bond.GetBondType()) for bond in atom.GetBonds()]
    
    single_bond_atoms = neighbor_atoms_with_specific_bonds(neighbors=neighbors, corresponding_bonds=bonds, bond_type='SINGLE')
    double_bond_atoms = neighbor_atoms_with_specific_bonds(neighbors=neighbors, corresponding_bonds=bonds, bond_type='DOUBLE')
    
    logP_values = {
        (frozenset({'C'}), frozenset({'H', 'H'}), 0): 0.5977,  # H-C*(=R)-H
        (frozenset({'X'}), frozenset({'H', 'H'}), 0): 0.5977,  # H-C*(=X)-H
        (frozenset({'C'}), frozenset({'H', 'X'}), 0): -0.0967,  # H-C*(=C)-X
        (frozenset({'C'}), frozenset({'H', 'C'}), 1): 0.4004,  # H-C*(=C)-R in a ring
        (frozenset({'C'}), frozenset({'H', 'C'}), 0): 0.3214,  # H-C*(=C)-R
        (frozenset({'X'}), frozenset({'H', 'C'}), 0): -0.8756,  # H-C*(=X)-R
        (frozenset({'X'}), frozenset({'H', 'X'}), 0): -0.8756,  # H-C*(=X)-X
        (frozenset({'X'}), frozenset({'H', 'H'}), 0): -0.8756,  # H-C*(=X)-H
        (frozenset({'C'}), frozenset({'X', 'C'}), 0): -0.2069,  # R-C*(=C)-X
        (frozenset({'C'}), frozenset({'X', 'X'}), 0): -0.2069,  # X-C*(=C)-X
        (frozenset({'C'}), frozenset({'C', 'C'}), 1): -0.2084,  # R-C*(=C)-R in a ring
        (frozenset({'C'}), frozenset({'C', 'C'}), 0): 0.4840,  # R-C*(=C)-R
        (frozenset({'X'}), frozenset({'C', 'X'}), 0): -0.8076,  # R-C*(=X)-X
        (frozenset({'X'}), frozenset({'X', 'X'}), 0): -0.8076,  # X-C*(=X)-X
        (frozenset({'X'}), frozenset({'C', 'C'}), 1): -0.5304,  # R-C*(=X)-R in a ring
        (frozenset({'X'}), frozenset({'C', 'C'}), 0): -0.6093,  # R-C*(=X)-R
    }
    # print((double_bond_atoms, single_bond_atoms, ring))
    key = (double_bond_atoms, single_bond_atoms, ring)
    
    # Return the logP value based on the key
    return logP_values.get(key, key)  # Default to 0.0 if no match is found


def sp_carbon_logP(mol, atom_idx):

    atom = mol.GetAtomWithIdx(atom_idx)
    neighbors = [x.GetSymbol() for x in atom.GetNeighbors()]
    bonds = [str(bond.GetBondType()) for bond in atom.GetBonds()]
    
    double_bond_atoms = neighbor_atoms_with_specific_bonds(neighbors=neighbors, corresponding_bonds=bonds, bond_type='DOUBLE')
    triple_bond_atoms = neighbor_atoms_with_specific_bonds(neighbors=neighbors, corresponding_bonds=bonds, bond_type='TRIPLE')
    logP = (double_bond_atoms, triple_bond_atoms)
    
    if len(triple_bond_atoms):
        logP = 0.1945
    else:
        bond_counter = Counter(bonds)
        num_double_bond = bond_counter.get('DOUBLE')
        if num_double_bond == 2:
            logP = -0.5879 
        
    return logP


def carbon_atom_logP(mol, atom_idx):
    
    atom = mol.GetAtomWithIdx(atom_idx)
    hybridization = str(atom.GetHybridization())
    isaromatic = int(atom.GetIsAromatic())
    logP_values_func = {
            'SP3' : sp3_carbon_logP,
            'SP2' : sp2_carbon_logP,
            'SP'  : sp_carbon_logP,
        }
    if isaromatic:
        logP =  aromatic_carbon_logP(mol, atom_idx)
        if not logP:
            logP = logP_values_func.get(hybridization, zero_func)(mol, atom_idx)
    else:
        logP = logP_values_func.get(hybridization, zero_func)(mol, atom_idx)
        
    return logP


def sp3_nitrogen_logP(mol, atom_idx):
    
    atom = mol.GetAtomWithIdx(atom_idx)
    nH = atom.GetTotalNumHs(includeNeighbors=True)
    pi = 1 if Chem.rdchem.GetNumPiElectrons(atom) != 0 else 0
    isinamide = int(is_nitrogen_in_amide(mol, atom_idx))
    
    logP_values = { # (nH, pi, isinamide)
        (2, 0, 1) : -0.6414, # A-N*H2 in an amide group (N.am.2h)
        (2, 1, 0) : -0.3637, # A-N*H2 connected to a conjugated moiety (N.3.2h.pi)
        (2, 0, 0) : -0.7445, # A-N*H2
        (1, 0, 1) : -0.3333, # A2-N*H in an amide group (N.am.h)
        (1, 1, 0) :  0.2172, # A2-N*H connected to a conjugated (N.3.h.pi)
        (1, 0, 0) : -0.2610, # A2-N*H (N.3.h)
        (0, 0, 1) : -0.1551, # A3N*in an amide group 
        (0, 1, 0) :  0.3776, # A3N*connected to a conjugated moiety 
        (0, 0, 0) :  0.1799, # A3N* 
    }
    key = (nH, pi, isinamide)
    
    # Return the logP value based on the key
    return logP_values.get(key, key)  # Default to 0.0 if no match is found


def aromatic_nitrogen_logP(mol, atom_idx):
    
    mol = Chem.AddHs(mol)
    atom = mol.GetAtomWithIdx(atom_idx)
    neighbors = [x.GetSymbol() for x in atom.GetNeighbors()]
    bonds = [str(bond.GetBondType()) for bond in atom.GetBonds()]
    
    rings_info = mol.GetRingInfo()
    ring_size = max(rings_info.AtomRingSizes(atom_idx))
    
    single_bond_atoms = neighbor_atoms_with_specific_bonds(neighbors=neighbors, corresponding_bonds=bonds, bond_type='SINGLE')
    aromatic_bond_atoms = neighbor_atoms_with_specific_bonds(neighbors=neighbors, corresponding_bonds=bonds, bond_type='AROMATIC')
    
    logP_values = {
        (frozenset()     , frozenset({'X', 'X'}), 6) : -0.2167,  # X≈N*≈X in a 6-member ring
        (frozenset()     , frozenset({'C', 'X'}), 6) : -0.2974,  # R≈N*≈X in a 6-member ring
        (frozenset()     , frozenset({'C', 'C'}), 6) :  0.0888,  # R≈N*≈R in a 6-member ring
        (frozenset({'H'}), frozenset({'C', 'X'}), 5) :  0.3675,  # A≈N*(-H)≈X in a 5-member ring (R≈N*(-H)≈X)
        (frozenset({'H'}), frozenset({'X', 'X'}), 5) :  0.3675,  # A≈N*(-H)≈X in a 5-member ring (X≈N*(-H)≈X)
        (frozenset({'H'}), frozenset({'C', 'C'}), 5) :  0.2364,  # R≈N*(-H)≈R in a 5-member ring
        (frozenset()     , frozenset({'X', 'X'}), 5) :  1.1022,  # X≈N*≈X in a 5-member ring
        (frozenset()     , frozenset({'C', 'X'}), 5) :  0.4854,  # R≈N*≈X in a 5-member ring
        (frozenset()     , frozenset({'C', 'C'}), 5) :  0.3181,  # R≈N*≈R in a 5-member ring
    }
    
    # print((single_bond_atoms, aromatic_bond_atoms, ring_size))
    key = (single_bond_atoms, aromatic_bond_atoms, ring_size)
    
    # Return the logP value based on the key
    return logP_values.get(key, key)  # Default to 0.0 if no match is found


def sp2_nitrogen_logP(mol, atom_idx):
    
    atom = mol.GetAtomWithIdx(atom_idx)
    nH = atom.GetTotalNumHs(includeNeighbors=True)
    neighbors = [x.GetSymbol() for x in atom.GetNeighbors()]
    bonds = [str(bond.GetBondType()) for bond in atom.GetBonds()]
    ring = int(atom.IsInRing())
    
    double_bond_atoms = neighbor_atoms_with_specific_bonds(neighbors=neighbors, corresponding_bonds=bonds, bond_type='DOUBLE')
    
    logP_values = {
        (1, frozenset({'C'}), 0) : 0.6927, # H-N*=A (H-N*=R)
        (1, frozenset({'X'}), 0) : 0.6927, # H-N*=A (H-N*=X)
        (0, frozenset({'C'}), 1) : 0.7974, # A-N*(=C) in a ring
        (0, frozenset({'C'}), 0) : 0.9794, # A-N*(=C)
        (0, frozenset({'X'}), 0) : 0.2698, # A-N*(=X)
    }
    
    # print((nH, double_bond_atoms, ring))
    key = (nH, double_bond_atoms, ring)
    
    # Return the logP value based on the key
    return logP_values.get(key, key)  # Default to 0.0 if no match is found


def sp_nitrogen_logP(mol, atom_idx):
    
    logP = None
    atom = mol.GetAtomWithIdx(atom_idx)
    neighbors = [x.GetSymbol() for x in atom.GetNeighbors()]
    bonds = [str(bond.GetBondType()) for bond in atom.GetBonds()]
    triple_bond_atoms = neighbor_atoms_with_specific_bonds(neighbors=neighbors, corresponding_bonds=bonds, bond_type='TRIPLE')
    logP = (triple_bond_atoms, bonds)
    if len(triple_bond_atoms):
        logP = 0.0337
    else:
        bond_counter = Counter(bonds)
        num_double_bond = bond_counter.get('DOUBLE')
        if num_double_bond == 2:
            logP =  0.5339  
    
    return logP

    
def nitrogen_atom_logP(mol, atom_idx):
    
    atom = mol.GetAtomWithIdx(atom_idx)
    isaromatic = int(atom.GetIsAromatic())
    hybridization = str(atom.GetHybridization())
    logP_values_func = {
        'SP3' : sp3_nitrogen_logP,
        'SP2' : sp2_nitrogen_logP,
        'SP' : sp_nitrogen_logP
    }
    if isaromatic:
        logP =  aromatic_nitrogen_logP(mol, atom_idx)
        if not logP:
            logP = logP_values_func.get(hybridization, zero_func)(mol, atom_idx)
    else:
        logP = logP_values_func.get(hybridization, zero_func)(mol, atom_idx)
        
    return logP


def sp3_oxygen_logP(mol, atom_idx):
    
    atom = mol.GetAtomWithIdx(atom_idx)
    
    isaromatic = int(atom.GetIsAromatic())
    if isaromatic:   # Oxygen atom in an aromatic ring
        return 0.5238
    
    nH = atom.GetTotalNumHs(includeNeighbors=True)
    pi = 1 if Chem.rdchem.GetNumPiElectrons(atom) != 0 else 0
    logP_values = {
        (1, 1) : -0.0381,    # A-O*H connected to a conjugated moiety (O.3.h.pi)
        (1, 0) : -0.4802,    # A-O*H (O.3.h)
        (0, 1) :  0.2701,    # A-O*-A connected to a conjugated moiety (O.3.pi )
        (0, 0) :  0.0059,    # A-O*-A
    }
    key = (nH, pi)
    
    # Return the logP value based on the key
    return logP_values.get(key, key)  # Default to 0.0 if no match is found


def sp2_oxygen_logP(mol, atom_idx):
    
    atom = mol.GetAtomWithIdx(atom_idx)
    neighbors = [x.GetSymbol() for x in atom.GetNeighbors()]
    bonds = [str(bond.GetBondType()) for bond in atom.GetBonds()]
    double_bond_atoms = neighbor_atoms_with_specific_bonds(neighbors=neighbors, corresponding_bonds=bonds, bond_type='DOUBLE')
    
    logP_values = {
        frozenset({'C'}) :  0.7148, # -C=O*
        frozenset({'X'}) : -0.5411, # -X=O*
    }

    key = double_bond_atoms
    
    # Return the logP value based on the key
    return logP_values.get(key, key)  # Default to 0.0 if no match is found


def oxygen_atom_logP(mol, atom_idx):
    
    atom = mol.GetAtomWithIdx(atom_idx)
    hybridization = str(atom.GetHybridization())
    logP_values_func = {
            'SP3' : sp3_oxygen_logP,
            'SP2' : sp2_oxygen_logP,
            }
        
    logP = logP_values_func.get(hybridization, zero_func)(mol, atom_idx)
    
    return logP


def sp3_sulfur_logP(mol, atom_idx):
    
    atom = mol.GetAtomWithIdx(atom_idx)
    isaromatic = int(atom.GetIsAromatic())
    if isaromatic:   # Oxygen atom in an aromatic ring
        return 1.1715
    
    neighbors = [x.GetSymbol() for x in atom.GetNeighbors()]
    bonds = [str(bond.GetBondType()) for bond in atom.GetBonds()]
    single_bond_atoms = neighbor_atoms_with_specific_bonds(neighbors=neighbors, corresponding_bonds=bonds, bond_type='SINGLE')
    
    if 'H' in single_bond_atoms:
        return 0.4927
    
    logP_values = {
        frozenset({'C', 'X'}) : 0.4125, # R-S*-X
        frozenset({'X', 'X'}) : 0.4125, # X-S*-X
        frozenset({'C', 'C'}) : 0.8300, # A-S*-A (R-S*-R)
    }
    key = single_bond_atoms
    
    # Return the logP value based on the key
    return logP_values.get(key, key)  # Default to 0.0 if no match is found


def sp2_sulfur_logP(mol, atom_idx):
    
    atom = mol.GetAtomWithIdx(atom_idx)
    neighbors = [x.GetSymbol() for x in atom.GetNeighbors()]
    bonds = [str(bond.GetBondType()) for bond in atom.GetBonds()]
    double_bond_atoms = neighbor_atoms_with_specific_bonds(neighbors=neighbors, corresponding_bonds=bonds, bond_type='DOUBLE')
    
    logP_values = {
        frozenset({'C'}) : 1.3544, # -C=S*
        frozenset({'X'}) : 1.2218, # -X=S*
    }
    key = double_bond_atoms

    # Return the logP value based on the key
    return logP_values.get(key, key)  # Default to 0.0 if no match is found


def sulfur_atom_logP(mol, atom_idx):
    
    atom = mol.GetAtomWithIdx(atom_idx)
    neighbors = [x.GetSymbol() for x in atom.GetNeighbors()]
    bonds = [str(bond.GetBondType()) for bond in atom.GetBonds()]
    
    indices = [i for i, x in enumerate(bonds) if x == 'DOUBLE']
    double_bond_atoms = [neighbors[i] for i in indices]
    
    if 'O' in double_bond_atoms:
        bond_counter = Counter(double_bond_atoms)
        num_oxygen = bond_counter.get('O')
        if num_oxygen == 2:
            return 0.5729
        elif num_oxygen == 1:
            return 0.0525
        else:
            return 0.0
        
    else:
        hybridization = str(atom.GetHybridization())
        
        logP_values_func = {
            'SP3' : sp3_sulfur_logP,
            'SP2' : sp2_sulfur_logP,
            }
        
        logP = logP_values_func.get(hybridization, zero_func)(mol, atom_idx)
        
        return logP
    
    
def logP_values_HPFIBrCl(mol, atom_idx):
    
    atom = mol.GetAtomWithIdx(atom_idx)
    symbol = atom.GetSymbol()
    pi = 1 if Chem.rdchem.GetNumPiElectrons(atom) != 0 else 0
    
    logP_values = {
        ('H', 0)  :  0.6123, # H 
        ('P', 1)  : -0.6694, # A3-P*(=A)
        ('P', 0)  : -0.6694, # A3-P*(=A)
        ('F', 1)  :  0.4401, # F.pi
        ('F', 0)  :  0.5360, # F
        ('Cl', 1) :  0.9610, # Cl.pi 
        ('Cl', 0) :  0.8036, # Cl
        ('Br', 1) :  1.0295, # Br.pi 
        ('Br', 0) :  0.9664, # Br
        ('I', 1)  :  0.7801, # I.pi 
        ('I', 0)  :  0.9071, # I.pi 
    }
    key = (symbol, pi)
    
    # Return the logP value based on the key
    return logP_values.get(key, key)  # Default to 0.0 if no match is found
                

def Atom_logP_Value(mol, atom_idx):
    
    atom = mol.GetAtomWithIdx(atom_idx)
    symbol = atom.GetSymbol()
    if symbol == 'C':
        logP = carbon_atom_logP(mol, atom_idx)
    elif symbol == 'N':
        logP = nitrogen_atom_logP(mol, atom_idx)
    elif symbol == 'O':
        logP = oxygen_atom_logP(mol, atom_idx)
    elif symbol == 'S':
        logP = sulfur_atom_logP(mol, atom_idx)
    elif symbol in ['H', 'P', 'F', 'Cl', 'Br', 'I']:
        logP = logP_values_HPFIBrCl(mol, atom_idx)
    else:
        raise ValueError(f'Atom Symbol: {symbol} is not in predefined list.')
    
    return logP


def smiles2logP(smiles):
    
    mol = Chem.MolFromSmiles(smiles)
    logP_values = [None]*mol.GetNumAtoms()
    all_atoms = mol.GetAtoms()
    for i in range(len(all_atoms)):
        logP_values[i] = Atom_logP_Value(mol, i)
        
    return logP_values


def get_logP_from_csv(logp_dict, mol, atom_idx):
    
    atom = mol.GetAtomWithIdx(atom_idx)
    
    symbol = atom.GetSymbol()
    hybridization = str(atom.GetHybridization())
    nH = atom.GetTotalNumHs(includeNeighbors=True)
    lipo = is_atom_lipo(mol, atom_idx)
    pi = True if Chem.rdchem.GetNumPiElectrons(atom) else False
    has_heteroatom = has_heteroatom_bond(atom)
    
    
    isinamide = is_nitrogen_in_amide(mol, atom_idx)
    isinring = atom.IsInRing()
    isaromatic = atom.GetIsAromatic()
    isconjugated = Chem.rdmolops.AtomHasConjugatedBond(atom)
    
    neighbors = [x.GetSymbol() for x in atom.GetNeighbors()]
    bonds = [str(bond.GetBondType()) for bond in atom.GetBonds()]
    bond_counter = Counter(bonds)
    num_single_bond = bond_counter.get('SINGLE', 0)
    num_double_bond = bond_counter.get('DOUBLE', 0)
    num_triple_bond = bond_counter.get('TRIPLE', 0)
    num_aromatic_bond = bond_counter.get('AROMATIC', 0)
    
    single_bond_atoms = str(neighbor_atoms_with_specific_bonds(neighbors=neighbors, corresponding_bonds=bonds, bond_type='SINGLE'))
    double_bond_atoms = str(neighbor_atoms_with_specific_bonds(neighbors=neighbors, corresponding_bonds=bonds, bond_type='DOUBLE'))
    triple_bond_atoms = str(neighbor_atoms_with_specific_bonds(neighbors=neighbors, corresponding_bonds=bonds, bond_type='TRIPLE'))
    aromatic_bond_atoms = str(neighbor_atoms_with_specific_bonds(neighbors=neighbors, corresponding_bonds=bonds, bond_type='AROMATIC'))
    
    data = (symbol, hybridization, nH, lipo, pi, has_heteroatom, isinamide, isinring, isaromatic, isconjugated,
            num_single_bond, num_double_bond, num_triple_bond, num_aromatic_bond,
            single_bond_atoms, double_bond_atoms, triple_bond_atoms, aromatic_bond_atoms)
    
    logP = logp_dict.get(data, 0.0)
    
    return logP


exceptions = {
    ('C', (0, 1, 0, 0), 'sp3'),
    ('C', (1, 1, 0, 0), 'sp3'),
    ('C', (4, 1, 0, 0), 'sp3'),
    ('C', (frozenset(), frozenset({'C', 'X'}), 'aromatic')),
    ('C', (frozenset(), frozenset({'C'}), 'aromatic')),
    ('C', (frozenset(), frozenset({'X'}), 'aromatic')),
    ('C', (frozenset({'C'}), frozenset({'C', 'X'}), 1, 'sp2')),
    ('C', (frozenset({'C'}), frozenset({'H', 'X'}), 1, 'sp2')),
    ('C', (frozenset({'C'}), frozenset({'X'}), 1, 'sp2')),
    ('C', (frozenset({'X'}), frozenset({'C', 'H'}), 1, 'sp2')),
    ('C', (frozenset({'X'}), frozenset({'C', 'X'}), 1, 'sp2')),
    ('C', (frozenset({'X'}), frozenset({'H', 'X'}), 1, 'sp2')),
    ('C', (frozenset({'X'}), frozenset({'X'}), 1, 'sp2')),
    ('C', None, 'sp'),
    ('N', (0, frozenset(), 0, 'sp2')),
    ('N', (0, frozenset(), 1, 'sp2')),
    ('N', (0, frozenset({'X'}), 1, 'sp2')),
    ('N', (1, frozenset(), 0, 'sp2')),
    ('N', (1, frozenset(), 1, 'sp2')),
    ('N', (2, frozenset(), 0, 'sp2')),
    ('N', (frozenset(), frozenset({'C'}), 18, 'aromatic')),
    ('N', (frozenset(), frozenset({'C'}), 21, 'aromatic')),
    ('N', (frozenset({'C'}), frozenset({'C', 'X'}), 5, 'aromatic')),
    ('N', (frozenset({'C'}), frozenset({'C', 'X'}), 6, 'aromatic')),
    ('N', (frozenset({'C'}), frozenset({'C'}), 14, 'aromatic')),
    ('N', (frozenset({'C'}), frozenset({'C'}), 15, 'aromatic')),
    ('N', (frozenset({'C'}), frozenset({'C'}), 17, 'aromatic')),
    ('N', (frozenset({'C'}), frozenset({'C'}), 5, 'aromatic')),
    ('N', (frozenset({'C'}), frozenset({'C'}), 6, 'aromatic')),
    ('N', (frozenset({'C'}), frozenset({'C'}), 7, 'aromatic')),
    ('N', (frozenset({'C'}), frozenset({'X'}), 5, 'aromatic')),
    ('N', (frozenset({'H'}), frozenset({'C', 'X'}), 6, 'aromatic')),
    ('N', (frozenset({'H'}), frozenset({'C'}), 6, 'aromatic')),
    ('N', (frozenset({'X'}), frozenset({'C', 'X'}), 5, 'aromatic')),
    ('N', (frozenset({'X'}), frozenset({'C'}), 5, 'aromatic')),
    ('N', (frozenset({'X'}), frozenset({'C'}), 6, 'aromatic')),
    ('N', 0),
    ('O', (2, 0), 'sp3'),
    ('O', frozenset(), 'sp2'),
    ('S', frozenset())
    }
        