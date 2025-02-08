import pandas as pd 
import numpy as np
# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
from rdkit import Chem
from rdkit.Chem import Draw, PandasTools
from rdkit.Chem import AllChem, MolFromSmiles as mfs, MolFromSmarts as mfsa, MolToSmiles as mts, MolToSmarts as mtsa
PandasTools.RenderImagesInAllDataFrames(images=True)

from .arrow_pushing import arrow_pushing
# from arrow_pushing import arrow_pushing

def clean_leaving_mapping(rxn):
    r, p = rxn.split('>>')
    p = max(p.split('.'), key = len)
    rmol, pmol = Chem.MolFromSmiles(r), Chem.MolFromSmiles(p)
    rmaps = {atom.GetIdx(): atom.GetAtomMapNum() for atom in rmol.GetAtoms()}
    pmaps = [atom.GetAtomMapNum() for atom in pmol.GetAtoms()]
    [atom.SetAtomMapNum(0) for atom in rmol.GetAtoms() if atom.GetAtomMapNum() not in pmaps]
    r = Chem.MolToSmiles(rmol)
    return '%s>>%s' % (Chem.MolToSmiles(rmol), p), rmaps

def bond_to_label(bond):
    '''This function takes an RDKit bond and creates a label describing
    the most important attributes'''
    a1_label = str(bond.GetBeginAtom().GetAtomicNum())
    a2_label = str(bond.GetEndAtom().GetAtomicNum())
    if bond.GetBeginAtom().HasProp('molAtomMapNumber'):
        a1_label += bond.GetBeginAtom().GetProp('molAtomMapNumber')
    if bond.GetEndAtom().HasProp('molAtomMapNumber'):
        a2_label += bond.GetEndAtom().GetProp('molAtomMapNumber')
    atoms = sorted([a1_label, a2_label])

    return '{}{}{}'.format(atoms[0], bond.GetSmarts(), atoms[1])

def atoms_are_different(atom1, atom2):
    '''Compares two RDKit atoms based on basic properties'''

    if atom1.GetAtomicNum() != atom2.GetAtomicNum(): return True # must be true for atom mapping
    if atom1.GetTotalNumHs() != atom2.GetTotalNumHs(): return True
    if atom1.GetFormalCharge() != atom2.GetFormalCharge(): return True
    if atom1.GetDegree() != atom2.GetDegree(): return True
    #if atom1.IsInRing() != atom2.IsInRing(): return True # do not want to check this!
    # e.g., in macrocycle formation, don't want the template to include the entire ring structure
    if atom1.GetNumRadicalElectrons() != atom2.GetNumRadicalElectrons(): return True
    if atom1.GetIsAromatic() != atom2.GetIsAromatic(): return True 

    # Check bonds and nearest neighbor identity
    bonds1 = sorted([bond_to_label(bond) for bond in atom1.GetBonds()]) 
    bonds2 = sorted([bond_to_label(bond) for bond in atom2.GetBonds()]) 
    if bonds1 != bonds2: return True

    return False

def get_tagged_atoms_from_mol(mol):
    '''Takes an RDKit molecule and returns list of tagged atoms and their
    corresponding numbers'''
    atoms = []
    atom_tags = []
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atoms.append(atom)
            atom_tags.append(str(atom.GetProp('molAtomMapNumber')))
    return atoms, atom_tags

def find_map_num(mol, mapnum):
    return [(a.GetIdx(), a) for a in mol.GetAtoms() if a.HasProp('molAtomMapNumber') 
         and a.GetProp('molAtomMapNumber') == str(mapnum)][0]

def get_map_num_from_index(mol, i):
    return mol.GetAtomWithIdx(i).GetAtomMapNum()

def get_symbol_from_map_num(mol, i):
    return find_map_num(mol, i)[-1].GetSymbol()


def get_neighbor_props(mol, atom_tag, prop):
    '''Given a molecule (mol), this function returns the list of properties (prop) of 
    the neighbors of an atom indicated by its map number (atom_tag)'''
    if prop == "map_num":
        return [atom.GetAtomMapNum() for atom in find_map_num(mol, atom_tag)[-1].GetNeighbors()]
    if prop == "symbol":
        return [atom.GetSymbol() for atom in find_map_num(mol, atom_tag)[-1].GetNeighbors()]
    if prop == "index":
        return [atom.GetIdx() for atom in find_map_num(mol, atom_tag)[-1].GetNeighbors()]
    
    
def get_map_numbered_matches(mol, template):
    '''Given a molecule (mol) and target template, this funnction returns 
    list of matches in the molecule indicated by map numbers '''
    return [tuple(map(lambda i: get_map_num_from_index(mol, i), match)) for match in mol.GetSubstructMatches(mfsa(template))]

# 
def update_map_nums(mol, get_added = False):
    '''Given a molecule (mol), this function adds mapping to unmapped atoms 
    without changing the rest (e.g. after adding hydrogens)
    * parameter get_added is set to True if newly added map numbers are needed '''
    all_map_nums = [atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.HasProp('molAtomMapNumber')]
    added = []
    for atom in mol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if not map_num:    
            map_num = max(all_map_nums) + 1
            all_map_nums.append(map_num)
            if get_added:
                added.append(map_num)
        atom.SetAtomMapNum(map_num)
    return (mol, added) if get_added else mol

# get list of map numbers of hydrogens attached to an atom in a molecule
def get_neighboring_hydrogens(mol, atom_map_num):
    all_neighbors = find_map_num(mol, atom_map_num)[-1].GetNeighbors()
    return [h.GetAtomMapNum() for h in all_neighbors if h.GetAtomicNum() == 1] 

def get_indexed_ss_seq(mol, map_numbered_ss_seq):
    '''Given a molecule and a map numbered ss_seq, this function returns index numbered ss_seq '''
    def update(s):
        if isinstance(s, int):
            return find_map_num(mol, s)[0] 
        elif isinstance(s, float):
            return s
        else:
            contents = map(lambda i: find_map_num(mol, i)[0] if isinstance(i, int) else i, s)
            return tuple(contents) if type(s) == tuple else list(contents) 
    idx_num_ss_seq = []
    for ss in map_numbered_ss_seq:
        idx_num_ss_seq.append(tuple([update(s) for s in ss]))
    return idx_num_ss_seq

def neutralize_atoms(mol):
    '''Given a molecule, this function neutralizes atoms with a +1 or -1 charge 
    by removing or adding hydrogen where possible. '''
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol


def get_num_unshared_el(mol, idx):
    '''Given a molecule(mol) and index of an atom (idx), 
    this function returns the number of unshared electrons around that atom'''
    Chem.Kekulize(mol)   # to get exact number of bonds in aromatic rings  (double and single bonds instead of 1.5 bond orders)
    
    atom = mol.GetAtomWithIdx(idx)

    fc = atom.GetFormalCharge()
    valence = Chem.GetPeriodicTable().GetNOuterElecs(atom.GetAtomicNum())

    bonds = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
    bonds.extend([atom.GetTotalNumHs()])
    num_bonds = int(sum(bonds))
        
    unshared_el = valence-fc-num_bonds
    
    return unshared_el


def compare_neighbors(atom_tag, mol_1, mol_2, prop):
    '''Given two molecules (mol_1 & mol_2), this function compares the neighbors of an atom 
    denoted by its map number (atom_tag) based on a given property (prop) and returns a boolean'''
    atom_in_mol_1, atom_in_mol_2 = [find_map_num(mol, atom_tag)[-1] for mol in [mol_1, mol_2]]
    neighbors_in_mol_1, neighbors_in_mol_2 = [atom.GetNeighbors() for atom in [atom_in_mol_1, atom_in_mol_2]]
    if prop == "length":
        return len(neighbors_in_mol_1) == len(neighbors_in_mol_2)
    elif prop == "map_num":
        map_nums_1 = [atom.GetAtomMapNum() for atom in neighbors_in_mol_1]
        map_nums_2 = [atom.GetAtomMapNum() for atom in neighbors_in_mol_2]
        return set(map_nums_1) == set(map_nums_2)
    elif prop == "symbol":
        symbols_1 = sorted([atom.GetSymbol() for atom in neighbors_in_mol_1])
        symbols_2 = sorted([atom.GetSymbol() for atom in neighbors_in_mol_2])
        if len(symbols_1) == len(symbols_2):
            for i in range(len(symbols_1)):
                if symbols_1[i] != symbols_2[i]:
                    return False
        else:
            return False
        return True
    
def get_bond_type(mol, atom_tag_1, atom_tag_2):
    '''Given a molecule (mol), this function returns 
    the type of the bond between two atoms indicated by their map numbers (atom_tag_1, atom_tag_2) '''
    atom_idx_1, atom_idx_2 = [find_map_num(mol, atom_tag)[0] for atom_tag in [atom_tag_1, atom_tag_2]]
    return mol.GetBondBetweenAtoms(atom_idx_1, atom_idx_2 ).GetBondTypeAsDouble()


def apply_seq(rxn, map_numbered_ss_seq):
    '''Given a reaction (rxn) and its label (map_numbered_ss_seq), 
    this function returns original and obtained products'''
    reactants, products_original = [mfs(smi) for smi in rxn.split(">>")]
    intermediate = reactants
    for source, sink in map_numbered_ss_seq:
        try: 
            intermediate = arrow_pushing(intermediate, source, sink)
        except Exception as e:
            print (e)
            return None   
    products_obtained = intermediate

    try: 
        products_obtained = neutralize_atoms(mfs(mts(Chem.rdmolops.RemoveAllHs(products_obtained, sanitize=False))))
    except: 
        return None 
    return [mfs(mts(products_original)), products_obtained]

def compare(given, obtained):
    '''Given original product (from dataset) and obtained product upon applying source_sink sequence to reactants, 
    this function returns a list of changed atoms and their map numbers'''
#     [Chem.SanitizeMol(mol) for mol in [given, obtained]]
    given, obtained = [Chem.AddHs(mol) for mol in [given, obtained]]
    changed_atoms = []
    changed_atom_tags = []
    given_prod_atoms, given_prod_atom_tags = get_tagged_atoms_from_mol(given)
    prod_atoms, prod_atom_tags = get_tagged_atoms_from_mol(obtained)
    for i, given_prod_atom_tag in enumerate(given_prod_atom_tags):
        for j, prod_tag in enumerate(prod_atom_tags):
            if given_prod_atom_tag != prod_tag: continue
            if atoms_are_different(given_prod_atoms[i], prod_atoms[j]):
                changed_atoms.append(prod_atoms[j])
                changed_atom_tags.append(prod_tag)
    return changed_atoms, changed_atom_tags

def check_label(rxn, label):
    result = apply_seq(rxn, label)
    if result:
        products_original, products_obtained = result
        Chem.rdmolops.SanitizeMol(products_obtained)
        diff = compare(neutralize_atoms(mfs(mts(Chem.rdmolops.RemoveAllHs(products_original, sanitize=False)))), products_obtained)[-1]
        if diff == []:     # the label achieved the transformation => done
            return True, None
        else:             # different, but chemically valid structure is obtained => subject to next iteration
            return True, mts(products_obtained)
    else:
        return False, None    # invalid label due to arrow_pushing error


def get_ortho_para(mol, target_idx):
    '''Given a molecule (mol) and index of target position, this function returns indices of atoms that hold ortho & para position w.r.t given target_map_num'''
    rings = mol.GetRingInfo().AtomRings()
    for ring in rings:
        if target_idx in ring:
            break # choose the ring that has reacting atom (i.e., target_idx)
    ring = list(ring) # indices of all atoms in the chosen ring
    # shift the ring atoms to start with target_map_num
    star_idx = ring.index(target_idx)
    ring = ring[star_idx:] + ring[:star_idx]
    # choose odd numbered indices => gives ortho/para positions w.r.t target_map_num
    ortho_para = [ring[i] for i in range(len(ring)) if i%2!=0]
    return ortho_para  # index based

def get_EWG_connected_atom(rmol, target_idx):
    '''Checks whether a ring contains any EWG in ortho/para position w.r.t. atom on position = temp_map_num '''
    EWGS = ['[N+](=O)[O-]', '[N+](=O)[OH1]', 'C#N', 'C=O', 'S=O', 'S(=O)(=O)', 'C=N',] # 'C(F)(F)(F)',
    atom_map_num = rmol.GetAtomWithIdx(target_idx).GetAtomMapNum()  # map number of a ring atom that's connected to leaving group
    op_substituents = []           # list of tuples containing (index of a ring atom in ortho/para position and index of an attached substituent)
    for pos in get_ortho_para(rmol, target_idx):   # index based ortho/para positions
        atom = rmol.GetAtomWithIdx(pos)              # atom in position = pos
        for neighbor in atom.GetNeighbors():
            if rmol.GetBondBetweenAtoms(pos, neighbor.GetIdx()).GetBondType() == Chem.BondType.SINGLE: # Check if the bond is a single bond (not part of the aromatic ring)
                op_substituents.append((pos, neighbor.GetIdx()))
    for fg in EWGS:
        matches = rmol.GetSubstructMatches(mfsa(fg))  # index based matches for EWG in whole rmol
        for match in matches:
            for sub in op_substituents:
                if match[0] in sub:
                    EWG_connected_atom_idx = sub[0]
                    if EWG_connected_atom_idx in get_neighbor_props(rmol, atom_map_num, 'index'):  # if EWG_connected_atom is neighbor of the ring atom that's connected to leaving group -> ortho attack
                        return EWG_connected_atom_idx, list(match[:2])  
                    else:
                        rings = rmol.GetRingInfo().AtomRings()
                        for ring in rings:
                            if target_idx in ring:
                                break # choose the ring that has reacting atom (i.e., target_idx)
                        ring = list(ring) # indices of all atoms in the chosen ring
                        # shift the ring atoms to start with target_map_num
                        star_idx = ring.index(target_idx)
                        ring = ring[star_idx:] + ring[:star_idx]
                        # choose indices of atoms spanning from ortho to meta to para (omp)
                        omp = ring[1:4] # ortho, meta, para positions to be marked as 'xl1', 'xl2', 'xl3'
                        return omp, list(match[:2])
                
def get_acidic_EWG_map_nums(rmol, target_idx):
    atom_map_num = rmol.GetAtomWithIdx(target_idx).GetAtomMapNum() 
    neighbor_map_nums = get_neighbor_props(rmol, atom_map_num, 'map_num')
    for fg in ['C=O', 'C#N', 'N(=O)O', 'C=N', 'cn', 'P=O', 'S=O',]:
        matches = get_map_numbered_matches(rmol, fg)
        for match in matches:
            if match[0] in neighbor_map_nums:
                return list(match[:2])
    return 