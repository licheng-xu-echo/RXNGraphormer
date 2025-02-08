import rdkit
from rdkit import Chem
from rdkit.Chem import MolFromSmiles as mfs, MolToSmiles as mts, rdAbbreviations as abbrev, Draw   
import copy

from .utils import *

bond_dict1 = {0:None, 1:Chem.rdchem.BondType.SINGLE, 2:Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}
bond_dict2 = {'SINGLE':1, 'DOUBLE':2, 'TRIPLE':3}

def find_map_num(mol, mapnum):
    return [(a.GetIdx(), a) for a in mol.GetAtoms() if a.HasProp('molAtomMapNumber') 
         and a.GetProp('molAtomMapNumber') == str(mapnum)][0]

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

def preprocess(mol, map_num):
    if isinstance(map_num, float):  # if it's hydrogen
        atom = [neighbor for neighbor in  find_map_num(mol, int(map_num))[-1].GetNeighbors() if neighbor.GetAtomicNum() == 1][-1] 
        atom_idx = atom.GetIdx()
    elif isinstance(map_num, int):  # if it's not hydrogen
        atom_idx, atom = find_map_num(mol, map_num)
    return atom_idx, atom

def get_h_attached_atoms(mol, pathway):
    h_attached_atoms = []
    for path in pathway:
        if isinstance(path, float):  h_attached_atoms.append(int(path))
        elif isinstance(path, list): h_attached_atoms.extend(get_h_attached_atoms(mol,path))
    return h_attached_atoms
    
def arrow_pushing(omol, source, sink, visualize=False, rxn_class = None, step_no = None):
    mol = copy.copy(omol)
    h_attached_atoms = [find_map_num(mol, map_num)[0] for map_num in get_h_attached_atoms(mol, [source,sink])]
    if h_attached_atoms: mol = update_map_nums(Chem.AddHs(mol, onlyOnAtoms=h_attached_atoms))
#     mol = Chem.rdmolops.AddHs(mol)
    
    # #     Visualize the intermediate before path occurrence 
    if visualize:
        mol_vis = Chem.RemoveHs(mol, sanitize=False)
        Draw.ShowMol(mol_vis, size=(1200,550), title = f"{rxn_class} reaction. Step #{step_no}: {source} =>  {sink}")
    
    
    # attack from lone pair
    if isinstance(source, int):
        
        # set formal charge of source atom
        source_atom = find_map_num(mol, source)[-1]
        source_atom.SetFormalCharge(source_atom.GetFormalCharge()+1)
        
        # to atom
        if isinstance(sink, int):
            sink_atom = find_map_num(mol, sink)[-1]
            sink_atom.SetFormalCharge(sink_atom.GetFormalCharge()-1)
            emol = Chem.EditableMol(mol)
            emol.AddBond(find_map_num(mol, source)[0], find_map_num(mol, sink)[0], Chem.rdchem.BondType.SINGLE)
            mol =  emol.GetMol()
        
        # to bond
        elif isinstance(sink, list):
            sink_start_idx, sink_start = find_map_num(mol, sink[0])
            sink_end_idx, sink_end = find_map_num(mol, sink[1])
                      
            sink_end.SetFormalCharge(sink_end.GetFormalCharge()-1)
            bond = mol.GetBondBetweenAtoms(sink_start_idx, sink_end_idx)
            bond_type = str(bond.GetBondType())
            # if there's an attack of lone pair to a bond which is AROMATIC, we consider it as degree one for mechanistic purposes
            if bond_type == 'AROMATIC': 
                bond_type = 'SINGLE'  # old bond
            bond_type = bond_dict1[bond_dict2[bond_type]+1]
            bond.SetBondType(bond_type)

        # to hydrogen
        # if sink is float, it represents hydrogen and is denoted by map num of attached atom, e.g, 5.1 shows hydrogen attached to atom having MAP NUMBER = 5 (not index) 
        elif isinstance(sink, float):   
            sink_hydrogen = [neighbor for neighbor in  find_map_num(mol, int(sink))[-1].GetNeighbors() if neighbor.GetAtomicNum() == 1][-1] 
            sink_hydrogen.SetFormalCharge(sink_hydrogen.GetFormalCharge()-1)
            emol = Chem.EditableMol(mol)
            emol.AddBond(find_map_num(mol, source)[0], sink_hydrogen.GetIdx(), Chem.rdchem.BondType.SINGLE)
            mol =  emol.GetMol()
            
            
   # attack from bond
    elif isinstance(source, list):
        # set formal charge of source atom
               
        for i, source_map_num in enumerate(source):
            if i==0  :    source_start_idx, source_start = preprocess(mol, source_map_num)
            elif i==1:    source_end_idx, source_end = preprocess(mol, source_map_num)
                    
        source_start.SetFormalCharge(source_start.GetFormalCharge()+1) 
        
        # set new bond
        bond = mol.GetBondBetweenAtoms(source_start_idx, source_end_idx)
        bond_type = str(bond.GetBondType())                # original bond
        
        ## only aromatic DOUBLE bonds attack
        if bond_type == 'AROMATIC':
            bond_type = 'DOUBLE'
            
        new_bond_type = bond_dict1[bond_dict2[bond_type]-1]    # breaking bond ==> new_bond_type = original_bond_type - 1
        
        if not new_bond_type:             # if new_bond_type = 0 (None) ==> remove bond
            emol = Chem.EditableMol(mol)
            emol.RemoveBond(source_start_idx, source_end_idx)
            mol =  emol.GetMol()
        else:                         # else ==> replace with new_bond_type
            bond.SetBondType(new_bond_type)
        
        # to atom
        if isinstance(sink, int): 
            sink_atom_idx, sink_atom = find_map_num(mol, sink)
            sink_atom.SetFormalCharge(sink_atom.GetFormalCharge()-1)
            
            # attack to different atom    
            if sink_atom_idx != source_end_idx:
                emol = Chem.EditableMol(mol)
                emol.AddBond(source_end_idx, sink_atom_idx, Chem.rdchem.BondType.SINGLE)
                mol =  emol.GetMol()
           
        # to hydrogen 
        elif isinstance(sink, float): 
            sink_atom_idx, sink_atom = preprocess(mol, sink)
            sink_atom.SetFormalCharge(sink_atom.GetFormalCharge()-1)
            
            # to different hydrogen
            if sink_atom_idx != source_end_idx:
                emol = Chem.EditableMol(mol)
                emol.AddBond(source_end_idx, sink_atom_idx, Chem.rdchem.BondType.SINGLE)
                mol =  emol.GetMol()
            
        # to bond
        elif isinstance(sink, list):
            sink_start_idx, sink_start = find_map_num(mol, sink[0])
            sink_end_idx, sink_end = find_map_num(mol, sink[1])
            if all(atom.GetIsAromatic() for atom in [sink_start, sink_end]):   # if new set bond is aromatic
                sink_end.SetFormalCharge(sink_end.GetFormalCharge()-1)
                bond = mol.GetBondBetweenAtoms(sink_start_idx, sink_end_idx)
                bond.SetBondType(Chem.rdchem.BondType.AROMATIC)  
            else:
                sink_end.SetFormalCharge(sink_end.GetFormalCharge()-1)
                bond = mol.GetBondBetweenAtoms(sink_start_idx, sink_end_idx)
                bond_type = str(bond.GetBondType())
                new_bond_type = bond_dict1[bond_dict2[bond_type]+1]
                bond.SetBondType(new_bond_type)


    return mol
