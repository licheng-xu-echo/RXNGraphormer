'''
This python script is modified from rdchiral template extractor 
https://github.com/connorcoley/rdchiral/blob/master/rdchiral/template_extractor.py
'''
import re
from numpy.random import shuffle
from collections import defaultdict
from pprint import pprint 
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import ChiralType


def get_special_groups(mol):
    '''Given an RDKit molecule, this function returns a list of tuples, where
    each tuple contains the AtomIdx's for a special group of atoms which should 
    be included in a fragment all together. This should only be done for the 
    reactants, otherwise the products might end up with mapping mismatches
    We draw a distinction between atoms in groups that trigger that whole
    group to be included, and "unimportant" atoms in the groups that will not
    be included if another atom matches.'''

    # Define templates
    group_templates = [ 
        # Functional groups
        (range(2), '[*]=[*]',), # any conjugation 
        (range(2), '[*]#[*]',), # any conjugation 
        (range(2), 'a:a',), # any conjugation 00
        
        # special groups
        (range(7), 'CC(=O)CC(=O)O',),
        ((0,1,), 'OOC(=O)',),
        ((0,1), 'OC1CCCCO1',),
        ((0,1,2), 'COCO',),
        ((1,2), 'NC(=O)[#6]',),
        (range(2), 'nC1CCCCO1',),
        (range(3), 'ClS(Cl)=O',),
        (range(2), 'ON(=O)O',), 
        ((6,), 'CC(C)(C)OC(N)=O',), # Boc with non-aromatic nitrogen
        ((6,), 'CC(C)(C)OC(n)=O',), # Boc with aromatic nitrogen
        (range(2), 'NC(=O)OCc1ccccc1',),
        (range(2), 'NC(=O)OCC1c2ccccc2c2ccccc21',)        
    ]
    
    # Build list
    groups = []
    for (add_if_match, template) in group_templates:
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(template), useChirality=True)
        for match in matches:
            add_if = []
            for pattern_idx, atom_idx in enumerate(match):
                if pattern_idx in add_if_match:
                    add_if.append(atom_idx)
            groups.append((add_if, match, template))
    return groups

def expand_atoms_to_use(mol, atoms_to_use, groups=[], symbol_replacements=[]):
    '''Given an RDKit molecule and a list of AtomIdX which should be included
    in the reaction, this function expands the list of AtomIdXs to include one 
    nearest neighbor with special consideration of (a) unimportant neighbors and
    (b) important functional groupings'''

    # Copy
    new_atoms_to_use = atoms_to_use[:]
    special_num = 200
    special_mappings = {}
    # Look for all atoms in the current list of atoms to use
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in atoms_to_use: continue
        # Ensure membership of changed atom is checked against group
        for group in groups:
            if int(atom.GetIdx()) in group[0]:
                for idx in group[1]:
                    if idx not in new_atoms_to_use:
                        n_atom = mol.GetAtomWithIdx(idx)
                        if n_atom.GetAtomMapNum() == 0:
                            special_num += 1
                            n_atom.SetAtomMapNum(special_num)
                            special_mappings[idx] = special_num
                        new_atoms_to_use.append(idx)
                        symbol = get_strict_smarts_for_special_atom(n_atom)
                        if symbol != atom.GetSmarts():
                            symbol_replacements.append((idx, symbol))
    return new_atoms_to_use, symbol_replacements, mol

def get_strict_smarts_for_special_atom(atom):
    '''
    For an RDkit atom object, generate a SMARTS pattern that
    matches the atom as strictly as possible
    '''
    
    symbol = '[%s:%s]' % (atom.GetSymbol(), atom.GetAtomMapNum())
    if 'H' in symbol and 'Hg' not in symbol:
        symbol = symbol.replace('H', '')
    if atom.GetIsAromatic():
        symbol = '[a:%s]' % (atom.GetAtomMapNum())
    if atom.GetSymbol() == 'H':
        symbol = '[#1]'
        
    if '[' not in symbol:
        symbol = '[' + symbol + ']'
            
    return symbol

def expand_atoms_to_use_atom(mol, atoms_to_use, atom_idx, groups=[], symbol_replacements=[]):
    '''Given an RDKit molecule and a list of AtomIdx which should be included
    in the reaction, this function extends the list of atoms_to_use by considering 
    a candidate atom extension, atom_idx'''

    # See if this atom belongs to any special groups (highest priority)
    found_in_group = False
    for group in groups: # first index is atom IDs for match, second is what to include
        if int(atom_idx) in group[0]: # int correction
            # Add the whole list, redundancies don't matter 
            # *but* still call convert_atom_to_wildcard!
            for idx in group[1]:
                if idx not in atoms_to_use:
                    atoms_to_use.append(idx)
            found_in_group = True
    if found_in_group:  
        return atoms_to_use, symbol_replacements

    # Skip current candidate atom if it is already included
    if atom_idx in atoms_to_use:
        return atoms_to_use, symbol_replacements

    # Include this atom
    atoms_to_use.append(atom_idx)

    return atoms_to_use, symbol_replacements