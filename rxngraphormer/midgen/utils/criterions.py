from rdkit import Chem
from .utils import *

def get_degree(rxn, replacement_dict, see_template_map):
    see_atom_map = replacement_dict[see_template_map]
    reactants, _ = rxn.split('>>')
    rmol = Chem.MolFromSmiles(reactants)
    return [atom.GetDegree() for atom in rmol.GetAtoms() if atom.GetAtomMapNum() == see_atom_map][0]

def get_neighbors(rxn, replacement_dict, see_template_map, modify_neighbors = False, undesired_neighbor = None):
    see_atom_map = replacement_dict[see_template_map]     
    rmol = mfs(rxn.split('>>')[0])
    neighbors = find_map_num(rmol, see_atom_map)[-1].GetNeighbors()
    if modify_neighbors:
        return [neighbor.GetSymbol().capitalize() for neighbor in neighbors if neighbor.GetAtomMapNum() != undesired_neighbor ] 
    return [neighbor.GetSymbol().capitalize() for neighbor in neighbors] 

def next_to_carbonyl(rxn, replacement_dict, see_template_map):
    see_atom_map = replacement_dict[see_template_map]
    reactants, _ = rxn.split('>>')
    rmol = Chem.MolFromSmiles(reactants)
    return any(neighbor_map_num in [match[0] for match in get_map_numbered_matches(rmol, 'C=O')] for neighbor_map_num in get_neighbor_props(rmol, see_atom_map, 'map_num' ))

def next_to_sulfur(rxn, replacement_dict, see_template_map):
    see_atom_map = replacement_dict[see_template_map]
    reactants, _ = rxn.split('>>')
    rmol = Chem.MolFromSmiles(reactants)
    return any(neighbor_map_num in [match[0] for match in get_map_numbered_matches(rmol, 'S')] for neighbor_map_num in get_neighbor_props(rmol, see_atom_map, 'map_num' ))

def has_alkyl_only(rxn, replacement_dict, see_maps):
    ''' Check if carbonyl carbon (denote by see_template_map) has alkyl groups only (ketone/aldehyde) or CAD'''
    if len(see_maps) == 2:
        see_template_map, oxygen_template_map = see_maps
        additional_criterion = False
    else:
        see_template_map, oxygen_template_map, additional_criterion = see_maps
    oxygen_map_num = replacement_dict[oxygen_template_map]
    neighbors = get_neighbors(rxn, replacement_dict, see_template_map, True, undesired_neighbor = oxygen_map_num)
    if additional_criterion:   
        if 'N' in neighbors:     
            return 'False_amide'
        elif 'O' in neighbors:   
            return 'False_acid/ester'
    return all(neighbor in ['H', 'C'] for neighbor in neighbors)

def is_carboxylic_acid(rxn, replacement_dict, see_maps):
    ''' Check if molecules is carboxylic acid or ester '''
    oxygen_template_map, carbon_template_map = see_maps
    carbon_map_num = replacement_dict[carbon_template_map]
    neighbors = get_neighbors(rxn, replacement_dict, oxygen_template_map, True, undesired_neighbor = carbon_map_num)   # check neighbors other than carbonyl carbon
    if not neighbors: 
        return True # it can be negatively charged carboxylic acid with no neighbors
    return all(neighbor == ['H'] for neighbor in neighbors)

def is_highly_substituted(rxn, replacement_dict, see_maps):
    '''Check if atom_1 has higher degrees than atom_2'''
    atom_1_temp_num, atom_2_temp_num = see_maps
    atom_1_degree, atom_2_degree = [get_degree(rxn, replacement_dict, temp_num) for temp_num in [atom_1_temp_num, atom_2_temp_num]]
    return atom_1_degree >= atom_2_degree

def is_acidic(rxn, replacement_dict, see_template_map):
    see_atom_map = replacement_dict[see_template_map]
    reactants, _ = rxn.split('>>')
    rmol = mfs(reactants)
    neighbor_map_nums = get_neighbor_props(rmol, see_atom_map, 'map_num')
    EWG_map_nums = []
    for fg in ['C=O', 'C#N', 'N(=O)O', 'C=N', 'cn', 'P=O', 'S=O',]:
        matches = get_map_numbered_matches(rmol, fg)
        EWG_map_nums.extend([match[0] for match in matches])
    return any(neighbor_map_num in EWG_map_nums for neighbor_map_num in neighbor_map_nums)

def is_Michael_acceptor(rxn, replacement_dict, see_template_map):
    def next_to_fg(rxn, replacement_dict, see_template_map, fg):
        see_atom_map = replacement_dict[see_template_map]
        reactants, _ = rxn.split('>>')
        rmol = Chem.MolFromSmiles(reactants)
        return any(neighbor_map_num in [match[0] for match in get_map_numbered_matches(rmol, fg)] for neighbor_map_num in get_neighbor_props(rmol, see_atom_map, 'map_num' ))
    is_next = []
    for fg in ['C=O', 'C#N', 'N(=O)O', 'S=O', 'P=O', 'C=N', 'cn', 'c1ccncc1',]:
        is_next.append(next_to_fg(rxn, replacement_dict, see_template_map, fg))  
    return any(is_next)

def is_enol_ether(rxn, replacement_dict, temp_map):
    neighbors = get_neighbors(rxn, replacement_dict, temp_map)  # check the neighbors of vinylic carbon
    if 'O' in neighbors:   # if oxygen is attached to vinylic carbon -> enol ether
        return True
    return False
    
def Michael_vs_Markovnikov(rxn, replacement_dict, vinylic_carbons):  
    carbon_2, carbon_3 = vinylic_carbons # assuming carbon #2 is bonded to the oxygen in the products
    if is_enol_ether(rxn, replacement_dict, carbon_2):
        return 'enol_ether'
    elif is_Michael_acceptor(rxn, replacement_dict, carbon_3):
        return 'Michael'
    elif is_highly_substituted(rxn, replacement_dict, (carbon_2, carbon_3)):
        return 'Markovnikov'
    else:
        return 'anti-Markovnikov'

def contains_op_EWG(rxn, replacement_dict, temp_map_num):
    '''Checks whether a ring contains any EWG in ortho/para position w.r.t. atom on position = temp_map_num '''
    EWGS = ['[N+](=O)[O-]', '[N+](=O)[OH1]', 'C#N', 'C=O', 'S=O', 'S(=O)(=O)', 'C=N',] # 'C(F)(F)(F)',
    rmol = mfs(rxn.split('>>')[0]) # reagent mol
    atom_map_num = replacement_dict[temp_map_num]  # map number of a ring atom that's connected to leaving group
    target_idx = find_map_num(rmol, atom_map_num)[0]
    op_substituents = []           # list of tuples containing (index of a ring atom in ortho/para position and index of an attached substituent)
    for pos in get_ortho_para(rmol, target_idx):   # index based ortho/para positions
        atom = rmol.GetAtomWithIdx(pos)              # atom in position = pos
        for neighbor in atom.GetNeighbors():
            if rmol.GetBondBetweenAtoms(pos, neighbor.GetIdx()).GetBondType() == Chem.BondType.SINGLE: # Check if the bond is a single bond (not part of the aromatic ring)
                op_substituents.append((pos, neighbor.GetIdx()))
    EWG_indices = []
    for fg in EWGS:
        matches = rmol.GetSubstructMatches(mfsa(fg))  # index based matches for EWG in whole rmol
        EWG_indices.extend([match[0] for match in matches]) 
    if any(op_sub[-1] in EWG_indices for op_sub in op_substituents):
        for idx in EWG_indices:
            for sub in op_substituents:
                if idx in sub:
                    EWG_connected_atom_idx = sub[0]
                    if EWG_connected_atom_idx in get_neighbor_props(rmol, atom_map_num, 'index'):  # if EWG_connected_atom is neighbor of the ring atom that's connected to leaving group -> ortho attack
                        return 'True_ortho'
                    else:
                        return 'True_para'  # if EWG_connected_atom is not the neighbor of the ring atom that's connected to leaving group -> para attack
    else:
        return False

def get_degree_and_check_nucleophile(rxn, replacement_dict, map_nums):
    '''If nucleophilic alcohol or thiol is already negatively charged -> no need for extra de-protonation step'''
    carbon_temp_map, nuc_temp_map = map_nums
    carbon_atom_map, nuc_atom_map = [replacement_dict[map_num] for map_num in [carbon_temp_map, nuc_temp_map]]
    rmol = Chem.MolFromSmiles(rxn.split('>>')[0])
    degree = find_map_num(rmol, carbon_atom_map)[-1].GetDegree()
    if degree in [1,2]:  # SN2
        if find_map_num(rmol, nuc_atom_map)[-1].GetFormalCharge() == 0:
            return 'SN2_deprotonation_needed'
        else:
            return 'SN2_no_deprotonation_needed'
    else:  # SN1
        return degree
    
    

def is_nuc_neg_and_ring_contains_op_EWG(rxn, replacement_dict, map_nums):
    carbon_temp_map, nuc_temp_map = map_nums
    op_EWG = contains_op_EWG(rxn, replacement_dict, carbon_temp_map)
    rmol = Chem.MolFromSmiles(rxn.split('>>')[0])
    nuc_atom_map = replacement_dict[nuc_temp_map] 
    if find_map_num(rmol, nuc_atom_map)[-1].GetFormalCharge() == 0 or not op_EWG:  # in case neutral nucleophile or op_EWG is False
        return op_EWG
    elif op_EWG == 'True_ortho':
        return 'ortho_no_deprotonation_needed'
    else:    # 'True_para'
        return 'para_no_deprotonation_needed'   
    

def is_ring_formed(rxn, replacement_dict, temp_maps):
    pmol = Chem.MolFromSmiles(rxn.split('>>')[-1])  # product molecule
    O_idx, C_idx = [find_map_num(pmol, replacement_dict[temp_map])[0] for temp_map in temp_maps]
    rings = pmol.GetRingInfo().AtomRings()  # index based rings
    for ring in rings:
        if O_idx in ring and C_idx in ring:
            return True 
    return False