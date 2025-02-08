
################### From https://github.com/snu-micc/MechFinder ###################
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, PandasTools
from rdkit.Chem import MolFromSmiles as mfs, MolFromSmarts as mfsa, MolToSmiles as mts, MolToSmarts as mtsa

from .utils.utils import *
from .utils.criterions import *

from .LocalTemplate.template_extractor import extract_from_reaction

out_of_scope_mechanisms = ['nitro_reduction', 'alkene_reduction', 'hydrogenative_deprotection', 'radical_reaction',
                        'aromatic_dehalogenation', 'Heck', 'alkyne_reduction', 'Stille_coupling', 'protodesilylation', 
                        'Grignard_reagent_prep', 'Suzuki_coupling', 'Negishi_coupling', 'one_pot_Grignard_synthesis', 
                        'one_pot_Weinreb_ketone_synthesis', 'Ullmann', 'Bouveault_aldehyde_synthesis', 'Huisgen_cycloaddition', 'Rosenmund_von_Braun',
                        'catalytic_amination', 'catalytic_coupling', 'Barton_McCombie_deoxygenation', 'radical_dehalogenation', 
                           ]

wrong_atom_mapped_reactions = ['wrong_atom_mapped_esterification', 'wrong_atom_mapped_ester_hydrolysis', 
                            'wrong_atom_mapped_reduction', 'wrong_atom_mapped_(hemi)acetal_hydrolysis', 
                            'wrong_atom_mapped_(hemi)acetal_formation', 'wrong_atom_mapped_hydroboration_oxidation', 
                            'wrong_atom_mapped_alcohol_condensation', 'wrong_atom_mapped_amide_formation',
                            'wrong_atom_mapped_carboxylic_acid_reduction', 'wrong_atom_mapped_Williamson_ether_synthesis',
                            'wrong_atom_mapped_Friedel_Crafts_acylation', 'wrong_atom_mapped_carboxylic_acid_LAH_reduction']
    
def add_reagent(rxn, reagents, replacement_dict):
    if reagents:
        reactants, products = rxn.split('>>')
        reagent_n = 300
        for reagent, template_no in reagents.items():
            if template_no == '_':
                l = reagent.split(':')
                for i, part in enumerate(l):
                    if part[0].isdigit():
                        reagent_n += 1
                        j = 1
                        while part[j].isdigit():
                            j+=1
                        l[i] = str(reagent_n) + part[j:]
                        replacement_dict[int(part[:j])] = reagent_n                           
                reagent_updated = ':'.join(l)
                reactants = '%s.%s' % (reactants, reagent_updated)
            elif reagent not in reactants.split('.'):
                reagent_n += 1
                reactants = '%s.[%s:%d]' % (reactants, reagent, reagent_n)
                replacement_dict[template_no] = reagent_n
        rxn = reactants + '>>' + products
    return rxn, replacement_dict

def change_atom_map(replacement_dict, template_path):
    map_path = []
    for path in template_path:
        if not isinstance(path, list): # if it's some atom (int) or hydrogen (float) or 'xl'
            path = replacement_dict[path] if any(isinstance(path, t) for t in [int, str]) else replacement_dict[int(path)] + 0.1
        else:
            path = [replacement_dict[p] if any(isinstance(p, t) for t in [int, str]) else replacement_dict[int(p)] + 0.1 for p in path]
        map_path.append(path)
    return tuple(map_path)

def adjust_template_atom_map(adjust_dict, mech_path):
    template_path = []
    for path in mech_path:
        if not isinstance(path, list): # if it's some atom (int) or hydrogen (float) or 'xl'
            path = adjust_dict[path] if any(isinstance(path, t) for t in [int, str]) else adjust_dict[int(path)] + 0.1
        else:
            path = [adjust_dict[p] if any(isinstance(p, t) for t in [int, str]) else adjust_dict[int(p)] + 0.1 for p in path]
        template_path.append(path)
    return tuple(template_path)

def swap_map_nums(smile, replacement_dict, temp_no_1, map_num_1, temp_no_2, map_num_2):
    l = smile.split(':')
    for i, part in enumerate(l):
        if part[0].isdigit():
            j = 1
            while part[j].isdigit():
                j+=1
            if part[:j] == str(map_num_1):
                l[i] = str(map_num_2) + part[j:]
                replacement_dict[temp_no_1] = map_num_2
            elif part[:j] == str(map_num_2):
                l[i] = str(map_num_1) + part[j:]  
                replacement_dict[temp_no_2] = map_num_1
    return ':'.join(l)

def replace_dict(rxn, replacement_dict, return_idx = False):
    if return_idx:
        reactants, _ = rxn.split('>>')
        rmol = Chem.MolFromSmiles(reactants)
        map2idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in rmol.GetAtoms()}
        return {int(v):map2idx[int(k)] for k, v in replacement_dict.items()}
    else:
        return {int(v):int(k) for k, v in replacement_dict.items()}
    
def neutralize_charge(rxn):
    charged_atoms = {'NH3+': 'NH2', 'NH4+': 'NH3', 'O-:': 'OH:'}
    for k, v in charged_atoms.items():
        rxn = rxn.replace(k, v)
    return rxn

def build_ext_dict(rxn, replacement_dict, ext_info):
    rmol = mfs(rxn.split('>>')[0])
    temp_map_num, fg, ext_strings = ext_info['target_map_num'], ext_info['FG'], ext_info['ext_strings']
    atom_map_num = replacement_dict[temp_map_num]
    target_idx = find_map_num(rmol, atom_map_num)[0]   # index of atom that is connected to leaving group
    if fg in 'aromatic_ring':   # case where LRT extension applies to the aromatic ring atoms beyond target_map_num
        # get idx of EWG-attached atom along with EWG indices (of double bond in EWG)
        ring_atom, EWG_db_indices = get_EWG_connected_atom(rmol, target_idx)  # ring_atom might be EWG_connected_atom_idx (int) if SNAr_ortho; or ortho, meta, para ring atoms (list) if SNAr_para 
        EWG_indices = [ring_atom] + EWG_db_indices if isinstance(ring_atom, int) else ring_atom + EWG_db_indices
        ext_dict = dict()
        for i, s in enumerate(ext_strings):
            ext_dict[s] = rmol.GetAtomWithIdx(EWG_indices[i]).GetAtomMapNum()    # 'xl1' -> ortho atom; 'xl2' & 'xl3' -> EWG db        
    elif fg =='alpha_EWG':
        EWG_map_nums = get_acidic_EWG_map_nums(rmol, target_idx)
        ext_dict = dict()
        for i, s in enumerate(ext_strings):
            ext_dict[s] = EWG_map_nums[i]
    else:                       # simple fg case, e.g., fg = 'C=O'
        fg_matches = get_map_numbered_matches(rmol, fg)
        for match in fg_matches:
            if match[0] in get_neighbor_props(rmol, atom_map_num, 'map_num'):
                break
        ext_dict = dict()
        for i, s in enumerate(ext_strings):
            ext_dict[s] = match[i]
    return ext_dict

def replace_xl(mech_pathway, ext_dict):
    '''Given final, actual atom map numbered mech_pathway with 'xl' strings to be replaced'''
    updated_mech_pathway = []
    for attack in mech_pathway:
        updated_attack = []
        for path in attack:
            if not isinstance(path, list): # if it's some atom (int) or hydrogen (float) or 'xl'
                path = ext_dict[path] if isinstance(path, str) else path
            else:
                path = [ext_dict[p] if isinstance(p, str) else p for p in path]
            updated_attack.append(path)
        updated_mech_pathway.append(tuple(updated_attack))
    return updated_mech_pathway

class MechFinder:
    def __init__(self, collection_dir='collections', debug = False):
        MT_collection = pd.read_csv('%s/MT_library.csv' % collection_dir)
        LRT_collection = pd.read_csv('%s/LRT_library.csv' % collection_dir)
        self.MT_collection = MT_collection.replace(np.nan, None).set_index('MT_class').to_dict('index')
        self.LRT_collection = LRT_collection.replace(np.nan, None).replace(np.nan, None).set_index('LRT').to_dict('index')
        self.out_of_scope_mechanisms = out_of_scope_mechanisms
        self.wrong_atom_mapped_reactions = wrong_atom_mapped_reactions
        self.debug = debug
        
    def check_exception(self, MT_class):
        if MT_class in ['wrong atom-mapping', 'mechanism not in collection']:
            return MT_class
        elif MT_class in self.out_of_scope_mechanisms:  
            return 'outside_scope_of_arrow_pushing'
        elif MT_class in self.wrong_atom_mapped_reactions:
            return 'wrong_atom_mapping'
        elif MT_class == 'missing_info':        
            return 'missing_info'
        return False
            
    def get_LRT(self, rxn):
        rxn, rmaps = clean_leaving_mapping(rxn)
        #print(rxn, rmaps,extract_from_reaction(rxn))
        rxn, template, replacement_dict = extract_from_reaction(rxn)
        if not template:
            return rxn, template, None, {'MT_class': 'wrong atom-mapping'}
        if template not in self.LRT_collection:    
            return rxn, template, None, {'MT_class': 'mechanism not in collection'}
        LRT_info = self.LRT_collection[template]
        replacement_dict = replace_dict(rxn, replacement_dict)
        rxn = neutralize_charge(rxn)
        return rxn, template, replacement_dict, LRT_info
        
    def get_electron_path(self, rxn):
        
        rxn, template, replacement_dict, LRT_info = self.get_LRT(rxn)

        MT_class = LRT_info['MT_class']
        exception = self.check_exception(MT_class)
        if exception:
            if exception == 'wrong_atom_mapping' and self.debug:
                print ('#########################################################')
                print ('MT: %s, LRT: %s', (MT_class, template))
                print ('Reaction:', rxn)
                print ('#########################################################')
            return rxn, template, MT_class, exception
        criterion, reagents, adjust_dict = LRT_info['Criterion'], LRT_info['Reagent'], LRT_info['remap']
        if criterion:
            criterion, see_maps = eval(criterion)
            criterion_result = criterion(rxn, replacement_dict, see_maps)
            MT_class = eval(MT_class)[criterion_result] 
            exception = self.check_exception(MT_class)
            if exception:
                if exception == 'wrong_atom_mapping' and self.debug:
                    print ('#########################################################')
                    print ('MT: %s, LRT: %s', (MT_class, template))
                    print ('Reaction:', rxn)
                    print ('#########################################################')
                return rxn, template, MT_class, exception

            if adjust_dict and not np.isnan(adjust_dict):
                adjust_dict = eval(adjust_dict)[criterion_result]
            if reagents:
                reagents = eval(reagents)[criterion_result]
                rxn, replacement_dict = add_reagent(rxn, reagents, replacement_dict)
        else:
            if adjust_dict and not np.isnan(adjust_dict):
                adjust_dict = eval(adjust_dict)
            if reagents:
                reagents = eval(reagents)
                rxn, replacement_dict = add_reagent(rxn, reagents, replacement_dict)
        mechanistic_pathway = eval(self.MT_collection[MT_class]['mechanistic pathway'])
        if LRT_info['LRT_extension']:
            ext_info = eval(LRT_info['LRT_extension'])[criterion_result] if criterion else eval(LRT_info['LRT_extension'])
            if ext_info:
                for d in [replacement_dict, adjust_dict]:
                    if isinstance(d, dict):
                        for s in ext_info['ext_strings']: d.update({s : s})
                ext_dict = build_ext_dict(rxn, replacement_dict, ext_info)
        electron_path = []
        for path in mechanistic_pathway:
            if adjust_dict and not np.isnan(adjust_dict):
                path = adjust_template_atom_map(adjust_dict, path)
            electron_path.append(change_atom_map(replacement_dict, path))
        if LRT_info['LRT_extension'] and ext_info: electron_path = replace_xl(electron_path, ext_dict)
        return rxn, template, MT_class, electron_path