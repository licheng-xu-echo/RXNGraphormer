import re,os
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from rxnmapper import RXNMapper
from localmapper import localmapper
import rxngraphormer
from rxngraphormer.midgen.MechFinder import MechFinder
from rxngraphormer.utils import canonical_smiles
mapper = localmapper('cpu')  ## avoid some bugs
finder = MechFinder(collection_dir=f'{os.path.dirname(rxngraphormer.__file__)}/midgen/collections')
rxn_mapper = RXNMapper()
### Mid molecule generation
# Extract key information and store it as a collection of the form (atom1 map, atom2 map)
def get_bond_set(molecules,allowed_at_idx_lst):
    bond_set = set()
    for mol in molecules:
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetAtomMapNum()
            a2 = bond.GetEndAtom().GetAtomMapNum()
            if not a1 in allowed_at_idx_lst or not a2 in allowed_at_idx_lst:
                continue
            if a1 > a2:  # Ensure that smaller mapping numbers are placed first
                a1, a2 = a2, a1
            bond_set.add((a1, a2))
    return bond_set

def analyze_reaction_bonds(reaction_smiles,allowed_at_idx_lst):
    # Decompose reaction SMILES string into reactants and products
    reactants_smiles, products_smiles = reaction_smiles.split(">>")
    
    # Use RDKit to build molecules of reactants and products
    reactants = [Chem.MolFromSmiles(smile) for smile in reactants_smiles.split('.')]
    products = [Chem.MolFromSmiles(smile) for smile in products_smiles.split('.')]
    


    reactant_bonds = get_bond_set(reactants,allowed_at_idx_lst)
    product_bonds = get_bond_set(products,allowed_at_idx_lst)
    
    # Find broken bonds and newly formed bonds
    broken_bonds = reactant_bonds - product_bonds
    formed_bonds = product_bonds - reactant_bonds
    
    return broken_bonds, formed_bonds

def get_atommap_atomidx_map(smiles,allowed_at_idx_lst):
    mol = Chem.MolFromSmiles(smiles)
    atommap_atomidx_map = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() in allowed_at_idx_lst:
            atommap_atomidx_map[atom.GetAtomMapNum()] = atom.GetIdx()
    return atommap_atomidx_map

def ex_atmap_inf(smiles):
    # Regular expression matching pattern
    pattern = r'\[([A-Za-z0-9+-]+?):(\d+)\]'
    matches = re.findall(pattern, smiles)
    atom_mappings = {}
    for atom, map_num in matches:
        atom_mappings[int(map_num)] = atom 

    return atom_mappings

def break_bond(smiles, idx1, idx2):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Invalid SMILES string")

    if idx1 < 0 or idx1 >= mol.GetNumAtoms() or idx2 < 0 or idx2 >= mol.GetNumAtoms():
        raise ValueError("Atom index out of allowed range")

    bond = mol.GetBondBetweenAtoms(idx1, idx2)
    if not bond:
        print("If the specified key does not exist, the original molecule is returned.")
        return mol

    bond_type = bond.GetBondType()
    if bond_type == Chem.BondType.SINGLE:
        elec_num = 1
    elif bond_type == Chem.BondType.DOUBLE:
        elec_num = 2
    elif bond_type == Chem.BondType.AROMATIC:
        elec_num = 1
    else:
        raise ValueError("The specified bond is not a single, double or aromatic bond")

    editable_mol = Chem.EditableMol(mol)
    editable_mol.RemoveBond(idx1, idx2)
    modified_mol = editable_mol.GetMol()

    modified_mol.GetAtomWithIdx(idx1).SetNumRadicalElectrons(
        modified_mol.GetAtomWithIdx(idx1).GetNumRadicalElectrons() + elec_num
    )
    modified_mol.GetAtomWithIdx(idx2).SetNumRadicalElectrons(
            modified_mol.GetAtomWithIdx(idx2).GetNumRadicalElectrons() + elec_num
    )

    rdmolops.SanitizeMol(modified_mol)

    modified_smiles = Chem.MolToSmiles(modified_mol)
    return modified_mol,modified_smiles

def remove_atmmap(smiles):

    cleaned_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(re.sub(r':\d+', '', smiles)))

    return cleaned_smiles

def get_mid_smi_from_rxn(updated_reaction):
    reactants_part,products_part = updated_reaction.split(">>")
    atmap_inf = ex_atmap_inf(updated_reaction)
    broken_bonds, formed_bonds = analyze_reaction_bonds(updated_reaction,list(atmap_inf.keys()))
    #print(f"broken bonds: {broken_bonds}, formed bonds {formed_bonds}")
    all_mid_mols = []
    all_mid_smis = []
    for smiles in reactants_part.split("."):
        atommap_idx_map = get_atommap_atomidx_map(smiles,list(atmap_inf.keys()))
        for broken_bond in broken_bonds:
            if broken_bond[0] in atommap_idx_map and broken_bond[1] in atommap_idx_map:
                modified_mol,modified_smi = break_bond(smiles,atommap_idx_map[broken_bond[0]], atommap_idx_map[broken_bond[1]])
                all_mid_mols.append(modified_mol)
                all_mid_smis.append(modified_smi)

    for smiles in products_part.split("."):
        atommap_idx_map = get_atommap_atomidx_map(smiles,list(atmap_inf.keys()))
        for formed_bond in formed_bonds:
            if formed_bond[0] in atommap_idx_map and formed_bond[1] in atommap_idx_map:
                modified_mol,modified_smi = break_bond(smiles,atommap_idx_map[formed_bond[0]], atommap_idx_map[formed_bond[1]])
                all_mid_mols.append(modified_mol)
                all_mid_smis.append(modified_smi)

    concat_mid_smis = []
    for mid_smis in all_mid_smis:
        concat_mid_smis += [remove_atmmap(smi) for smi in mid_smis.split(".")]
    return concat_mid_smis

def gen_mech_mid_smi(task,confidence_threshold=0.002):

    rct_line, pdt_line = task
    rct_smi_lst = rct_line.split('.')
    pdt_smi_lst = pdt_line.split('.')
    rxn_smiles = f"{rct_line}>>{pdt_line}"
    #atmap_rxn = mapper.get_atom_map(rxn_smiles)
    atmap_rxn_res = mapper.get_atom_map(rxn_smiles, return_dict=True)
    atmap_rxn = atmap_rxn_res["mapped_rxn"]
    confident = atmap_rxn_res["confident"]
    if not confident:
        results = rxn_mapper.get_attention_guided_atom_maps([rxn_smiles])
        atmap_rxn = results[0]["mapped_rxn"]
        if results[0]["confidence"] < confidence_threshold:
            print("confidence too low")
            atmap_rxn = ""
    if atmap_rxn != "":
        updated_reaction, LRT, MT_class, electron_path = finder.get_electron_path(atmap_rxn)
        mid_smi_lst = get_mid_smi_from_rxn(updated_reaction)
        pot_mech_smi_lst = np.concatenate([remove_atmmap(smi).split('.') for smi in updated_reaction.split(">>")]).tolist()
        mech_mid_smi_lst = [smi for smi in pot_mech_smi_lst+mid_smi_lst if smi not in rct_smi_lst and smi not in pdt_smi_lst]
        mech_mid_smi = canonical_smiles('.'.join(list(set(mech_mid_smi_lst))))
    else:
        mid_smi_lst = []
        mech_mid_smi = ''
        updated_reaction = rxn_smiles
    
    return mech_mid_smi,atmap_rxn,updated_reaction