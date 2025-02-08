from datetime import datetime
from rdkit import RDLogger,Chem
from rdkit.Chem import AllChem
import logging
import os,sys
import math,torch,random
import numpy as np
from box import Box
import torch.nn.functional as F
from rdkit.Chem import rdChemReactions
from .data import ATOM_DICT,ATOM_FEAT_DIMS,gen_onehot
def setup_logger(save_dir):
    RDLogger.DisableLog("rdApp.*")
    RDLogger.DisableLog("rdApp.warning")
    os.makedirs(save_dir, exist_ok=True)
    #os.makedirs(f"{config.model.save_dir}/{config.data.data_path.split('/')[-1]}", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"{save_dir}/{dt}.log")
    sh = logging.StreamHandler(sys.stdout)
    fh.setLevel(logging.INFO)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger

def idx2token(ids,vocab_rev):
    """
    Recover tokens string from a list of indices
    """
    return ' '.join([vocab_rev[idx] for idx in ids])

def param_count(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

def param_norm(m):
    return math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))

def grad_norm(m):
    return math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

def get_lr(optimizer):
    lr_lst = []
    for param_group in optimizer.param_groups:
        lr_lst.append(str(round(param_group["lr"],8)))
    return ",".join(lr_lst)

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_seq_acc(t1, t2):
    ## seq of token acc
    ## seq of token mask
    both_zero = (t1 == 0) & (t2 == 0)
    both_one = (t1 == 1) & (t2 == 1)
    result = both_zero | both_one
    return result.all(dim=1).float()

def get_random_shuffle_smiles(init_smi, rxn_template="[C,N,O:1]-[*:2]~[*:3]-[*:4]>>[*:4]-[*:2]~[*:3]-[C,N,O:1]", sel_num=2, random_seed=42):
    """
    get random shuffle smiles with given rxn_template and init_smi
    init_smi : initial smiles, e.g. "CCc1ncNccc1O"
    """
    np.random.seed(random_seed)
    rxn1 = AllChem.ReactionFromSmarts(rxn_template)
    rxn_res = rxn1.RunReactants((AllChem.AddHs(Chem.MolFromSmiles(init_smi)),))
    if len(rxn_res) > 0:
        rxn_smi_res = []
        for item in rxn_res:
            try:
                _smi = Chem.MolToSmiles(Chem.RemoveHs(item[0]))
                if _smi == init_smi:
                    continue
                rxn_smi_res.append(_smi)
            except:
                continue
        rxn_smi_res = list(set(rxn_smi_res))
        sel_idx = np.random.choice(len(rxn_smi_res),min(sel_num,len(rxn_smi_res)),
                                   replace=False)
        sel_smi = [rxn_smi_res[idx] for idx in sel_idx]
    else:
        sel_smi = []
    return sel_smi

def gen_truth_false_rxn_smi(rct_pdt_smi__split,rxn_template="[C,N,O:1]-[*:2]~[*:3]-[*:4]>>[*:4]-[*:2]~[*:3]-[C,N,O:1]",
                            sel_num=2, random_seed=42):
    # default split=False
    rct_smi,pdt_smi,split = rct_pdt_smi__split
    false_pdt_smi_lst = get_random_shuffle_smiles(pdt_smi,rxn_template=rxn_template,sel_num=sel_num,random_seed=random_seed)
    if not split:
        truth_rxn_smi = canonical_smiles(f"{rct_smi}.{pdt_smi}")
        false_rxn_smi_lst = []
        for f_pdt_smi in false_pdt_smi_lst:
            false_rxn_smi = canonical_smiles(f"{rct_smi}.{f_pdt_smi}")
            false_rxn_smi_lst.append(false_rxn_smi)
    else:
        rct_smi = canonical_smiles(rct_smi)
        pdt_smi = canonical_smiles(pdt_smi)
        truth_rxn_smi = f'{rct_smi}>>{pdt_smi}'
        false_rxn_smi_lst = []
        for f_pdt_smi in false_pdt_smi_lst:
            f_pdt_smi = canonical_smiles(f_pdt_smi)
            false_rxn_smi = f'{rct_smi}>>{f_pdt_smi}'
            false_rxn_smi_lst.append(false_rxn_smi)
    return [truth_rxn_smi],false_rxn_smi_lst

def pad_feat(feat, batch, num_features):
    device = feat.device
    batch_size = batch.max() + 1
    batch = batch.to(device)

    counts = torch.bincount(batch, minlength=batch_size)

    max_length = counts.max().item()

    padded_feat = torch.zeros(batch_size, max_length, num_features).to(device)

    current_idx = torch.zeros(batch_size, dtype=torch.long).to(device)
    for idx, b in enumerate(batch):
        padded_feat[b, current_idx[b]] = feat[idx]
        current_idx[b] += 1

    return padded_feat

def update_batch_idx(mol_index,device):

    mol_tensors = [torch.tensor(m, device=device) for m in mol_index]

    max_values = torch.tensor([torch.max(m).item() for m in mol_tensors], device=device)
    offsets = torch.cumsum(max_values + 1, dim=0) - (max_values + 1)

    batch_mol_index = torch.cat([m + offset for m, offset in zip(mol_tensors, offsets)])
    batch_sizes = max_values + 1
    batch_ = torch.cat([torch.full((size,), i, dtype=torch.long, device=device) for i, size in enumerate(batch_sizes)])

    return batch_mol_index, batch_

def get_sin_encodings(rel_pos_buckets, model_dim):
    pe = torch.zeros(rel_pos_buckets + 1, model_dim)
    position = torch.arange(0, rel_pos_buckets).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, model_dim, 2, dtype=torch.float) *
                          -(math.log(10000.0) / model_dim)))
    pe[:-1, 0::2] = torch.sin(position.float() * div_term)          # leaving last "position" as padding
    pe[:-1, 1::2] = torch.cos(position.float() * div_term)

    return pe

def scaled_dot_product_attention(query, key, value, query_mask=None, key_mask=None, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(dim_k)
    if query_mask is not None and key_mask is not None:
        mask = torch.bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float("inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)

def index_scatter(sub_data, all_data, index):
    d0, d1 = all_data.size()
    buf = torch.zeros_like(all_data).scatter_(0, index.repeat(d1, 1).t(), sub_data)
    mask = torch.ones(d0, device=all_data.device).scatter_(0, index, 0)

    return all_data * mask.unsqueeze(-1) + buf

def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))

    return target.view(final_size)

def update_dict_key(old_state_dict,prefix='module.',compat=True):
    new_state_dict = {}
    for key in old_state_dict.keys():
        if key.startswith(prefix):
            new_key = key[len(prefix):] 
            new_state_dict[new_key] = old_state_dict[key]  
        else:
            new_state_dict[key] = old_state_dict[key]
    if compat:
        new_state_dict_ = {}
        for key in list(new_state_dict.keys()):
            if key.startswith("rct_encoder.x_embedding") or key.startswith("rct_encoder.gnns") or key.startswith("rct_encoder.batch_norms") or \
            key.startswith("pdt_encoder.x_embedding") or key.startswith("pdt_encoder.gnns") or key.startswith("pdt_encoder.batch_norms"):
                name_blks = key.split(".")
                name_blks.insert(1,"rxn_graph_encoder")
                new_state_dict_[".".join(name_blks)] = new_state_dict[key]
            else:
                new_state_dict_[key] = new_state_dict[key]
        new_state_dict = new_state_dict_
    return new_state_dict

def mod_mol(mol1,mol2=None,type="add", bond_idx=[0,1], bond_type=Chem.BondType.SINGLE):
    type = type.lower()
    if type == "add":
        rwmol = Chem.RWMol(mol1)
        rwmol.InsertMol(mol2)
        rwmol.AddBond(bond_idx[0],bond_idx[1]+len(mol1.GetAtoms()),bond_type)
        new_mol = rwmol.GetMol()
    elif type == "remove":
        rwmol = Chem.RWMol(mol1)
        rwmol.RemoveBond(bond_idx[0],bond_idx[1])
        new_mol = rwmol.GetMol()
    return new_mol

def gen_mid_mols(rct_blks, pdt_blks):
    rct_mols = [Chem.MolFromSmiles(item) for item in rct_blks]
    pdt_mols = [Chem.MolFromSmiles(item) for item in pdt_blks]

    # 反应 SMILES
    reaction_smiles = ".".join(rct_blks) + ">>" + ".".join(pdt_blks)

    # 解析反应
    reaction = rdChemReactions.ReactionFromSmarts(reaction_smiles)
    reactants = [reaction.GetReactantTemplate(i) for i in range(reaction.GetNumReactantTemplates())]
    products = [reaction.GetProductTemplate(i) for i in range(reaction.GetNumProductTemplates())]

    # 找出断裂的键
    reactant_bonds = set()
    rct_mol_atom_idx_map = {}
    for rct_idx,reactant in enumerate(reactants):
        atoms = list(reactant.GetAtoms())
        rct_mol_atom_idx_map.update({atom.GetAtomMapNum():[rct_idx,atom.GetIdx()] for atom in atoms})
        for bond in reactant.GetBonds():
            begin_idx = bond.GetBeginAtom().GetAtomMapNum()
            end_idx = bond.GetEndAtom().GetAtomMapNum()
            if begin_idx and end_idx:
                reactant_bonds.add(frozenset([begin_idx, end_idx]))

    product_bonds = set()
    pdt_mol_atom_idx_map = {}
    for pdt_idx,product in enumerate(products):
        atoms = list(product.GetAtoms())
        pdt_mol_atom_idx_map.update({atom.GetAtomMapNum():[pdt_idx,atom.GetIdx()] for atom in atoms})
        for bond in product.GetBonds():
            begin_idx = bond.GetBeginAtom().GetAtomMapNum()
            end_idx = bond.GetEndAtom().GetAtomMapNum()
            if begin_idx and end_idx:
                product_bonds.add(frozenset([begin_idx, end_idx]))

    broken_bonds = reactant_bonds - product_bonds
    new_bonds = product_bonds - reactant_bonds

    broken_bonds = [list(bond) for bond in broken_bonds]
    new_bonds = [list(bond) for bond in new_bonds]

    broken_bonds_in_rct = [[rct_mol_atom_idx_map[bond[0]],rct_mol_atom_idx_map[bond[1]]] for bond in broken_bonds]
    broken_bonds_in_pdt = [[pdt_mol_atom_idx_map[bond[0]],pdt_mol_atom_idx_map[bond[1]]] for bond in broken_bonds]
    new_bonds_in_rct = [[rct_mol_atom_idx_map[bond[0]],rct_mol_atom_idx_map[bond[1]]] for bond in new_bonds if bond[0] in rct_mol_atom_idx_map and bond[1] in rct_mol_atom_idx_map]
    new_bonds_in_pdt = [[pdt_mol_atom_idx_map[bond[0]],pdt_mol_atom_idx_map[bond[1]]] for bond in new_bonds if bond[0] in pdt_mol_atom_idx_map and bond[1] in pdt_mol_atom_idx_map]

    all_mid_mols = []
    for bond in new_bonds_in_rct:
        try:
            mid_mol = mod_mol(rct_mols[bond[0][0]],rct_mols[bond[1][0]],"add",[bond[0][1],bond[1][1]])
        except:
            continue
        all_mid_mols.append(mid_mol)
    for bond in new_bonds_in_pdt:
        try:
            mid_mol = mod_mol(pdt_mols[bond[0][0]],None,"remove",[bond[0][1],bond[1][1]])
        except:
            continue
        all_mid_mols.append(mid_mol)

    for bond in broken_bonds_in_rct:
        try:
            mid_mol = mod_mol(rct_mols[bond[0][0]],rct_mols[bond[1][0]],"remove",[bond[0][1],bond[1][1]])
        except:
            continue
        all_mid_mols.append(mid_mol)
    for bond in broken_bonds_in_pdt:
        try:
            mid_mol = mod_mol(pdt_mols[bond[0][0]],pdt_mols[bond[1][0]],"add",[bond[0][1],bond[1][1]])
        except:
            continue
        all_mid_mols.append(mid_mol)
    #mod_mol(rct_mols[0],rct_mols)
    return all_mid_mols

def canonical_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return ""

    original_smi = smiles
    viewed_smi = {original_smi: 1}
    while original_smi != (
        canonical_smi := Chem.CanonSmiles(original_smi, useChiral=True)
    ) and (canonical_smi not in viewed_smi or viewed_smi[canonical_smi] < 2):
        original_smi = canonical_smi
        if original_smi not in viewed_smi:
            viewed_smi[original_smi] = 1
        else:
            viewed_smi[original_smi] += 1
    else:
        return original_smi
    
def align_config(input_dict, type_="classifier"):
    assert type_ in ["classifier", "regressor", "sequence_generation"]
    """
    class RXNGClassifier(torch.nn.Module):
    def __init__(self,gnum_layer,tnum_layer,onum_layer,emb_dim=256,JK="last",output_size=2,drop_ratio=0.0,
                 num_heads=4,gnn_type="gcn",bond_feat_red="mean",gnn_aggr='add',node_readout='sum',
                 trans_readout='mean',graph_pooling='attention',attn_drop_ratio=0.0,encoder_filter_size=2048,
                 rel_pos_buckets=11,rel_pos="emb_only",split_process=False,split_merge_method="all",output_act_func='relu'):

                 
    class RXNGRegressor(torch.nn.Module):
        def __init__(self,gnum_layer,tnum_layer,onum_layer,emb_dim,JK="last",output_size=1,drop_ratio=0.0,
                    num_heads=4,gnn_type="gcn",bond_feat_red="mean",gnn_aggr='add',node_readout='sum',
                    trans_readout='mean',graph_pooling='attention',attn_drop_ratio=0.0,encoder_filter_size=2048,
                    rel_pos_buckets=11,rel_pos="emb_only",pretrained_encoder=None,pretrained_rct_encoder=None,pretrained_pdt_encoder=None,
                    output_norm=False,split_process=False,use_mid_inf=False,interaction=False,interaction_layer_num=3,pretrained_mid_encoder=None,mid_iteract_method="attention",
                    split_merge_method="all",output_act_func='relu',rct_batch_norm=True,pdt_batch_norm=True,mid_batch_norm=True,mid_layer_num=1):
    """
    class_dict = {"emb_dim":256,"JK":"last","output_size":2,"drop_ratio":0.0,
                 "num_heads":4,"gnn_type":"gcn","bond_feat_red":"mean","gnn_aggr":'add',"node_readout":'sum',
                 "trans_readout":'mean',"graph_pooling":'attention',"attn_drop_ratio":0.0,"encoder_filter_size":2048,
                 "rel_pos_buckets":11,"rel_pos":"emb_only","split_process":False,"split_merge_method":"all","output_act_func":'relu'}
    regress_dict = {"JK":"last","output_size":1,"drop_ratio":0.0,
                    "num_heads":4,"gnn_type":"gcn","bond_feat_red":"mean","gnn_aggr":'add',"node_readout":'sum',
                    "trans_readout":'mean',"graph_pooling":'attention',"attn_drop_ratio":0.0,"encoder_filter_size":2048,
                    "rel_pos_buckets":11,"rel_pos":"emb_only","output_norm":False,"split_process":False,"use_mid_inf":False,
                    "interaction":False,"interaction_layer_num":3,"mid_iteract_method":"attention","split_merge_method":"all",
                    "output_act_func":'relu',"rct_batch_norm":True,"pdt_batch_norm":True,"mid_batch_norm":True,"mid_layer_num":1}

    if type_ == "classifier":
        class_dict.update(input_dict)
        return Box(class_dict)
    elif type_ == "regressor":
        regress_dict.update(input_dict)
        return Box(regress_dict)

def add_empty_node_and_edge(batch_data):
    
    empty_node = [ATOM_DICT.get("*", ATOM_DICT["unk"]),
                    0, ## Degree
                    0, ## Formal Charge
                    0, ## Hybridization
                    0, ## Chiral Tag
                    0, ## Is Aromatic
                    0, ## Total Valence
                    0, ## Total Num Hs
                    0, ## RS Tag
                    ]
    oh_empty_node = torch.from_numpy(gen_onehot(empty_node,ATOM_FEAT_DIMS)).unsqueeze(0).long().to(batch_data.x_oh.device)
    oh_empty_edge = torch.zeros(1,batch_data.edge_oh_attr.shape[1]).long().to(batch_data.x_oh.device)
    a_graphs_empty = torch.zeros(1,batch_data.a_graphs.shape[1]).long().to(batch_data.x_oh.device)
    b_graphs_empty = torch.zeros(1,batch_data.b_graphs.shape[1]).long().to(batch_data.x_oh.device)

    x_oh_merge_ = torch.cat([oh_empty_node, batch_data.x_oh], dim=0)
    batch_data.edge_oh_attr[:,:2] = batch_data.edge_oh_attr[:,:2].clone() + 1 
    edge_oh_attr_merge_ = torch.cat([oh_empty_edge, batch_data.edge_oh_attr], dim=0)
    a_graphs_merge_ = torch.cat([a_graphs_empty, batch_data.a_graphs+1], dim=0)
    b_graphs_merge_ = torch.cat([b_graphs_empty, batch_data.b_graphs+1], dim=0)
    
    a_graphs_merge_[a_graphs_merge_>=999999999] = 0
    b_graphs_merge_[b_graphs_merge_>=999999999] = 0
    
    #a_graphs_merge_ = a_graphs_merge_.numpy()
    #b_graphs_merge_ = b_graphs_merge_.numpy()
    ## drop trailing columns of 0
    #column_idx = np.argwhere(np.all(a_graphs_merge_[..., :] == 0, axis=0))
    #a_graphs_merge_ = a_graphs_merge_[:, :column_idx[0, 0] + 1]       # drop trailing columns of 0, leaving only 1 last column of 0
    
    #column_idx = np.argwhere(np.all(b_graphs_merge_[..., :] == 0, axis=0))
    #b_graphs_merge_ = b_graphs_merge_[:, :column_idx[0, 0] + 1]       # drop trailing columns of 0, leaving only 1 last column of 0
    
    batch_data.x_oh = x_oh_merge_
    batch_data.edge_oh_attr = edge_oh_attr_merge_
    batch_data.a_graphs = a_graphs_merge_ # torch.from_numpy(a_graphs_merge_).long()
    batch_data.b_graphs = b_graphs_merge_ # torch.from_numpy(b_graphs_merge_).long()
    return batch_data

def add_dense_empty_node_edge(batch_data):
    empty_node = torch.tensor([[ATOM_DICT.get("*", ATOM_DICT["unk"]),
                    0, ## Degree
                    0, ## Formal Charge
                    0, ## Hybridization
                    0, ## Chiral Tag
                    0, ## Is Aromatic
                    0, ## Total Valence
                    0, ## Total Num Hs
                    0, ## RS Tag
                    ]]).to(batch_data.x.device)
    empty_edge = torch.zeros(1,batch_data.edge_attr.shape[1]).long().to(batch_data.x.device)
    x_merge_ = torch.cat([empty_node, batch_data.x], dim=0)
    edge_merge_ = torch.cat([empty_edge, batch_data.edge_attr], dim=0)
    edge_index_ = torch.cat([torch.zeros(2,1).long().to(batch_data.x.device), batch_data.edge_index+1], dim=1)
    # batch_data.batch = torch.cat([torch.tensor([-1]).long().to(batch_data.x.device), batch_data.batch], dim=0) + 1
    
    batch_data.mol_index = [[0]] + batch_data.mol_index
    batch_data.x = x_merge_
    batch_data.edge_attr = edge_merge_
    batch_data.edge_index = edge_index_
    return batch_data

def inchi_to_smiles(inchi):
    try:
        mol = Chem.MolFromInchi(inchi)
        if mol is not None:
            return canonical_smiles(Chem.MolToSmiles(mol))
        else:
            return None
    except:
        return None