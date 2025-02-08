import torch,glob,os,re
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem,RDLogger
from torch_geometric.data import Batch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from sklearn.utils import shuffle
from multiprocessing import Pool
from torch_geometric.data.separate import separate
from .ext_feat import ext_feat_gen
RDLogger.DisableLog('rdApp.*')

NUM_ATOM_TYPE = 65
NUM_DEGRESS_TYPE = 11
NUM_FORMCHRG_TYPE = 5
NUM_HYBRIDTYPE = 6
NUM_CHIRAL_TYPE = 3
NUM_AROMATIC_NUM = 2
NUM_VALENCE_TYPE = 7
NUM_Hs_TYPE = 5
NUM_RS_TPYE = 3

NUM_BOND_TYPE = 6
NUM_BOND_DIRECTION = 3
NUM_BOND_STEREO = 3
NUM_BOND_INRING = 2
NUM_BOND_ISCONJ = 2
ATOM_FEAT_DIMS = [NUM_ATOM_TYPE,NUM_DEGRESS_TYPE,NUM_FORMCHRG_TYPE,NUM_HYBRIDTYPE,NUM_CHIRAL_TYPE,
                    NUM_AROMATIC_NUM,NUM_VALENCE_TYPE,NUM_Hs_TYPE,NUM_RS_TPYE]
BOND_FEAT_DIME = [NUM_BOND_TYPE,NUM_BOND_DIRECTION,NUM_BOND_STEREO,NUM_BOND_INRING,NUM_BOND_ISCONJ]
ATOM_LST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
             'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
             'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
             'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir',
             'Ce', 'Gd', 'Ga', 'Cs', '*', 'unk']
ATOM_DICT = {symbol: i for i, symbol in enumerate(ATOM_LST)}
MAX_NEIGHBORS = 10
CHIRAL_TAG_LST = [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                  Chem.rdchem.ChiralType.CHI_UNSPECIFIED]
CHIRAL_TAG_DICT = {ct: i for i, ct in enumerate(CHIRAL_TAG_LST)}
HYBRIDTYPE_LST = [Chem.rdchem.HybridizationType.SP,Chem.rdchem.HybridizationType.SP2,Chem.rdchem.HybridizationType.SP3,
                  Chem.rdchem.HybridizationType.SP3D,Chem.rdchem.HybridizationType.SP3D2,Chem.rdchem.HybridizationType.UNSPECIFIED]
HYBRIDTYPE_DICT = {hb: i for i, hb in enumerate(HYBRIDTYPE_LST)}
VALENCE_LST = [0, 1, 2, 3, 4, 5, 6]
VALENCE_DICT = {vl: i for i, vl in enumerate(VALENCE_LST)}
NUM_Hs_LST = [0, 1, 3, 4, 5]
NUM_Hs_DICT = {nH: i for i, nH in enumerate(NUM_Hs_LST)}
BOND_TYPE_LST = [Chem.rdchem.BondType.SINGLE,
                 Chem.rdchem.BondType.DOUBLE,
                 Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC,
                 Chem.rdchem.BondType.DATIVE,
                 Chem.rdchem.BondType.UNSPECIFIED]
BOND_DIR_LST = [ # only for double bond stereo information
                Chem.rdchem.BondDir.NONE,
                Chem.rdchem.BondDir.ENDUPRIGHT,
                Chem.rdchem.BondDir.ENDDOWNRIGHT]
BOND_STEREO_LST = [Chem.rdchem.BondStereo.STEREONONE,
                   Chem.rdchem.BondStereo.STEREOE,
                   Chem.rdchem.BondStereo.STEREOZ,
                   ]
FORMAL_CHARGE_LST = [-1, -2, 1, 2, 0]
FC_DICT = {fc: i for i, fc in enumerate(FORMAL_CHARGE_LST)}
RS_TAG_LST = ["R","S","None"]
RS_TAG_DICT = {rs: i for i, rs in enumerate(RS_TAG_LST)}

def gen_onehot(features,feature_dims):
    assert len(features) == len(feature_dims), "size of 'features' and 'feature_dims' should be same"
    onehot = []
    for feat,feat_dim in zip(features,feature_dims):
        f_oh = np.zeros(feat_dim)
        f_oh[feat] = 1
        onehot.append(f_oh)
    return np.concatenate(onehot)

def mol2graphinfo(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    """
    # atoms
    atom_features_list = []
    atom_oh_features_list = []
    atom_mass_list = []
    
    atom_feat_dims = [NUM_ATOM_TYPE,NUM_DEGRESS_TYPE,NUM_FORMCHRG_TYPE,NUM_HYBRIDTYPE,NUM_CHIRAL_TYPE,
                      NUM_AROMATIC_NUM,NUM_VALENCE_TYPE,NUM_Hs_TYPE,NUM_RS_TPYE]
    bond_feat_dims = [NUM_BOND_TYPE,NUM_BOND_DIRECTION,NUM_BOND_STEREO,NUM_BOND_INRING,NUM_BOND_ISCONJ]
    
    for atom in mol.GetAtoms():
        atom_feature = [ATOM_DICT.get(atom.GetSymbol(), ATOM_DICT["unk"]),
                        min(atom.GetDegree(),MAX_NEIGHBORS),
                        FC_DICT.get(atom.GetFormalCharge(), 4),
                        HYBRIDTYPE_DICT.get(atom.GetHybridization(), 5),
                        CHIRAL_TAG_DICT.get(atom.GetChiralTag(),2),
                        int(atom.GetIsAromatic()),
                        VALENCE_DICT.get(atom.GetTotalValence(), 6),
                        NUM_Hs_DICT.get(atom.GetTotalNumHs(), 4),
                        RS_TAG_DICT.get(atom.GetPropsAsDict().get("_CIPCode", "None"), 2)]
        atom_oh_feature = gen_onehot(atom_feature,atom_feat_dims)
        atom_mass = atom.GetMass()
        atom_features_list.append(atom_feature)
        atom_oh_features_list.append(atom_oh_feature)
        atom_mass_list.append(atom_mass)
    x = torch.tensor(np.array(atom_features_list),dtype=torch.long)
    x_oh = torch.tensor(np.array(atom_oh_features_list),dtype=torch.long)
    atom_mass = torch.from_numpy(np.array(atom_mass_list))
    # bonds
    num_bond_features = 5   # bond type, bond direction, bond stereo, isinring, isconjugated
    num_oh_bond_features = sum(bond_feat_dims)
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        edge_oh_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [BOND_TYPE_LST.index(bond.GetBondType()),
                            BOND_DIR_LST.index(bond.GetBondDir()),
                            BOND_STEREO_LST.index(bond.GetStereo()),
                            int(bond.IsInRing()),
                            int(bond.GetIsConjugated())]
            edge_oh_feature = gen_onehot(edge_feature,bond_feat_dims)
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edge_oh_features_list.append(edge_oh_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
            edge_oh_features_list.append(edge_oh_feature)

        edge_index = np.array(edges_list).T
        
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
        edge_oh_attr = torch.tensor(np.array(edge_oh_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = np.empty((2,0),dtype=np.int32)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
        edge_oh_attr = torch.empty((0,num_oh_bond_features), dtype=torch.long)
    #data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    a_graphs, edge_dict = get_agraph(len(x),edge_index)
    b_graphs = get_bgraphs(edge_index,edge_dict)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return x,edge_index,edge_attr,atom_mass,x_oh,edge_oh_attr,a_graphs,b_graphs

def calc_graph_distance(atom_feat,edge_index,task="retrosynthesis"):
    """
    task : "forward_prediction" or "retrosynthesis".
    g_dist = torch.from_numpy(calc_graph_distance(x_merge,edge_index_merge,task=task))
    """
    a_length = atom_feat.shape[0]
    adjacency = np.zeros((a_length, a_length), dtype=np.int32)

    for u,v in edge_index.T:
        adjacency[u, v] = 1
        
    # compute graph distance
    distance = adjacency.copy()
    shortest_paths = adjacency.copy()
    path_length = 2
    stop_counter = 0
    non_zeros = 0


    while 0 in distance:
        shortest_paths = np.matmul(shortest_paths, adjacency)
        shortest_paths = path_length * (shortest_paths > 0)
        new_distance = distance + (distance == 0) * shortest_paths

        # if np.count_nonzero(new_distance) == np.count_nonzero(distance):
        if np.count_nonzero(new_distance) <= non_zeros:
            stop_counter += 1
        else:
            non_zeros = np.count_nonzero(new_distance)
            stop_counter = 0

        if task == "forward_prediction" and stop_counter == 3:
            break

        distance = new_distance
        path_length += 1
        
    # bucket
    distance[(distance > 8) & (distance < 15)] = 8
    distance[distance >= 15] = 9
    if task == "forward_prediction":
        distance[distance == 0] = 10
        
    # reset diagonal
    np.fill_diagonal(distance, 0)

    return distance

def calc_batch_graph_distance(batch, edge_index, task):
    ## adapted from https://github.com/coleygroup/Graph2SMILES
    assert task in ["forward_prediction", "retrosynthesis"], "task must be 'forward_prediction' or 'retrosynthesis'"
    num_nodes = batch.size(0)
    num_graphs = batch.max().item() + 1
    max_len = int(torch.bincount(batch).max())

    # Create a large adjacency matrix for all nodes
    full_adj_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.int32, device=batch.device)
    src = edge_index[0]
    dest = edge_index[1]
    full_adj_matrix[src, dest] = 1

    # Create a mask to separate each graph's adjacency matrix
    graph_masks = batch.unsqueeze(0) == batch.unsqueeze(1)
    
    # Use the mask to get separate adjacency matrices
    adj_matrices = full_adj_matrix * graph_masks.float()

    distances = []
    for i in range(num_graphs):
        # Extract the adjacency matrix for each graph
        graph_mask = batch == i
        adj_matrix = adj_matrices[graph_mask][:, graph_mask]

        # Compute the shortest paths using matrix power method
        max_power = max_len 
        identity = torch.eye(adj_matrix.size(0), 
                             dtype=torch.int32, device=adj_matrix.device)
        dist = torch.full_like(adj_matrix, float('inf'))
        dist[adj_matrix > 0] = 1
        power = adj_matrix.clone()

        for _ in range(2, max_power + 1):
            power = torch.matmul(power, adj_matrix)
            new_paths = (power > 0) & (dist == float('inf'))
            dist[new_paths] = _

        # Apply task-specific transformations
        dist[(dist > 8) & (dist < 15)] = 8  # Adjust these numbers based on your bucketing
        dist[dist >= 15] = 9
        if task == "forward_prediction":
            dist[dist == 0] = 10
        dist.fill_diagonal_(0)

        # Padding to maximum size
        padded_dist = torch.full((max_len, max_len), 11 if task == "forward_prediction" else 10, 
                                 dtype=torch.int32, device=dist.device)
        actual_size = dist.size(0)
        padded_dist[:actual_size, :actual_size] = dist
        distances.append(padded_dist)

    distances = torch.stack(distances)
    return distances

def get_agraph(node_num,edge_index):
    # edge_index : numpy.ndarray
    a_graphs = [[] for _ in range(node_num)]
    edge_dict = {}
    # edge iteration to get (dense) bond features
    for u, v, in edge_index.T:
        eid = len(edge_dict)
        edge_dict[(u, v)] = eid
        a_graphs[v].append(eid)
    for a_graph in a_graphs:
        while len(a_graph) < 11:            
            a_graph.append(1e9)
    a_graphs = torch.tensor(a_graphs).long()
    return a_graphs, edge_dict

def get_bgraphs(edge_index,edge_dict):
    # edge_index : numpy.ndarray
    src_tgt_lst_map = {}
    for src,tgt in edge_index.T:
        if not src in src_tgt_lst_map:
            src_tgt_lst_map[src] = [tgt]
        else:
            src_tgt_lst_map[src].append(tgt)
            

    # second edge iteration to get neighboring edges (after edge_dict is updated fully)
    b_graphs = [[] for _ in range(len(edge_index.T))]
    for u, v, in edge_index.T:
        u = int(u)
        v = int(v)
        eid = edge_dict[(u, v)]

        for w in src_tgt_lst_map[u]:
            if not w == v:
                b_graphs[eid].append(edge_dict[(w, u)])
                
    for b_graph in b_graphs:
        while len(b_graph) < 11:           
            b_graph.append(1e9)
    b_graphs = torch.tensor(b_graphs).long()
    return b_graphs

class RXNG2SDataset(InMemoryDataset):
    def __init__(self, root, src_file, tgt_file, vocab_file, max_length=1024, 
                 transform=None, pre_transform=None,train=True,trunck=None,
                 num_worker=8,multi_process=True,oh=True):
        self.root = root
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.vocab_file = f'{root}/{vocab_file}'
        self.vocab = load_vocab(self.vocab_file)
        self.max_length = max_length
        self.train = train
        self.trunck = trunck if trunck is not None and trunck != 0 else None
        self.num_worker = num_worker
        self.multi_process = multi_process
        self.oh = oh
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.src_file,self.tgt_file]
    @property
    def processed_file_names(self):
        if self.oh:
            return [f'{self.src_file[:-4]}_{self.trunck}.pt']
        else:
            return [f'{self.src_file[:-4]}_{self.trunck}_mini.pt']
    
    def process(self):
        self.data_list = []
        with open(f'{self.root}/{self.src_file}','r') as fr:
            self.src_lines = fr.readlines()
        with open(f'{self.root}/{self.tgt_file}','r') as fr:
            self.tgt_lines = fr.readlines()
        assert len(self.src_lines) == len(self.tgt_lines), 'src and tgt file length not equal'
        
        if self.trunck is not None:
            self.src_lines = self.src_lines[:self.trunck]
            self.tgt_lines = self.tgt_lines[:self.trunck]
        if self.multi_process:
            print(f"[INFO] {self.num_worker} workers are used to process data...")
            pool = Pool(self.num_worker)
            tasks = ((self.src_lines[i], self.tgt_lines[i],self.vocab,self.max_length) for i in range(len(self.src_lines)))
            results = []
            for result in tqdm(pool.imap(get_rxn_seq_info, tasks), total=len(self.src_lines)):
                results.append(result)
            pool.close()
            pool.join()
            for rxn_inf in results:
                if rxn_inf is None:
                    continue
                x_merge,edge_index_merge,edge_attr_merge,mol_index,atom_mass_merge,x_oh_merge,edge_oh_attr_merge,\
                a_graphs_merge,b_graphs_merge,tgt_token_ids,tgt_lens = rxn_inf
                if self.oh:
                    data = Data(x=x_merge,edge_index=edge_index_merge,edge_attr=edge_attr_merge,x_oh=x_oh_merge,
                                edge_oh_attr=torch.cat((edge_index_merge.T, edge_oh_attr_merge), dim=1),mol_index=mol_index,tgt_token_ids=tgt_token_ids,tgt_lens=tgt_lens,a_graphs=a_graphs_merge,
                                b_graphs=b_graphs_merge)
                else:
                    data = Data(x=x_merge,edge_index=edge_index_merge,edge_attr=edge_attr_merge,
                                mol_index=mol_index,tgt_token_ids=tgt_token_ids,tgt_lens=tgt_lens)
                self.data_list.append(data)
        else:
            for i in tqdm(range(len(self.src_lines))):
                src_line,tgt_line = self.src_lines[i],self.tgt_lines[i]
                rxn_inf = get_rxn_seq_info((src_line, tgt_line, self.vocab, self.max_length))
                if rxn_inf is None:
                    continue
                x_merge,edge_index_merge,edge_attr_merge,mol_index,atom_mass_merge,x_oh_merge,edge_oh_attr_merge,\
                a_graphs_merge,b_graphs_merge,tgt_token_ids,tgt_lens = rxn_inf
                if self.oh:
                    data = Data(x=x_merge,edge_index=edge_index_merge,edge_attr=edge_attr_merge,x_oh=x_oh_merge,
                                edge_oh_attr=torch.cat((edge_index_merge.T, edge_oh_attr_merge), dim=1),mol_index=mol_index,tgt_token_ids=tgt_token_ids,tgt_lens=tgt_lens,a_graphs=a_graphs_merge,
                                b_graphs=b_graphs_merge)
                else:
                    data = Data(x=x_merge,edge_index=edge_index_merge,edge_attr=edge_attr_merge,
                            mol_index=mol_index,tgt_token_ids=tgt_token_ids,tgt_lens=tgt_lens)
                self.data_list.append(data)
                
        data, slices = self.collate(self.data_list)
        print(f'[INFO] {len(self.data_list)} Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def download(self):
        pass

class MultiRXNDataset(InMemoryDataset):
    def __init__(self, root, name_regrex='pretrain_rxn_dataset_test_0_*.csv', raw_data=[],transform=None, pre_transform=None,
                 trunck=None,file_num_trunck=0,task="regression",num_worker=8,multi_process=False,ext_feat=False,ext_feat_type='Morgan',
                 ext_feat_param={'radius':2,'nBits':2048,'useChirality':True}, oh=False, mul_ext_readout='mean',name_tag='',
                 start_idx=None,end_idx=None):
        '''
        Multi-process is useless in this procedure
        '''
        
        self.name_regrex = name_regrex
        #print(glob.glob(f"{root}/{self.name_regrex}"))
        self.raw_data_files = sorted(glob.glob(f"{root}/{self.name_regrex}"),key=lambda x:int(x.split('.')[-2].split('_')[-1]))
        print(f"[INFO] There are {len(self.raw_data_files)} data files in total")
        self.raw_data = raw_data
        self.trunck = trunck if trunck is not None and trunck != 0 else None
        self.file_num_trunck = file_num_trunck
        if self.file_num_trunck != 0:
            self.raw_data_files = self.raw_data_files[:self.file_num_trunck]
            print(f"[INFO] {self.file_num_trunck} data files will be used")
        else:
            print(f"[INFO] All data {len(self.raw_data_files)} files will be used")
        if start_idx is not None and end_idx is not None:
            self.raw_data_files = self.raw_data_files[start_idx:end_idx]
            print(f"[INFO] {len(self.raw_data_files)} (from {start_idx} to {end_idx}) data files will be used")
        self.task = task
        self.num_worker = num_worker
        self.multi_process = multi_process
        self.oh = oh
        self.ext_feat = ext_feat
        self.ext_feat_type = ext_feat_type.lower()
        self.ext_feat_param = ext_feat_param
        self.mul_ext_readout = mul_ext_readout.lower()
        self.name_tag = name_tag
        super().__init__(root, transform, pre_transform)
        self.data_lst = []
        self.slices_lst = []
        self.data_num_lst = [0]
        for processed_path in self.processed_paths:
            data, slices = torch.load(processed_path)
            self.data_lst.append(data)
            self.slices_lst.append(slices)
            self.data_num_lst.append(self.data_num_lst[-1]+len(slices['x'])-1)
        #self.data, self.slices = torch.load(self.processed_paths[0])
        self.data_num_lst = self.data_num_lst[1:]
    @property
    def raw_file_names(self):
        return [os.path.basename(file) for file in self.raw_data_files]

    @property
    def processed_file_names(self):
        if not self.ext_feat:
            return [f"{os.path.basename(file).split('.')[0]}_{self.trunck}_{self.name_tag}.pt" for file in self.raw_data_files]
        else:
            return [f"{os.path.basename(file).split('.')[0]}_{self.trunck}_{self.ext_feat_type}_{self.mul_ext_readout}_{self.name_tag}.pt" for file in self.raw_data_files]
    
    def process(self):
        for idx,raw_data_f in enumerate(self.raw_data_files):
            if os.path.exists(self.processed_paths[idx]):
                print(f'[INFO] {self.processed_paths[idx]} already exists, skip it...')
                continue
            print(f'[INFO] {raw_data_f} is processing...')
            data_list = []
            with open(raw_data_f, 'r') as f:
                rxn_smi_tgt_lst = [line.strip() for line in f.readlines()]
            if self.trunck is not None:
                rxn_smi_tgt_lst = rxn_smi_tgt_lst[:self.trunck]

            for rxn_smi_tgt in tqdm(rxn_smi_tgt_lst):
                try:
                    rxn_inf = get_rxn_pfm_info((rxn_smi_tgt, self.task, self.ext_feat, self.ext_feat_type, self.ext_feat_param, self.mul_ext_readout))
                except:
                    print(f"[ERROR] {rxn_smi_tgt}, {self.task}, {self.ext_feat}, {self.ext_feat_type}, {self.ext_feat_param}, {self.mul_ext_readout}")
                    continue
                if rxn_inf is None:
                    print(f"[ERROR] {rxn_smi_tgt}, {self.task}, {self.ext_feat}, {self.ext_feat_type}, {self.ext_feat_param}, {self.mul_ext_readout}")
                    continue
                x_merge,edge_index_merge,edge_attr_merge,atom_mass_merge,x_oh_merge,edge_oh_attr_merge,\
                a_graphs_merge,b_graphs_merge,mol_index,tgt_,ext_feat_desc = rxn_inf
                data = Data(x=x_merge,edge_index=edge_index_merge,edge_attr=edge_attr_merge,
                            mol_index=mol_index,y=tgt_,ext_feat=torch.tensor(ext_feat_desc).float())
                data_list.append(data)
            data, slices = self.collate(data_list)
            print(f'[INFO] {len(data_list)} data index {idx} is saving...')
            torch.save((data, slices), self.processed_paths[idx])
    def len(self):
        ct = 0
        for slices in self.slices_lst:
            ct += len(slices['x']) - 1
        return ct
    def get(self,idx):
        for blk_i,data_num in enumerate(self.data_num_lst):
            if idx < data_num:
                break
        data_num_lst = [0] + self.data_num_lst
        idx -= data_num_lst[blk_i]
        self.data,self.slices = self.data_lst[blk_i],self.slices_lst[blk_i]
        data = separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False)
        return data
    
    def download(self):
        pass

class RXNDataset(InMemoryDataset):
    def __init__(self, root, name='rxn_dataset.csv', raw_data=[],transform=None, pre_transform=None,train=True,trunck=None,task="regression",num_worker=8,batch_size=128,multi_process=False,ext_feat=False,ext_feat_type="morgan",ext_feat_param={'radius':2,'nBits':2048,'useChirality':True},mul_ext_readout='mean',tag=""):
        '''
        Multi-process is useless in this procedure
        '''
        self.name = name
        self.train = train
        self.raw_data = raw_data
        self.trunck = trunck if trunck is not None and trunck != 0 else None
        self.task = task
        self.num_worker = num_worker
        self.batch_size = batch_size
        self.multi_process = multi_process
        self.ext_feat = ext_feat                                        # whether to extend the description, eg. Morgan fingerprint, RDKit descriptors, etc.
        self.ext_feat_type = ext_feat_type.lower()
        self.ext_feat_param = ext_feat_param
        self.mul_ext_readout = mul_ext_readout.lower()
        self.tag = tag
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self.name

    @property
    def processed_file_names(self):
        if not self.ext_feat:
            if not self.tag:
                return [f'{self.name[:-4]}_{self.trunck}.pt']
            else:
                return [f'{self.name[:-4]}_{self.trunck}_{self.tag}.pt']
        else:
            if self.ext_feat_type == 'rdkit':
                if not self.tag:
                    return [f'{self.name[:-4]}_{self.trunck}_{self.ext_feat_type}.pt']
                else:
                    return [f'{self.name[:-4]}_{self.trunck}_{self.ext_feat_type}_{self.tag}.pt']
            elif self.ext_feat_type in ['morgan','atompair','toptorsion','rdfp']:
                if not self.tag:
                    return [f'{self.name[:-4]}_{self.trunck}_{self.ext_feat_type}_{self.ext_feat_param["radius"]}_{self.ext_feat_param["nBits"]}.pt']
                else:
                    return [f'{self.name[:-4]}_{self.trunck}_{self.ext_feat_type}_{self.ext_feat_param["radius"]}_{self.ext_feat_param["nBits"]}_{self.tag}.pt']
            elif "++" in self.ext_feat_type:
                if not self.tag:
                    return [f'{self.name[:-4]}_{self.trunck}_{self.ext_feat_type}_{self.mul_ext_readout}.pt']
                else:
                    return [f'{self.name[:-4]}_{self.trunck}_{self.ext_feat_type}_{self.mul_ext_readout}_{self.tag}.pt']
            else:
                raise ValueError(f"ext_feat_type {self.ext_feat_type} is not supported")

    def process(self):
        self.data_list = []
        with open(f'{self.root}/{self.name}', 'r') as f:
            rxn_smi_tgt_lst = [line.strip() for line in f.readlines()]
        if self.trunck is not None:
            rxn_smi_tgt_lst = rxn_smi_tgt_lst[:self.trunck]
            
        if self.multi_process:
            print(f"[INFO] {self.num_worker} workers are used to process data...")
            # Accelerate the data processing using multiprocessing
            pool = Pool(self.num_worker)
            results = []
            for i in tqdm(range(0, len(rxn_smi_tgt_lst), self.batch_size)):
                batch = ((rxn_smi_tgt_lst[j], self.task, self.ext_feat, self.ext_feat_type, self.ext_feat_param, self.mul_ext_readout) for j in range(i, min(i + self.batch_size, len(rxn_smi_tgt_lst))))
                #tasks = ((rxn_smi_tgt_lst[i], self.task) for i in range(len(rxn_smi_tgt_lst)))
                for result in pool.imap(get_rxn_pfm_info, batch):
                    results.append(result)
            pool.close()
            pool.join()
            
            for rxn_inf in results:
                if rxn_inf is None:
                    continue
                x_merge,edge_index_merge,edge_attr_merge,atom_mass_merge,x_oh_merge,edge_oh_attr_merge,\
                a_graphs_merge,b_graphs_merge,mol_index,tgt_,ext_feat_desc = rxn_inf
                
                data = Data(x=x_merge,edge_index=edge_index_merge,edge_attr=edge_attr_merge,
                                mol_index=mol_index,atom_mass=atom_mass_merge,y=tgt_,ext_feat=torch.tensor(ext_feat_desc).float())
                self.data_list.append(data)
        else:
            for rxn_smi_tgt in tqdm(rxn_smi_tgt_lst):
                rxn_inf = get_rxn_pfm_info((rxn_smi_tgt, self.task, self.ext_feat, self.ext_feat_type, self.ext_feat_param, self.mul_ext_readout))
                if rxn_inf is None:
                    continue
                x_merge,edge_index_merge,edge_attr_merge,atom_mass_merge,x_oh_merge,edge_oh_attr_merge,\
                a_graphs_merge,b_graphs_merge,mol_index,tgt_,ext_feat_desc = rxn_inf
                
                data = Data(x=x_merge,edge_index=edge_index_merge,edge_attr=edge_attr_merge,
                            mol_index=mol_index,atom_mass=atom_mass_merge,y=tgt_,ext_feat=torch.tensor(ext_feat_desc).float())
                self.data_list.append(data)

        data, slices = self.collate(self.data_list)
        print(f'[INFO] {len(self.data_list)} Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def download(self):
        pass

class PairDataset(torch.utils.data.Dataset):
    def __init__(self,rct_dataset,pdt_dataset):
        self.rct_dataset = rct_dataset
        self.pdt_dataset = pdt_dataset
    def __getitem__(self,index):

        return self.rct_dataset[index],self.pdt_dataset[index]

    def __len__(self):
        return len(self.rct_dataset)
class TripleDataset(torch.utils.data.Dataset):
    def __init__(self,rct_dataset,pdt_dataset,mid_dataset):
        self.rct_dataset = rct_dataset
        self.pdt_dataset = pdt_dataset
        self.mid_dataset = mid_dataset

    def __getitem__(self,index):
        return self.rct_dataset[index],self.pdt_dataset[index],self.mid_dataset[index]
    
    def __len__(self):
        return len(self.rct_dataset)

def single_collate_fn(data_list):
    batch = Batch.from_data_list(data_list)
    return batch

def pair_collate_fn(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB

def triple_collate_fn(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    batchC = Batch.from_data_list([data[2] for data in data_list])
    return batchA, batchB, batchC

def get_idx_split(data_size, train_size, valid_size, seed):
    ids = shuffle(range(data_size), random_state=seed)
    if abs(train_size + valid_size - data_size) < 2:
        train_idx, val_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:])
        test_idx = val_idx
    else:    
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
    split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
    return split_dict

def load_vocab(vocab_file: str):
    vocab = {}
    with open(vocab_file, "r") as f:
        for i, line in enumerate(f):
            token = line.strip().split("\t")[0]
            token = token.split()[0]
            vocab[token] = i
    return vocab

def get_token_ids(tokens, vocab, max_len):
    
    token_ids = []
    token_ids.extend([vocab[token] for token in tokens])
    token_ids = token_ids[:max_len-1]
    token_ids.append(vocab["_EOS"])

    lens = len(token_ids)
    while len(token_ids) < max_len:
        token_ids.append(vocab["_PAD"])

    return token_ids, lens

def get_cpd_rxn_info(cpd_smi,ext,ext_param,ext_type,mul_ext_readout):

    rxn_mol = Chem.MolFromSmiles(cpd_smi)
    if ext:
        ext_feat = ext_feat_gen(rxn_mol,params=ext_param,desc_type=ext_type,multi_readout=mul_ext_readout)
    else:
        ext_feat = torch.tensor([0.0]).float()
    smi_blk_lst = cpd_smi.split('.')
    x_edge_index_attr_lst = []
    failed = False
    for smi in smi_blk_lst:
        rdkit_mol = Chem.MolFromSmiles(smi)
        if rdkit_mol == None:
            failed = True
            break
        x_edge_index_attr_lst.append(mol2graphinfo(rdkit_mol))
        
    if failed:
        return None
    
    atom_num_lst = [len(item[0]) for item in x_edge_index_attr_lst]
    atom_num_start = [0]
    mol_index = []
    for num in atom_num_lst[:-1]:
        atom_num_start.append(atom_num_start[-1]+num)
    for i,num in enumerate(atom_num_lst):
        mol_index += [i] * num
    if len(mol_index) == 0:
        # fix for empty molecule
        x_merge = torch.zeros([1,9],dtype=torch.int64)
        atom_mass_merge = torch.tensor([0],dtype=torch.float64)
        mol_index = [0]
    else:
        x_merge = torch.cat([item[0] for item in x_edge_index_attr_lst])
        atom_mass_merge = torch.cat([item[3] for item in x_edge_index_attr_lst])
    edge_index_merge = torch.cat([item[1]+num for item,num in zip(x_edge_index_attr_lst,atom_num_start)],dim=1)
    edge_attr_merge = torch.cat([item[2] for item in x_edge_index_attr_lst])
    
    x_oh_merge = torch.cat([item[4] for item in x_edge_index_attr_lst])
    edge_oh_attr_merge = torch.cat([item[5] for item in x_edge_index_attr_lst])
    a_graphs_merge = torch.cat([item[6] for item in x_edge_index_attr_lst])
    b_graphs_merge = torch.cat([item[7] for item in x_edge_index_attr_lst])

    return x_merge,edge_index_merge,edge_attr_merge,atom_mass_merge,x_oh_merge,edge_oh_attr_merge,a_graphs_merge,b_graphs_merge,mol_index,ext_feat

def get_rxn_pfm_info(rxn_smi_tgt_task_ens):
    rxn_smi_tgt,task,ext,ext_type,ext_param,mul_ext_readout = rxn_smi_tgt_task_ens
    task = task.lower()
    assert task in ["regression","classification"], "task must be regression or classification"
    rxn_smi,tgt_ = rxn_smi_tgt.split(',')
    if task.lower() == "regression":
        tgt_ = torch.tensor([float(tgt_)]).float()
    elif task.lower() == "classification":
        tgt_ = torch.tensor([int(tgt_)]).long()
    else:
        raise ValueError("task must be regression or classification")
    x_merge,edge_index_merge,edge_attr_merge,atom_mass_merge,x_oh_merge,edge_oh_attr_merge,a_graphs_merge,b_graphs_merge,mol_index,ext_feat = \
        get_cpd_rxn_info(rxn_smi,ext,ext_param,ext_type,mul_ext_readout)
    return x_merge,edge_index_merge,edge_attr_merge,atom_mass_merge,x_oh_merge,edge_oh_attr_merge,\
    a_graphs_merge,b_graphs_merge,mol_index,tgt_,ext_feat


def get_rxn_seq_info(input_):
    src_line,tgt_line,vocab,max_length = input_
    src_smi = "".join(src_line.strip().split())
    tgt_tokens = tgt_line.strip().split()
    tgt_token_ids,tgt_lens = get_token_ids(tgt_tokens,vocab,max_length)
    tgt_token_ids = torch.tensor([tgt_token_ids], dtype=torch.long)
    tgt_lens = torch.tensor([tgt_lens], dtype=torch.long)
    
    ## Molecular Graph for Reactant Molecules
    src_smi_blk_lst = src_smi.split('.')
    x_edge_index_attr_lst = []
    failed = False
    for smi in src_smi_blk_lst:
        rdkit_mol = Chem.MolFromSmiles(smi)
        if rdkit_mol == None:
            failed = True
            break
        x_edge_index_attr_lst.append(mol2graphinfo(rdkit_mol))
    if failed:
        return None
    atom_num_lst = [len(item[0]) for item in x_edge_index_attr_lst]
    atom_num_start = [0]
    mol_index = []
    for num in atom_num_lst[:-1]:
        atom_num_start.append(atom_num_start[-1]+num)
    for i,num in enumerate(atom_num_lst):
        mol_index += [i] * num
    
    x_merge = torch.cat([item[0] for item in x_edge_index_attr_lst])
    edge_index_merge = torch.cat([item[1]+num for item,num in zip(x_edge_index_attr_lst,atom_num_start)],dim=1)
    edge_attr_merge = torch.cat([item[2] for item in x_edge_index_attr_lst])
    atom_mass_merge = torch.cat([item[3] for item in x_edge_index_attr_lst])
    x_oh_merge = torch.cat([item[4] for item in x_edge_index_attr_lst])
    edge_oh_attr_merge = torch.cat([item[5] for item in x_edge_index_attr_lst])
    a_graphs_merge = torch.cat([item[6] for item in x_edge_index_attr_lst])
    b_graphs_merge = torch.cat([item[7] for item in x_edge_index_attr_lst])
    if len(mol_index) == 0:
        return None
    return x_merge,edge_index_merge,edge_attr_merge,mol_index,atom_mass_merge,x_oh_merge,edge_oh_attr_merge,\
           a_graphs_merge,b_graphs_merge,tgt_token_ids,tgt_lens

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    pattern =  "(\[[^\]]+]|Br?|Cl?|Se?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

def get_train_val_test_token_data(src, tgt, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    np.random.seed(random_seed)
    sf_idx = list(range(len(src)))
    np.random.shuffle(sf_idx)
    train_idx = sf_idx[:int(len(sf_idx)*(1-val_ratio-test_ratio))]
    val_idx = sf_idx[int(len(sf_idx)*(1-val_ratio-test_ratio)):int(len(sf_idx)*(1-test_ratio))]
    test_idx = sf_idx[int(len(sf_idx)*(1-test_ratio)):]
    src_token = [smi_tokenizer(smi) for smi in src]
    tgt_token = [smi_tokenizer(smi) for smi in tgt]

    train_src = [src_token[i] for i in train_idx]
    train_tgt = [tgt_token[i] for i in train_idx]
    val_src = [src_token[i] for i in val_idx]
    val_tgt = [tgt_token[i] for i in val_idx]
    test_src = [src_token[i] for i in test_idx]
    test_tgt = [tgt_token[i] for i in test_idx]

    return train_src, train_tgt, val_src, val_tgt, test_src, test_tgt

def gen_vocab_map(all_token,token_file):
    vocab_map = {}
    for line in tqdm(all_token):
        vocab_lst = line.split()
        for vocab in vocab_lst:
            if vocab not in vocab_map:
                vocab_map[vocab] = 1
            else:
                vocab_map[vocab] += 1
    vocab_inf_lst = ['_PAD','_UNK','_SOS','_EOS']
    for vocab in vocab_map:
        vocab_inf_lst.append(f'{vocab}    {vocab_map[vocab]}')
    with open(token_file,'w') as fw:
        fw.writelines('\n'.join(vocab_inf_lst))

def generate_regression_dataset(dataset_file,rct_cols=["Reactant1","Reactant2","Solvents","Reagents"],pdt_cols=["Product"],tgt_inf=None):
    
    raw_dataset = pd.read_csv(dataset_file)
    rct_inf_lst = raw_dataset[rct_cols].to_numpy()
    pdt_inf_lst = raw_dataset[pdt_cols].to_numpy()
    if tgt_inf is None:
        rct_parts = [f"{Chem.MolToSmiles(Chem.MolFromSmiles('.'.join(rct_inf)))},0.0" for rct_inf in tqdm(rct_inf_lst)]
        pdt_parts = [f"{Chem.MolToSmiles(Chem.MolFromSmiles('.'.join(pdt_inf)))},0.0" for pdt_inf in tqdm(pdt_inf_lst)]
    else:
        tgt_lst = raw_dataset[tgt_inf].to_list()
        rct_parts = [f"{Chem.MolToSmiles(Chem.MolFromSmiles('.'.join(rct_inf)))},{tgt}" for rct_inf,tgt in tqdm(zip(rct_inf_lst,tgt_lst))]
        pdt_parts = [f"{Chem.MolToSmiles(Chem.MolFromSmiles('.'.join(pdt_inf)))},{tgt}" for pdt_inf,tgt in tqdm(zip(pdt_inf_lst,tgt_lst))]
    return rct_parts,pdt_parts

def gen_mol_in_rxn(rxn_smiles,show_atom_map=True):
    smi_blk_lst = rxn_smiles.split('.')
    mol_lst = []
    failed = False
    for smi in smi_blk_lst:
        rdkit_mol = Chem.MolFromSmiles(smi)
        if show_atom_map:
            [atom.SetAtomMapNum(atom.GetIdx()+1) for atom in rdkit_mol.GetAtoms()]  ## atom map number have to start from 1
        if rdkit_mol == None:
            failed = True
            break
        mol_lst.append(rdkit_mol)
    return mol_lst,failed