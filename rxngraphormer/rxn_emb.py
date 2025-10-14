import json,torch,os,shutil
from box import Box
from tqdm import tqdm
from .model import RXNGClassifier,RXNGRegressor,RXNGraphormer
from .utils import update_dict_key,canonical_smiles,align_config
from .data import MultiRXNDataset,PairDataset,pair_collate_fn,single_collate_fn
from torch.nn.init import xavier_uniform_
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RXNEMB():
    def __init__(self,pretrained_model_path,random_init=False,model_type="classifier"):
        pretrained_para_json = f"{pretrained_model_path}/parameters.json"
        with open(pretrained_para_json,'r') as fr:
            pretrained_config_dict = json.load(fr)
        pretrained_config = Box(pretrained_config_dict)
        ckpt_file = f"{pretrained_model_path}/model/valid_checkpoint.pt"
        ckpt_inf = torch.load(ckpt_file,map_location=device)
        if model_type == "classifier":
            input_param = {"emb_dim":pretrained_config.model.emb_dim,
                            "gnn_type":pretrained_config.model.gnn_type,
                            "gnn_aggr":pretrained_config.model.gnn_aggr,
                            "gnum_layer":pretrained_config.model.gnn_num_layer,
                            "node_readout":pretrained_config.model.node_readout,
                            "num_heads":pretrained_config.model.num_heads,
                            "JK":pretrained_config.model.gnn_jk,
                            "graph_pooling":pretrained_config.model.graph_pooling,
                            "tnum_layer":pretrained_config.model.trans_num_layer,
                            "trans_readout":pretrained_config.model.trans_readout,
                            "onum_layer":pretrained_config.model.output_num_layer,
                            "drop_ratio":pretrained_config.model.drop_ratio,
                            "output_size":2,
                            "split_process":True,
                            "split_merge_method":pretrained_config.model.split_merge_method,
                            "output_act_func":pretrained_config.model.output_act_func}
            rxng = RXNGraphormer("classification",align_config(input_param,"classifier"),"")
            model = rxng.get_model()


        elif model_type == "regressor":
            
            input_param = {"emb_dim":pretrained_config.model.emb_dim,
                            "gnn_type":pretrained_config.model.gnn_type,
                            "gnn_aggr":pretrained_config.model.gnn_aggr,
                            "gnum_layer":pretrained_config.model.gnn_num_layer,
                            "node_readout":pretrained_config.model.node_readout,
                            "num_heads":pretrained_config.model.num_heads,
                            "JK":pretrained_config.model.gnn_jk,
                            "graph_pooling":pretrained_config.model.graph_pooling,
                            "tnum_layer":pretrained_config.model.trans_num_layer,
                            "trans_readout":pretrained_config.model.trans_readout,
                            "onum_layer":pretrained_config.model.output_num_layer,
                            "drop_ratio":pretrained_config.model.drop_ratio,
                            "output_size":1,
                            "output_norm":eval(pretrained_config.model.output_norm),
                            "split_process":True,
                            "split_merge_method":pretrained_config.model.split_merge_method,
                            "output_act_func":pretrained_config.model.output_act_func,
                            "rct_batch_norm":eval(pretrained_config.model.rct_batch_norm),
                            "pdt_batch_norm":eval(pretrained_config.model.pdt_batch_norm),
                            "use_mid_inf":pretrained_config.model.use_mid_inf,
                            "mid_iteract_method":pretrained_config.model.mid_iteract_method,
                            "mid_batch_norm":eval(pretrained_config.model.mid_batch_norm),
                            "mid_layer_num":pretrained_config.model.mid_layer_num}
            
            rxng = RXNGraphormer("regression",align_config(input_param,"regressor"),"")
            model = rxng.get_model()

        if not random_init:
            model.load_state_dict(update_dict_key(ckpt_inf["model_state_dict"]))
        else:
            print("[INFO] Randomly initialize model parameters")
            for p in model.parameters():
                if p.dim() > 1 and p.requires_grad:
                    xavier_uniform_(p)
            
        model.to(device)
        model.eval()
        self.model = model
    def gen_rxn_emb_from_dataset(self,root,
                                 rct_name_regrex="50k_rxn_type_rct_0.csv",pdt_name_regrex="50k_rxn_type_pdt_0.csv",batch_size=128):
        rct_dataset = MultiRXNDataset(root=root, name_regrex=rct_name_regrex)
        pdt_dataset = MultiRXNDataset(root=root, name_regrex=pdt_name_regrex)
        pair_dataset = PairDataset(rct_dataset,pdt_dataset)
        pair_dataloader = torch.utils.data.DataLoader(pair_dataset, batch_size=batch_size, shuffle=False,collate_fn=pair_collate_fn)
        all_rxn_emb = []
        print("[INFO] Generating reaction embedding...")
        with torch.no_grad():
            for data in tqdm(pair_dataloader):
                rct_data,pdt_data = data
                rct_data.to(device)
                pdt_data.to(device)
                rct_padded_memory_bank,rct_batch,rct_memory_lengths = self.model.rct_encoder(rct_data)
                pdt_padded_memory_bank,pdt_batch,pdt_memory_lengths = self.model.pdt_encoder(pdt_data)
                rct_rxn_transf_emb = rct_padded_memory_bank.transpose(0,1)
                pdt_rxn_transf_emb = pdt_padded_memory_bank.transpose(0,1)
                if self.model.trans_readout == 'mean':
                    rct_rxn_transf_emb_merg = rct_rxn_transf_emb.mean(dim=1)
                    pdt_rxn_transf_emb_merg = pdt_rxn_transf_emb.mean(dim=1)
                diff_emb = torch.abs(rct_rxn_transf_emb_merg - pdt_rxn_transf_emb_merg)
                if self.model.split_merge_method == "all":
                    rxn_emb = torch.cat([rct_rxn_transf_emb_merg,pdt_rxn_transf_emb_merg,diff_emb],dim=-1)
                elif self.model.split_merge_method == "only_diff":
                    rxn_emb = diff_emb
                elif self.model.split_merge_method == "rct_pdt":
                    rxn_emb = torch.cat([rct_rxn_transf_emb_merg,pdt_rxn_transf_emb_merg],dim=-1)
                for lin_layer,norm_layer in zip(self.model.decoder.layers[:-1],self.model.decoder.batch_norms[:-1]):
                    rxn_emb = lin_layer(rxn_emb)
                    rxn_emb = norm_layer(rxn_emb)
                all_rxn_emb.append(rxn_emb.detach().cpu())
        return torch.cat(all_rxn_emb,dim=0)
    def gen_half_rxn_mol_emb_from_dataset(self,root,
                                                name_regrex="50k_rxn_type_rct_0.csv",batch_size=128,mol_type="rct"):
        assert mol_type in ["rct","pdt"], "mol_type must be either 'rct' or 'pdt'"
        dataset = MultiRXNDataset(root=root, name_regrex=name_regrex)
        pair_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,collate_fn=single_collate_fn)
        all_rxn_transf_emb = []
        print("[INFO] Generating reaction embedding...")
        with torch.no_grad():
            for data in tqdm(pair_dataloader):
                data.to(device)
                if mol_type == "rct":
                    padded_memory_bank,batch,memory_lengths = self.model.rct_encoder(data)
                elif mol_type == "pdt":
                    padded_memory_bank,batch,memory_lengths = self.model.pdt_encoder(data)

                rxn_transf_emb = padded_memory_bank.transpose(0,1)
                all_rxn_transf_emb.append(rxn_transf_emb.detach().cpu())
                ## TODO 把padding的部分去掉
        return all_rxn_transf_emb
        
    def gen_mol_emb_from_dataset(self,root,
                                 name_regrex="50k_rxn_type_rct_0.csv",batch_size=128,mol_type="rct"):
        assert mol_type in ["rct","pdt"], "mol_type must be either 'rct' or 'pdt'"
        dataset = MultiRXNDataset(root=root, name_regrex=name_regrex)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=single_collate_fn)
        all_mol_emb = []
        print("[INFO] Generating molecular embedding...")
        with torch.no_grad():
            for data in tqdm(dataloader):
                data.to(device)
                x = data.x
                mol_index = data.mol_index
                edge_index = data.edge_index
                edge_attr = data.edge_attr
                if mol_type == "rct":
                    node_representation, mol_index,batch = self.model.rct_encoder.rxn_graph_encoder(x, mol_index, edge_index, edge_attr)
                elif mol_type == "pdt":
                    node_representation, mol_index,batch = self.model.pdt_encoder.rxn_graph_encoder(x, mol_index, edge_index, edge_attr)
                    
                # 获取唯一的分子标识符
                uniq_mol = mol_index.unique()

                # 使用列表解析选择对应于每个分子的行
                mol_representation = [node_representation[mol_index == mol].cpu() for mol in uniq_mol]
                all_mol_emb += mol_representation
        return all_mol_emb
    
    def gen_rxn_emb(self,rxn_smiles_lst,batch_size=128):
        assert len(rxn_smiles_lst) >= 2, "rxn_smiles_lst must contain at least 2 reactions"
        rct_smi_lst = [f'{canonical_smiles(smi.split(">>")[0])},0' for smi in rxn_smiles_lst]
        pdt_smi_lst = [f'{canonical_smiles(smi.split(">>")[1])},0' for smi in rxn_smiles_lst]
        os.makedirs("./rxn_emb_tmp/",exist_ok=True)
        with open("./rxn_emb_tmp/rct_smiles_0.csv","w") as fw:
            fw.writelines("\n".join(rct_smi_lst))
        with open("./rxn_emb_tmp/pdt_smiles_0.csv","w") as fw:
            fw.writelines("\n".join(pdt_smi_lst))
        rxn_emb = self.gen_rxn_emb_from_dataset(root="./rxn_emb_tmp",rct_name_regrex="rct_smiles_0.csv",pdt_name_regrex="pdt_smiles_0.csv",batch_size=batch_size)
        shutil.rmtree("./rxn_emb_tmp")
        return rxn_emb
    
    def gen_mult_mol_emb(self,mult_mol_smiles_lst,mol_type="rct",batch_size=128):
        
        assert mol_type in ["rct","pdt"], "mol_type must be either 'rct' or 'pdt'"
        assert len(mult_mol_smiles_lst) >= 2, "mult_mol_smiles_lst must contain at least 2 molecules"
        mol_num_lst = []
        tot_mol_smi_lst = []
        for idx,mult_mol_smi in enumerate(mult_mol_smiles_lst):
            tot_mol_smi_lst += mult_mol_smi.split(".")
            mol_num_lst += len(mult_mol_smi.split(".")) * [idx]
        tot_mol_emb = self.gen_mol_emb(tot_mol_smi_lst,mol_type=mol_type,batch_size=batch_size)
        return tot_mol_emb,mol_num_lst
    
    def gen_mol_emb(self,mol_smiles_lst,mol_type="rct",batch_size=128):
        assert mol_type in ["rct","pdt"], "mol_type must be either 'rct' or 'pdt'"
        assert len(mol_smiles_lst) >= 2, "mol_smiles_lst must contain at least 2 molecule"
        mol_smi_lst = [f'{canonical_smiles(smi)},0' for smi in mol_smiles_lst]
        os.makedirs("./mol_emb_tmp/",exist_ok=True)
        with open(f"./mol_emb_tmp/{mol_type}_smiles_0.csv","w") as fw:
            fw.writelines("\n".join(mol_smi_lst))
        mol_emb = self.gen_mol_emb_from_dataset(root="./mol_emb_tmp",name_regrex=f"{mol_type}_smiles_0.csv",mol_type=mol_type,batch_size=batch_size)
        shutil.rmtree("./mol_emb_tmp")
        return mol_emb
    
    def gen_half_rxn_mol_emb(self,half_rxn_smiles_lst,mol_type="rct",batch_size=128):
        assert mol_type in ["rct","pdt"], "mol_type must be either 'rct' or 'pdt'"
        assert len(half_rxn_smiles_lst) >= 2, "half_rxn_smiles_lst must contain at least 2 half reactions"
        half_rxn_smi_lst = [f'{canonical_smiles(smi)},0' for smi in half_rxn_smiles_lst]
        os.makedirs("./half_rxn_emb_tmp/",exist_ok=True)
        with open(f"./half_rxn_emb_tmp/{mol_type}_smiles_0.csv","w") as fw:
            fw.writelines("\n".join(half_rxn_smi_lst))
        half_rxn_emb = self.gen_half_rxn_mol_emb_from_dataset(root="./half_rxn_emb_tmp",name_regrex=f"{mol_type}_smiles_0.csv",mol_type=mol_type,batch_size=batch_size)
        shutil.rmtree("./half_rxn_emb_tmp")
        return half_rxn_emb

class RXNClassifier():
    def __init__(self,pretrained_model_path,random_init=False):
        pretrained_para_json = f"{pretrained_model_path}/parameters.json"
        with open(pretrained_para_json,'r') as fr:
            pretrained_config_dict = json.load(fr)
        pretrained_config = Box(pretrained_config_dict)
        ckpt_file = f"{pretrained_model_path}/model/valid_checkpoint.pt"
        ckpt_inf = torch.load(ckpt_file,map_location=device)

        input_param = {"emb_dim":pretrained_config.model.emb_dim,
                        "gnn_type":pretrained_config.model.gnn_type,
                        "gnn_aggr":pretrained_config.model.gnn_aggr,
                        "gnum_layer":pretrained_config.model.gnn_num_layer,
                        "node_readout":pretrained_config.model.node_readout,
                        "num_heads":pretrained_config.model.num_heads,
                        "JK":pretrained_config.model.gnn_jk,
                        "graph_pooling":pretrained_config.model.graph_pooling,
                        "tnum_layer":pretrained_config.model.trans_num_layer,
                        "trans_readout":pretrained_config.model.trans_readout,
                        "onum_layer":pretrained_config.model.output_num_layer,
                        "drop_ratio":pretrained_config.model.drop_ratio,
                        "output_size":2,
                        "split_process":True,
                        "split_merge_method":pretrained_config.model.split_merge_method,
                        "output_act_func":pretrained_config.model.output_act_func}
        rxng = RXNGraphormer("classification",align_config(input_param,"classifier"),"")
        model = rxng.get_model()
        
        if not random_init:
            model.load_state_dict(update_dict_key(ckpt_inf["model_state_dict"]))
        else:
            print("[INFO] Randomly initialize model parameters")
            for p in model.parameters():
                if p.dim() > 1 and p.requires_grad:
                    xavier_uniform_(p)
            
        model.to(device)
        # model.eval()
        self.model = model
    
    def rxn_pred(self,rxn_smiles_lst,batch_size=128):
        assert len(rxn_smiles_lst) >= 2, "rxn_smiles_lst must contain at least 2 reactions"
        rct_smi_lst = [f'{canonical_smiles(smi.split(">>")[0])},0' for smi in rxn_smiles_lst]
        pdt_smi_lst = [f'{canonical_smiles(smi.split(">>")[1])},0' for smi in rxn_smiles_lst]
        os.makedirs("./rxn_emb_tmp/",exist_ok=True)
        with open("./rxn_emb_tmp/rct_smiles_0.csv","w") as fw:
            fw.writelines("\n".join(rct_smi_lst))
        with open("./rxn_emb_tmp/pdt_smiles_0.csv","w") as fw:
            fw.writelines("\n".join(pdt_smi_lst))
        rxn_preds,rxn_confidences = self.rxn_pred_from_dataset(root="./rxn_emb_tmp",rct_name_regrex="rct_smiles_0.csv",pdt_name_regrex="pdt_smiles_0.csv",batch_size=batch_size)
        shutil.rmtree("./rxn_emb_tmp")
        return rxn_preds,rxn_confidences
    
    def rxn_pred_from_dataset(self,root,
                                 rct_name_regrex,
                                 pdt_name_regrex,
                                 batch_size=128):
        rct_dataset = MultiRXNDataset(root=root, name_regrex=rct_name_regrex)
        pdt_dataset = MultiRXNDataset(root=root, name_regrex=pdt_name_regrex)
        pair_dataset = PairDataset(rct_dataset,pdt_dataset)
        pair_dataloader = torch.utils.data.DataLoader(pair_dataset, batch_size=batch_size, shuffle=False,collate_fn=pair_collate_fn)
        all_rxn_pred = []
        all_rxn_confidence = []
        print("[INFO] Predict whether the reaction is real...")
        with torch.no_grad():
            for data in tqdm(pair_dataloader):
                rct_data,pdt_data = data
                rct_data.to(device)
                pdt_data.to(device)
                out = self.model([rct_data,pdt_data])
                preds = out.argmax(1)
                confidence = out.max(1).values
                all_rxn_pred.append(preds.detach().cpu())
                all_rxn_confidence.append(confidence.detach().cpu())
        return torch.cat(all_rxn_pred,dim=0),torch.cat(all_rxn_confidence,dim=0)