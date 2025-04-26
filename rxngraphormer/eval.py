import torch,json,os,shutil
import pandas as pd
import numpy as np
from rdkit import Chem
from box import Box
from .model import RXNGRegressor,RXNG2Sequencer,RXNGraphormer
from .data import load_vocab,RXNG2SDataset,RXNDataset,get_idx_split,TripleDataset,PairDataset,triple_collate_fn,pair_collate_fn,smi_tokenizer
from .utils import canonical_smiles,update_dict_key,align_config
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score,mean_absolute_error
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SeqEval():
    def __init__(self,trained_model_path,topk=10,beam_size=10,temperature=1.0,n_best=10,min_length=1,max_length=512,batch_size=32,ckpt_file="valid_checkpoint.pt"):
        self.trained_model_path = trained_model_path
        self.topk = topk
        self.beam_size = beam_size
        self.temperature = temperature
        self.n_best = n_best
        self.min_length = min_length
        self.max_length = max_length

        print(f"[INFO] Loading trained model from {trained_model_path}")
        trained_para_json = f"{trained_model_path}/parameters.json"
        with open(trained_para_json,'r') as fr:
            pretrained_config_dict = json.load(fr)
        trained_config = Box(pretrained_config_dict)
        vocab = load_vocab(f'{trained_config.data.data_path}/{trained_config.data.vocab_file}')
        self.vocab_rev = [k for k, v in sorted(vocab.items(), key=lambda tup: tup[1])]
        #model = RXNG2Sequencer(trained_config,vocab)
        rxng = RXNGraphormer("sequence_generation",trained_config,vocab)
        model = rxng.get_model()

        ckpt_file = f"{trained_model_path}/model/{ckpt_file}"
        ckpt_inf = torch.load(ckpt_file,map_location=device)
        
        model.to(device)
        model.load_state_dict(update_dict_key(ckpt_inf['model_state_dict']))
        self.model = model
        print(f"[INFO] Loading test dataset from {trained_config.data.data_path}")

    
        test_dataset = RXNG2SDataset(root=trained_config.data.data_path,
                                    src_file=trained_config.data.test_src_file,
                                    tgt_file=trained_config.data.test_tgt_file,
                                    vocab_file=trained_config.data.vocab_file,
                                    trunck=0,multi_process=False,oh=False)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                            num_workers=4)
        self.test_ground_truth_smiles_lst = ["".join([self.vocab_rev[idx] for idx in test_dataset[idx].tgt_token_ids[0][:int(test_dataset[idx].tgt_lens[0])-1]]) for idx in range(len(test_dataset))]
    def eval(self,dataloader,ground_truth_smiles_lst):
        self.model.eval()
        all_predictions = []
        with torch.no_grad():
            step = 0
            for batch_data in dataloader:
                step += 1
                batch_data = batch_data.to(device)
                results = self.model.infer(reaction_batch=batch_data,
                                           batch_size=len(batch_data.tgt_lens),
                                            beam_size=self.beam_size,
                                            n_best=self.n_best,
                                            temperature=self.temperature,
                                            min_length=self.min_length,
                                            max_length=self.max_length)
                                
                for predictions in results["predictions"]:
                    smis = []
                    for prediction in predictions:
                        predicted_idx = prediction.detach().cpu().numpy()
                        predicted_tokens = [self.vocab_rev[idx] for idx in predicted_idx[:-1]]
                        smi = " ".join(predicted_tokens)
                        smis.append(smi)
                    smis = ",".join(smis)
                    all_predictions.append(smis)
                    
        accuracies = np.zeros([len(ground_truth_smiles_lst), self.n_best], dtype=np.float32)
        for i in range(len(ground_truth_smiles_lst)):
            smi_tgt = ground_truth_smiles_lst[i]
            line_predict = all_predictions[i]
            line_predict = "".join(line_predict.split())
            smis_predict = line_predict.split(",")
            smis_predict = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else "" for smi in smis_predict]
            for j, smi in enumerate(smis_predict):
                if smi == smi_tgt:
                    accuracies[i, j:] = 1.0
                    break
        self.accuracies = accuracies
        for i in range(self.topk):
            print(f"Top-{i+1} Accuracy: {np.mean(self.accuracies[:, i])}")
        return accuracies
    
def eval_regression_performance(pretrained_model_path,ckpt_file="valid_checkpoint.pt",scale=1.0,specific_val=False,yield_constrain=False,return_train_results=False):
    pretrained_para_json = f"{pretrained_model_path}/parameters.json"
    with open(pretrained_para_json,'r') as fr:
        pretrained_config_dict = json.load(fr)
    pretrained_config = Box(pretrained_config_dict)

    ckpt_file = f"{pretrained_model_path}/model/{ckpt_file}"
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
                    "output_size":1,
                    "output_norm":eval(pretrained_config.model.output_norm),
                    "split_process":True,
                    "split_merge_method":pretrained_config.model.split_merge_method,
                    "output_act_func":pretrained_config.model.output_act_func,
                    "rct_batch_norm":eval(pretrained_config.model.rct_batch_norm),
                    "pdt_batch_norm":eval(pretrained_config.model.pdt_batch_norm),
                    "use_mid_inf":eval(pretrained_config.model.use_mid_inf),
                    "mid_iteract_method":pretrained_config.model.mid_iteract_method,
                    "mid_batch_norm":eval(pretrained_config.model.mid_batch_norm),
                    "mid_layer_num":pretrained_config.model.mid_layer_num}
    
    rxng = RXNGraphormer("regression",align_config(input_param,"regressor"),"")
    model = rxng.get_model()

    model.load_state_dict(update_dict_key(ckpt_inf["model_state_dict"]))
    model.to(device)
    model.eval()
    if not specific_val:
        rct_dataset = RXNDataset(root=pretrained_config.data.data_path,name=pretrained_config.data.rct_data_file,trunck=pretrained_config.data.data_trunck)
        pdt_dataset = RXNDataset(root=pretrained_config.data.data_path,name=pretrained_config.data.pdt_data_file,trunck=pretrained_config.data.data_trunck)
        mid_dataset = RXNDataset(root=pretrained_config.data.data_path,name=pretrained_config.data.mid_data_file,trunck=pretrained_config.data.data_trunck)

        split_ids_map = get_idx_split(len(rct_dataset),int(pretrained_config.data.train_ratio*len(rct_dataset)), 
                                    int(pretrained_config.data.valid_ratio*len(rct_dataset)),pretrained_config.data.seed)
        test_rct_dataset = rct_dataset[split_ids_map['test']]
        test_pdt_dataset = pdt_dataset[split_ids_map['test']]
        test_mid_dataset = mid_dataset[split_ids_map['test']]
        test_dataset = TripleDataset(test_rct_dataset,test_pdt_dataset,test_mid_dataset)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=pretrained_config.data.batch_size, shuffle=False,collate_fn=triple_collate_fn)
        if return_train_results:
            train_rct_dataset = rct_dataset[split_ids_map['train']]
            train_pdt_dataset = pdt_dataset[split_ids_map['train']]
            train_mid_dataset = mid_dataset[split_ids_map['train']]
            train_dataset = TripleDataset(train_rct_dataset,train_pdt_dataset,train_mid_dataset)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=pretrained_config.data.batch_size, shuffle=False,collate_fn=triple_collate_fn)
    else:
        test_rct_dataset = RXNDataset(root=pretrained_config.data.data_path,name=pretrained_config.data.test_rct_data_file,trunck=pretrained_config.data.data_trunck)
        test_pdt_dataset = RXNDataset(root=pretrained_config.data.data_path,name=pretrained_config.data.test_pdt_data_file,trunck=pretrained_config.data.data_trunck)
        test_mid_dataset = RXNDataset(root=pretrained_config.data.data_path,name=pretrained_config.data.test_mid_data_file,trunck=pretrained_config.data.data_trunck)
        test_dataset = TripleDataset(test_rct_dataset,test_pdt_dataset,test_mid_dataset)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=pretrained_config.data.batch_size, shuffle=False,collate_fn=triple_collate_fn)
        if return_train_results:
            train_rct_dataset = RXNDataset(root=pretrained_config.data.data_path,name=pretrained_config.data.train_rct_data_file,trunck=pretrained_config.data.data_trunck)
            train_pdt_dataset = RXNDataset(root=pretrained_config.data.data_path,name=pretrained_config.data.train_pdt_data_file,trunck=pretrained_config.data.data_trunck)
            train_mid_dataset = RXNDataset(root=pretrained_config.data.data_path,name=pretrained_config.data.train_mid_data_file,trunck=pretrained_config.data.data_trunck)
            train_dataset = TripleDataset(train_rct_dataset,train_pdt_dataset,train_mid_dataset)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=pretrained_config.data.batch_size, shuffle=False,collate_fn=triple_collate_fn)
    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)
    with torch.no_grad():
        for step, batch_data in enumerate(test_dataloader):
            if not eval(pretrained_config.model.use_mid_inf):
                rct_data,pdt_data = batch_data
                rct_data = rct_data.to(device)
                pdt_data = pdt_data.to(device)
                out = model([rct_data,pdt_data])
            else:
                rct_data,pdt_data,mid_data = batch_data
                rct_data = rct_data.to(device)
                pdt_data = pdt_data.to(device)
                mid_data = mid_data.to(device)
                out = model([rct_data,pdt_data,mid_data])
            preds = torch.cat([preds, out.detach_()], dim=0)
            targets = torch.cat([targets, rct_data.y.unsqueeze(1)], dim=0)
    targets = targets.cpu()
    preds = preds.cpu()
    if yield_constrain:
        preds = torch.where(preds < 0, torch.tensor(0.0), preds)
        preds = torch.where(preds > 1, torch.tensor(1.0), preds)
        
    preds = preds * scale
    targets = targets * scale
    r2 = r2_score(targets,preds)
    mae = mean_absolute_error(targets,preds)

    if return_train_results:
        train_preds = torch.Tensor([]).to(device)
        train_targets = torch.Tensor([]).to(device)
        with torch.no_grad():
            for step, batch_data in enumerate(train_dataloader):
                if not eval(pretrained_config.model.use_mid_inf):
                    rct_data,pdt_data = batch_data
                    rct_data = rct_data.to(device)
                    pdt_data = pdt_data.to(device)
                    out = model([rct_data,pdt_data])
                else:
                    rct_data,pdt_data,mid_data = batch_data
                    rct_data = rct_data.to(device)
                    pdt_data = pdt_data.to(device)
                    mid_data = mid_data.to(device)
                    out = model([rct_data,pdt_data,mid_data])
                train_preds = torch.cat([train_preds, out.detach_()], dim=0)
                train_targets = torch.cat([train_targets, rct_data.y.unsqueeze(1)], dim=0)
        train_targets = train_targets.cpu()
        train_preds = train_preds.cpu()
        if yield_constrain:
            train_preds = torch.where(train_preds < 0, torch.tensor(0.0), train_preds)
            train_preds = torch.where(train_preds > 1, torch.tensor(1.0), train_preds)

        train_preds = train_preds * scale
        train_targets = train_targets * scale
        train_r2 = r2_score(train_targets,train_preds)
        train_mae = mean_absolute_error(train_targets,train_preds)
        return train_r2,train_mae,train_preds,train_targets,r2,mae,preds,targets

    return r2,mae,preds,targets

def load_pred_model(pretrained_model_path,ckpt_filename="valid_checkpoint.pt",task_type="reactivity"):
    assert task_type in ["reactivity","selectivity","retro-synthesis","forward-synthesis"]
    task_type = task_type.lower()
    pretrained_para_json = f"{pretrained_model_path}/parameters.json"
    with open(pretrained_para_json,'r') as fr:
        pretrained_config_dict = json.load(fr)
    pretrained_config = Box(pretrained_config_dict)

    ckpt_file = f"{pretrained_model_path}/model/{ckpt_filename}"
    ckpt_inf = torch.load(ckpt_file,map_location=device)
    if task_type in ["reactivity","selectivity"]:
        
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
                        "use_mid_inf":eval(pretrained_config.model.use_mid_inf),
                        "mid_iteract_method":pretrained_config.model.mid_iteract_method,
                        "mid_batch_norm":eval(pretrained_config.model.mid_batch_norm),
                        "mid_layer_num":pretrained_config.model.mid_layer_num}
        
        rxng = RXNGraphormer("regression",align_config(input_param,"regressor"),"")
        model = rxng.get_model()



    elif task_type in ["retro-synthesis","forward-synthesis"]:
        vocab = load_vocab(f'{pretrained_config.data.data_path}/{pretrained_config.data.vocab_file}')
        # vocab_rev = [k for k, v in sorted(vocab.items(), key=lambda tup: tup[1])]
        #model = RXNG2Sequencer(pretrained_config,vocab)
        rxng = RXNGraphormer("sequence_generation",pretrained_config,vocab)
        model = rxng.get_model()
    model.load_state_dict(update_dict_key(ckpt_inf["model_state_dict"]))
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return model

def get_eval_dataloader(root,data_file_dict,pretrained_model_path,batch_size=4,task_type="reactivity"):
    assert task_type in ["reactivity","selectivity","retro-synthesis","forward-synthesis"]
    task_type = task_type.lower()
    pretrained_para_json = f"{pretrained_model_path}/parameters.json"
    with open(pretrained_para_json,'r') as fr:
        pretrained_config_dict = json.load(fr)
    pretrained_config = Box(pretrained_config_dict)

    if task_type == "reactivity" or task_type == "selectivity":
        rct_dataset = RXNDataset(root=root,name=data_file_dict["rct"],trunck=pretrained_config.data.data_trunck)
        pdt_dataset = RXNDataset(root=root,name=data_file_dict["pdt"],trunck=pretrained_config.data.data_trunck)
        mid_dataset = RXNDataset(root=root,name=data_file_dict["delta-mol"],trunck=pretrained_config.data.data_trunck)
        eval_dataset = TripleDataset(rct_dataset,pdt_dataset,mid_dataset)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False,collate_fn=triple_collate_fn)
        return eval_dataloader
    elif task_type == "forward-synthesis" or task_type == "retro-synthesis":
        eval_dataset = RXNG2SDataset(root=root,
                                src_file=data_file_dict["src"],
                                tgt_file=data_file_dict["tgt"],
                                vocab_file=pretrained_config.data.vocab_file,
                                trunck=0,multi_process=False,oh=False)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        return eval_dataloader
    
def reaction_prediction(model_path, rxn_smiles_lst, task_type, params={"batch_size":4,"beam_size":10,"n_best":10,
                                                                       "temperature":1.0,"min_length":1,"max_length":512}):
    from rxngraphormer.midgen.midmol import gen_mech_mid_smi
    task_type = task_type.lower()
    assert len(rxn_smiles_lst) >= 2, "'rxn_smiles_lst' must contain at least 2 reactions"
    assert task_type in ["reactivity","selectivity","forward-synthesis","retro-synthesis"], "task_type must be 'reactivity', 'selectivity', 'forward-synthesis' or 'retro-synthesis'"
    model = load_pred_model(model_path, task_type=task_type)
    if task_type in ["reactivity","selectivity"]:
        rct_smi_lst = [f'{canonical_smiles(smi.split(">>")[0])},0' for smi in rxn_smiles_lst]
        pdt_smi_lst = [f'{canonical_smiles(smi.split(">>")[1])},0' for smi in rxn_smiles_lst]
        rct_pdt_pair_lst = [rxn_smiles.split(">>") for rxn_smiles in rxn_smiles_lst]
        delta_mol_smi_lst = [f"{gen_mech_mid_smi(rct_pdt_pair)[0]},0" for rct_pdt_pair in rct_pdt_pair_lst]
        os.makedirs("./rxn_emb_tmp/",exist_ok=True)
        with open("./rxn_emb_tmp/rct_smiles_0.csv","w") as fw:
            fw.writelines("\n".join(rct_smi_lst))
        with open("./rxn_emb_tmp/pdt_smiles_0.csv","w") as fw:
            fw.writelines("\n".join(pdt_smi_lst))
        with open("./rxn_emb_tmp/delta_mol_smiles_0.csv","w") as fw:
            fw.writelines("\n".join(delta_mol_smi_lst))

        eval_dataloader = get_eval_dataloader(root="./rxn_emb_tmp",
                                        data_file_dict={"rct":"rct_smiles_0.csv",
                                                        "pdt":"pdt_smiles_0.csv",
                                                        "delta-mol":"delta_mol_smiles_0.csv"},
                                        pretrained_model_path=model_path,
                                        task_type=task_type,
                                        batch_size=params["batch_size"])
        preds = torch.Tensor([]).to(device)
        with torch.no_grad():
            for step, batch_data in enumerate(eval_dataloader):
                rct_data,pdt_data,mid_data = batch_data
                rct_data = rct_data.to(device)
                pdt_data = pdt_data.to(device)
                mid_data = mid_data.to(device)
                out = model([rct_data,pdt_data,mid_data])
                preds = torch.cat([preds, out.detach_()], dim=0)
        preds = preds.cpu()
        if task_type == "reactivity":
            preds = torch.where(preds < 0, torch.tensor(0.0), preds)
            preds = torch.where(preds > 1, torch.tensor(1.0), preds)
            preds = preds.numpy() * 100
        shutil.rmtree("./rxn_emb_tmp")
    
    elif task_type in ["forward-synthesis","retro-synthesis"]:
        vocab_rev = [k for k, v in sorted(model.vocab.items(), key=lambda tup: tup[1])]
        tokenized_smiles_lst = [smi_tokenizer(smi) for smi in rxn_smiles_lst]
        os.makedirs("./rxn_emb_tmp/",exist_ok=True)
        with open("./rxn_emb_tmp/src_tokenized_smiles.txt","w") as fw:
            fw.writelines("\n".join(tokenized_smiles_lst))
        with open("./rxn_emb_tmp/tgt_tokenized_smiles.txt","w") as fw: # just a placeholder
            fw.writelines("\n".join(tokenized_smiles_lst))
        pretrained_para_json = f"{model_path}/parameters.json"
        with open(pretrained_para_json,'r') as fr:
            pretrained_config_dict = json.load(fr)
        pretrained_config = Box(pretrained_config_dict)
        shutil.copyfile(f'{pretrained_config.data.data_path}/{pretrained_config.data.vocab_file}',f"./rxn_emb_tmp/{pretrained_config.data.vocab_file}")
        eval_dataloader = get_eval_dataloader("./rxn_emb_tmp",{"src":"src_tokenized_smiles.txt","tgt":"tgt_tokenized_smiles.txt"},pretrained_model_path=model_path,batch_size=params["batch_size"],task_type=task_type)

        all_predictions = []
        with torch.no_grad():
            step = 0
            for batch_data in eval_dataloader:
                step += 1
                batch_data = batch_data.to(device)
                results = model.infer(reaction_batch=batch_data,
                                            batch_size=len(batch_data.tgt_lens),
                                            beam_size=params["beam_size"],
                                            n_best=params["n_best"],
                                            temperature=params["temperature"],
                                            min_length=params["min_length"],
                                            max_length=params["max_length"])
                                
                for predictions in results["predictions"]:
                    smis = []
                    for prediction in predictions:
                        predicted_idx = prediction.detach().cpu().numpy()
                        predicted_tokens = [vocab_rev[idx] for idx in predicted_idx[:-1]]
                        smi = "".join(predicted_tokens)
                        smis.append(smi)
                    #smis = ",".join(smis)
                    all_predictions.append(smis)
        preds = pd.DataFrame(all_predictions).T
        preds.index = [f"Top-{i+1}" for i in range(len(smis))]
        shutil.rmtree("./rxn_emb_tmp")
    print("Done!")
    return preds