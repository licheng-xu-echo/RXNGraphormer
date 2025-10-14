import torch,datetime,os,logging,json,time,sys
import numpy as np
from box import Box
from .model import RXNGRegressor,RXNGClassifier,RXNG2Sequencer,RXNGraphormer
from .utils import setup_logger,grad_norm,param_norm,get_lr,update_dict_key,add_dense_empty_node_edge,align_config
from torch.optim import Adam,AdamW
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import nn
from torch.nn.init import xavier_uniform_
from .data import RXNDataset,get_idx_split,RXNG2SDataset,load_vocab,MultiRXNDataset,PairDataset,pair_collate_fn,TripleDataset,triple_collate_fn
from .scheduler import get_linear_scheduler_with_warmup,NoamLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from rdkit import Chem


class SPLITClassifierTrainer():
    def __init__(self,config):
        self.config = config
        self.multi_gpu = self.config.others.multi_gpu
        
        input_param = {"emb_dim":self.config.model.emb_dim,
                        "gnn_type":self.config.model.gnn_type,
                        "gnn_aggr":self.config.model.gnn_aggr,
                        "gnum_layer":self.config.model.gnn_num_layer,
                        "node_readout":self.config.model.node_readout,
                        "num_heads":self.config.model.num_heads,
                        "JK":self.config.model.gnn_jk,
                        "graph_pooling":self.config.model.graph_pooling,
                        "tnum_layer":self.config.model.trans_num_layer,
                        "trans_readout":self.config.model.trans_readout,
                        "onum_layer":self.config.model.output_num_layer,
                        "drop_ratio":self.config.model.drop_ratio,
                        "output_size":2,
                        "split_process":True,
                        "split_merge_method":self.config.model.split_merge_method,
                        "output_act_func":self.config.model.output_act_func}
        
        rxng = RXNGraphormer("classification",align_config(input_param,"classifier"),"")
        self.model = rxng.get_model()

        if self.multi_gpu:
            self.local_rank = self.config.others.local_rank
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend='nccl')
            self.model.to(self.local_rank)
            if self.config.model.graph_pooling != "attentionxl":
                self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank,find_unused_parameters=False)
            else:
                self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank,find_unused_parameters=True)
        else:
            self.device = self.config.others.device if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)

        # Initial parameters
        for p in self.model.parameters():
            if p.dim() > 1 and p.requires_grad:
                xavier_uniform_(p)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        self.save_dir = f"{self.config.model.save_dir}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}-classifier-split-{self.config.data.tag}"
        if self.multi_gpu and dist.get_rank() == 0:
            logger = setup_logger(f"{self.save_dir}/log")
        elif not self.multi_gpu:
            logger = setup_logger(f"{self.save_dir}/log")
        if self.multi_gpu:
            self.device_num = dist.get_world_size()
        if self.multi_gpu and dist.get_rank() == 0:
            #self.device_num = dist.get_world_size()
            logging.info(str(self.config))
            logging.info(f'[INFO] Model parameters: {int(total_params/1024/1024)} M')
            logging.info(f'[INFO] World size: {self.device_num}')
            
        elif not self.multi_gpu:
            logging.info(str(self.config))
            logging.info(f'[INFO] Model parameters: {int(total_params/1024/1024)} M')
        self.init_optimizer()
        self.init_scheduler()
        
        if self.config.training.loss.lower() == 'ce':
            self.loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
        else:
            raise NotImplementedError(f'Loss function {self.config.training.loss} is not implemented yet.')
        if self.multi_gpu and dist.get_rank() == 0:
            logging.info(f'[INFO] Load reactant dataset {self.config.data.data_path}/{self.config.data.rct_name_regrex}, file trunck {self.config.data.file_num_trunck}, data trunck {self.config.data.data_trunck}...')
            logging.info(f'[INFO] Load product dataset {self.config.data.data_path}/{self.config.data.pdt_name_regrex}, file trunck {self.config.data.file_num_trunck}, data trunck {self.config.data.data_trunck}...')
        elif not self.multi_gpu:
            logging.info(f'[INFO] Load reactant dataset {self.config.data.data_path}/{self.config.data.rct_name_regrex}, file trunck {self.config.data.file_num_trunck}, data trunck {self.config.data.data_trunck}...')
            logging.info(f'[INFO] Load product dataset {self.config.data.data_path}/{self.config.data.pdt_name_regrex}, file trunck {self.config.data.file_num_trunck}, data trunck {self.config.data.data_trunck}...')

        
        self.rct_dataset = MultiRXNDataset(root=self.config.data.data_path,name_regrex=self.config.data.rct_name_regrex,
                                           trunck=self.config.data.data_trunck,task=self.config.data.task,file_num_trunck=self.config.data.file_num_trunck,
                                           name_tag='rct')
        self.pdt_dataset = MultiRXNDataset(root=self.config.data.data_path,name_regrex=self.config.data.pdt_name_regrex,
                                           trunck=self.config.data.data_trunck,task=self.config.data.task,file_num_trunck=self.config.data.file_num_trunck,
                                           name_tag='pdt')
        assert len(self.rct_dataset) == len(self.pdt_dataset), 'The number of reactant and product data are not equal.'
        #self.dataset = PairDataset(self.rct_dataset,self.pdt_dataset)
        
        self.split_ids_map = get_idx_split(len(self.rct_dataset), 
                                           int(self.config.data.train_ratio*len(self.rct_dataset)), 
                                           int(self.config.data.valid_ratio*len(self.rct_dataset)), 
                                           self.config.data.seed)
        
        self.train_rct_dataset = self.rct_dataset[self.split_ids_map['train']]
        self.valid_rct_dataset = self.rct_dataset[self.split_ids_map['valid']]
        self.train_pdt_dataset = self.pdt_dataset[self.split_ids_map['train']]
        self.valid_pdt_dataset = self.pdt_dataset[self.split_ids_map['valid']]

        if not self.multi_gpu:
            self.train_dataset = PairDataset(self.train_rct_dataset,self.train_pdt_dataset)
            self.valid_dataset = PairDataset(self.valid_rct_dataset,self.valid_pdt_dataset)
            
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config.data.batch_size, shuffle=True,collate_fn=pair_collate_fn)
            self.valid_dataloader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.config.data.batch_size, shuffle=False,collate_fn=pair_collate_fn)  ## TODO
        else:
            self.train_dataset = PairDataset(self.train_rct_dataset,self.train_pdt_dataset)
            self.valid_dataset = PairDataset(self.valid_rct_dataset,self.valid_pdt_dataset)
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True)
            valid_sampler = torch.utils.data.distributed.DistributedSampler(self.valid_dataset, shuffle=False)
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.config.data.batch_size//self.device_num, sampler=train_sampler,
                                            num_workers=0)
            self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.config.data.batch_size//self.device_num, sampler=valid_sampler,
                                            num_workers=0)
            
        if self.config.model.pretrained_model:
            pretrained_inf = torch.load(f'{self.config.model.pretrained_model}/model/valid_checkpoint.pt')
            model_state_dict = pretrained_inf['model_state_dict']
            optimizer_state_dict = pretrained_inf['optimizer_state_dict']
            scheduler_state_dict = pretrained_inf['scheduler_state_dict']
            self.model.load_state_dict(model_state_dict)
            if not self.multi_gpu:
                self.model.to(self.device)
            else:
                self.model.to(self.local_rank)
            if self.config.model.fine_tune:
                if self.multi_gpu and dist.get_rank() == 0:
                    logging.info("Fine-tune setup!")
                elif not self.multi_gpu:
                    logging.info("Fine-tune setup!")
                self.fine_tune()
                self.save_dir += '_ft'
        if self.multi_gpu and dist.get_rank() == 0:
            logging.info(f'[INFO] Training results will be saved in {self.save_dir}')
        elif not self.multi_gpu:
            logging.info(f'[INFO] Training results will be saved in {self.save_dir}')

        #if not os.path.exists(self.save_dir):
        #    os.makedirs(self.save_dir)
        
        self.log_dir = f"{self.save_dir}/log"
        self.model_save_dir = f"{self.save_dir}/model"

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.config.to_json(filename=f"{self.save_dir}/parameters.json")

        
    def init_optimizer(self):
        if self.config.optimizer.optimizer.lower() == 'adam':
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                  lr=self.config.optimizer.learning_rate, 
                                  weight_decay=self.config.optimizer.weight_decay)
        elif self.config.optimizer.optimizer.lower() == 'adamw':
            self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                  lr=self.config.optimizer.learning_rate, 
                                  weight_decay=self.config.optimizer.weight_decay)
        else:
            raise Exception(f"Unsupport optimizer: '{self.config.optimizer.optimizer.lower()}'")
    
    def init_scheduler(self):
        if self.config.scheduler.type.lower() == 'steplr':
            self.scheduler = StepLR(self.optimizer, 
                                    step_size=self.config.scheduler.lr_decay_step_size, 
                                    gamma=self.config.scheduler.lr_decay_factor)
        elif self.config.scheduler.type.lower() == 'warmup':
            self.scheduler = get_linear_scheduler_with_warmup(self.optimizer,
                                                              num_warmup_steps=self.config.scheduler.warmup_step, 
                                                              num_training_steps=self.config.training.epoch)
        elif self.config.scheduler.type.lower() == 'noamlr':
            self.scheduler = NoamLR(self.optimizer,model_size=self.config.model.emb_dim,
                                    warmup_steps=self.config.scheduler.warmup_step)
        else:
            raise Exception(f"Unsupport scheduler: '{self.config.scheduler.type.lower()}'")
        
    def fine_tune(self):
        
        if self.config.model.trainable == 'decoder':
            trainable_params_id = list(map(id,self.model.decoder.parameters()))
            fixed_params = filter(lambda p: id(p) not in trainable_params_id,self.model.parameters())
            for p in  fixed_params:
                p.requires_grad = False
        else:
            raise Exception("trainable should be in ['decoder']")
        self.init_optimizer()
        self.init_scheduler()
        
    def train(self):
        self.model.train()
        loss_accum = 0
        acc_lst = []
        if self.multi_gpu:
            self.train_dataloader.sampler.set_epoch(self.epoch)
        for step, batch_data in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            rct_data,pdt_data = batch_data
            if not self.multi_gpu:
                rct_data = rct_data.to(self.device)
                pdt_data = pdt_data.to(self.device)
            else:
                rct_data = rct_data.to(self.local_rank)
                pdt_data = pdt_data.to(self.local_rank)


            out = self.model([rct_data,pdt_data])
            loss = self.loss_func(out, rct_data.y)
            acc_lst.append((out.argmax(dim=1) == rct_data.y).float().mean().cpu())
            loss.backward()
            if (step+1) % self.config.training.accum == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.clip_norm)
                self.optimizer.step()
                if self.config.scheduler.type.lower() == 'noamlr':
                    self.scheduler.step()
                g_norm = grad_norm(self.model)
                self.model.zero_grad()
            
            loss_accum += loss.detach().cpu().item()
            if (step+1) % self.config.training.log_iter_step == 0:
                p_norm = param_norm(self.model)
                lr_cur = get_lr(self.optimizer)
                if self.multi_gpu and dist.get_rank() == 0:
                    logging.info(f'Training step {step+1}, gradient norm: {g_norm:.8f}, parameters norm: {p_norm:.8f}, lr: {lr_cur}, loss: {loss_accum/(step+1):.4f}, acc: {np.mean(acc_lst):.4f}')
                elif not self.multi_gpu:
                    logging.info(f'Training step {step+1}, gradient norm: {g_norm:.8f}, parameters norm: {p_norm:.8f}, lr: {lr_cur}, loss: {loss_accum/(step+1):.4f}, acc: {np.mean(acc_lst):.4f}')
        loss_ave = loss_accum/(step+1)
        acc_ave = np.mean(acc_lst)
        return loss_ave,acc_ave
    def val(self,dataloader):
        self.model.eval()
        if not self.multi_gpu:
            preds = torch.Tensor([]).to(self.device)
            targets = torch.Tensor([]).to(self.device)
        else:
            preds = torch.Tensor([]).to(self.local_rank)
            targets = torch.Tensor([]).to(self.local_rank)
            dataloader.sampler.set_epoch(self.epoch)
        with torch.no_grad():
            for step, batch_data in enumerate(dataloader):
                rct_data,pdt_data = batch_data
                if not self.multi_gpu:
                    rct_data = rct_data.to(self.device)
                    pdt_data = pdt_data.to(self.device)
                else:
                    rct_data = rct_data.to(self.local_rank)
                    pdt_data = pdt_data.to(self.local_rank)
                out = self.model([rct_data,pdt_data])
                pred = torch.argmax(out, dim=1)
                preds = torch.cat([preds, pred.detach_()], dim=0)
                targets = torch.cat([targets, rct_data.y.unsqueeze(1)], dim=0)
        return (preds == targets.view(-1)).float().mean().cpu().item()
    
    def run(self):
        best_valid = -float('inf')
        best_test = -float('inf')
        self.model.zero_grad()
        for self.epoch in range(1, self.config.training.epoch + 1):
            logging.info(f'============= Epoch {self.epoch} =============')
            
            logging.info('Training...')
            
            train_loss,train_acc = self.train()

            logging.info('Evaluating...')
            valid_acc = self.val(self.valid_dataloader)
            
            lr_cur = get_lr(self.optimizer)
            
            logging.info(f'Train loss: {train_loss:.8f}, train acc: {train_acc:.8f}, valid acc: {valid_acc:.8f}, lr: {lr_cur}')

            self.writer.add_scalar('train_loss', train_loss, self.epoch)
            self.writer.add_scalar('valid_acc', valid_acc, self.epoch)
            
            if valid_acc > best_valid:
                best_valid = valid_acc
                best_test = valid_acc

                logging.info('Saving checkpoint...')
                checkpoint = {'epoch': self.epoch, 
                                'model_state_dict': self.model.state_dict(), 
                                'optimizer_state_dict': self.optimizer.state_dict(), 
                                'scheduler_state_dict': self.scheduler.state_dict(), 
                                'best_valid_mae': best_valid}
                torch.save(checkpoint, os.path.join(self.model_save_dir, 'valid_checkpoint.pt'))
            if self.config.scheduler.type.lower() == 'steplr':
                self.scheduler.step()
            #torch.distributed.barrier()  ## 强制同步
        if self.multi_gpu and dist.get_rank() == 0:
            logging.info(f'Best validation accuracy so far: {best_valid}')
            logging.info(f'Test accuracy when got best validation result: {best_test}')
        elif not self.multi_gpu:
            logging.info(f'Best validation accuracy so far: {best_valid}')
            logging.info(f'Test accuracy when got best validation result: {best_test}')
        self.writer.close()

class SPLITRegressorTrainer():
    def __init__(self,config):
        self.config = config
        self.device = self.config.others.device if torch.cuda.is_available() else "cpu"
        self.use_mid_inf = eval(self.config.model.use_mid_inf)
        if self.config.data.rct_data_file:
            prefix = self.config.data.rct_data_file.split('.')[0][:20]
        else:
            prefix = self.config.data.train_rct_data_file.split('.')[0][:20]
        if self.use_mid_inf:
            prefix = prefix + '_mid'
        prefix = f"{prefix}-{self.config.others.tag}"
        self.save_dir = f"{self.config.model.save_dir}/{prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not self.config.model.pretrained_model_path:
            logging.info('Training from scratch')
            '''
            self.model = RXNGRegressor(emb_dim=self.config.model.emb_dim,
                                        gnn_type=self.config.model.gnn_type,
                                        gnn_aggr=self.config.model.gnn_aggr,
                                        gnum_layer=self.config.model.gnn_num_layer,
                                        node_readout=self.config.model.node_readout,
                                        num_heads=self.config.model.num_heads,
                                        JK=self.config.model.gnn_jk,
                                        graph_pooling=self.config.model.graph_pooling,
                                        tnum_layer=self.config.model.trans_num_layer,
                                        trans_readout=self.config.model.trans_readout,
                                        onum_layer=self.config.model.output_num_layer,
                                        drop_ratio=self.config.model.drop_ratio,
                                        output_size=1,
                                        output_norm=eval(self.config.model.output_norm),
                                        split_process=True,
                                        split_merge_method=self.config.model.split_merge_method,
                                        output_act_func=self.config.model.output_act_func,
                                        rct_batch_norm=eval(self.config.model.rct_batch_norm),
                                        pdt_batch_norm=eval(self.config.model.pdt_batch_norm),
                                        use_mid_inf=self.use_mid_inf,
                                        pretrained_mid_encoder=None,
                                        mid_iteract_method=self.config.model.mid_iteract_method,
                                        mid_batch_norm=eval(self.config.model.mid_batch_norm),
                                        mid_layer_num=self.config.model.mid_layer_num)
            '''
            input_param = {"emb_dim":self.config.model.emb_dim,
                            "gnn_type":self.config.model.gnn_type,
                            "gnn_aggr":self.config.model.gnn_aggr,
                            "gnum_layer":self.config.model.gnn_num_layer,
                            "node_readout":self.config.model.node_readout,
                            "num_heads":self.config.model.num_heads,
                            "JK":self.config.model.gnn_jk,
                            "graph_pooling":self.config.model.graph_pooling,
                            "tnum_layer":self.config.model.trans_num_layer,
                            "trans_readout":self.config.model.trans_readout,
                            "onum_layer":self.config.model.output_num_layer,
                            "drop_ratio":self.config.model.drop_ratio,
                            "output_size":1,
                            "output_norm":eval(self.config.model.output_norm),
                            "split_process":True,
                            "split_merge_method":self.config.model.split_merge_method,
                            "output_act_func":self.config.model.output_act_func,
                            "rct_batch_norm":eval(self.config.model.rct_batch_norm),
                            "pdt_batch_norm":eval(self.config.model.pdt_batch_norm),
                            "use_mid_inf":self.use_mid_inf,
                            "mid_iteract_method":self.config.model.mid_iteract_method,
                            "mid_batch_norm":eval(self.config.model.mid_batch_norm),
                            "mid_layer_num":self.config.model.mid_layer_num}
            
            rxng = RXNGraphormer("regression",align_config(input_param,"regressor"),"") # pretrain models are all None
            self.model = rxng.get_model()






            

        else:
            self.pretrained_model_freeze = self.config.model.pretrained_model_freeze
            self.pretrained_lr_scaled_coef = self.config.model.pretrained_lr_scaled_coef
            self.save_dir += '_ft'
            logging.info('Loading pretrained model')
            pretrained_para_json = f"{self.config.model.pretrained_model_path}/parameters.json"
            with open(pretrained_para_json,'r') as fr:
                pretrained_config_dict = json.load(fr)
            pretrained_config = Box(pretrained_config_dict)
            ckpt_file = f"{self.config.model.pretrained_model_path}/model/valid_checkpoint.pt"
            ckpt_inf = torch.load(ckpt_file,map_location=self.device)
            '''
            self.pretrained_model = RXNGClassifier(emb_dim=pretrained_config.model.emb_dim,
                                                    gnn_type=pretrained_config.model.gnn_type,
                                                    gnn_aggr=pretrained_config.model.gnn_aggr,
                                                    gnum_layer=pretrained_config.model.gnn_num_layer,
                                                    node_readout=pretrained_config.model.node_readout,
                                                    num_heads=pretrained_config.model.num_heads,
                                                    JK=pretrained_config.model.gnn_jk,
                                                    graph_pooling=pretrained_config.model.graph_pooling,
                                                    tnum_layer=pretrained_config.model.trans_num_layer,
                                                    trans_readout=pretrained_config.model.trans_readout,
                                                    onum_layer=pretrained_config.model.output_num_layer,
                                                    drop_ratio=pretrained_config.model.drop_ratio,
                                                    output_size=2,split_process=True,
                                                    split_merge_method=pretrained_config.model.split_merge_method,
                                                    output_act_func=self.config.model.output_act_func)
            '''

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
                            "output_act_func":self.config.model.output_act_func}
            
            rxng = RXNGraphormer("classification",align_config(input_param,"classifier"),"")
            self.pretrained_model = rxng.get_model()
            
            pretrained_model_params = ckpt_inf['model_state_dict']
            if list(pretrained_model_params.keys())[0].startswith("module."):
                pretrained_model_params = update_dict_key(pretrained_model_params, "module.")
            self.pretrained_model.load_state_dict(pretrained_model_params)
            rct_encoder = self.pretrained_model.rct_encoder
            pdt_encoder = self.pretrained_model.pdt_encoder
            ## before initializing all model parameters, freeze the parameters in pretrained part
            for param in rct_encoder.parameters():
                param.requires_grad = False
            for param in pdt_encoder.parameters():
                param.requires_grad = False    

            '''
            self.model = RXNGRegressor(emb_dim=self.config.model.emb_dim,
                                        gnn_type=self.config.model.gnn_type,
                                        gnn_aggr=self.config.model.gnn_aggr,
                                        gnum_layer=self.config.model.gnn_num_layer,
                                        node_readout=self.config.model.node_readout,
                                        num_heads=self.config.model.num_heads,
                                        JK=self.config.model.gnn_jk,
                                        graph_pooling=self.config.model.graph_pooling,
                                        tnum_layer=self.config.model.trans_num_layer,
                                        trans_readout=self.config.model.trans_readout,
                                        onum_layer=self.config.model.output_num_layer,
                                        drop_ratio=self.config.model.drop_ratio,
                                        output_size=1,pretrained_rct_encoder=rct_encoder,
                                        pretrained_pdt_encoder=pdt_encoder,
                                        output_norm=eval(self.config.model.output_norm),
                                        split_process=True,
                                        split_merge_method=self.config.model.split_merge_method,
                                        output_act_func=self.config.model.output_act_func,
                                        rct_batch_norm=eval(self.config.model.rct_batch_norm),
                                        pdt_batch_norm=eval(self.config.model.pdt_batch_norm),
                                        use_mid_inf=self.use_mid_inf,
                                        pretrained_mid_encoder=None,
                                        mid_iteract_method=self.config.model.mid_iteract_method,
                                        mid_batch_norm=eval(self.config.model.mid_batch_norm),
                                        mid_layer_num=self.config.model.mid_layer_num)
                '''
            

            input_param = {"emb_dim":self.config.model.emb_dim,
                            "gnn_type":self.config.model.gnn_type,
                            "gnn_aggr":self.config.model.gnn_aggr,
                            "gnum_layer":self.config.model.gnn_num_layer,
                            "node_readout":self.config.model.node_readout,
                            "num_heads":self.config.model.num_heads,
                            "JK":self.config.model.gnn_jk,
                            "graph_pooling":self.config.model.graph_pooling,
                            "tnum_layer":self.config.model.trans_num_layer,
                            "trans_readout":self.config.model.trans_readout,
                            "onum_layer":self.config.model.output_num_layer,
                            "drop_ratio":self.config.model.drop_ratio,
                            "output_size":1,
                            "output_norm":eval(self.config.model.output_norm),
                            "split_process":True,
                            "split_merge_method":self.config.model.split_merge_method,
                            "output_act_func":self.config.model.output_act_func,
                            "rct_batch_norm":eval(self.config.model.rct_batch_norm),
                            "pdt_batch_norm":eval(self.config.model.pdt_batch_norm),
                            "use_mid_inf":self.use_mid_inf,
                            "mid_iteract_method":self.config.model.mid_iteract_method,
                            "mid_batch_norm":eval(self.config.model.mid_batch_norm),
                            "mid_layer_num":self.config.model.mid_layer_num}
            
            rxng = RXNGraphormer("regression",align_config(input_param,"regressor"),"",{"pretrained_encoder": None,
                                                                                        "pretrained_rct_encoder": rct_encoder,
                                                                                        "pretrained_pdt_encoder": pdt_encoder,
                                                                                        "pretrained_mid_encoder": None})
            self.model = rxng.get_model()


        self.model.to(self.device)

        # Initial parameters
        for p in self.model.parameters():
            if p.dim() > 1 and p.requires_grad:
                xavier_uniform_(p)
        if self.config.model.pretrained_model_path and not self.pretrained_model_freeze:
            ## unfreeze pretrained model
            logging.info(f'[INFO] Unfreeze pretrained model')
            for param in self.model.rct_encoder.parameters():
                param.requires_grad = True
            for param in self.model.pdt_encoder.parameters():
                param.requires_grad = True

        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger = setup_logger(f"{self.save_dir}/log")
        logging.info(str(self.config))
        logging.info(f'[INFO] Model parameters: {int(total_params/1024/1024)} M')
        self.init_optimizer()
        self.init_scheduler()
        
        if self.config.training.loss.lower() == 'l1':
            self.loss_func = torch.nn.L1Loss()
        elif self.config.training.loss.lower() == 'l2':
            self.loss_func = torch.nn.MSELoss()
        


        if self.config.data.rct_data_file:
            logging.info(f'[INFO] Load dataset {self.config.data.data_path}/{self.config.data.rct_data_file} trunck {self.config.data.data_trunck}...')
            self.rct_dataset = RXNDataset(root=self.config.data.data_path,name=self.config.data.rct_data_file,trunck=self.config.data.data_trunck)
            self.pdt_dataset = RXNDataset(root=self.config.data.data_path,name=self.config.data.pdt_data_file,trunck=self.config.data.data_trunck)
            
            assert len(self.rct_dataset) == len(self.pdt_dataset), "The number of reactants and products should be the same."
            self.split_ids_map = get_idx_split(len(self.rct_dataset), 
                                               int(self.config.data.train_ratio*len(self.rct_dataset)), 
                                               int(self.config.data.valid_ratio*len(self.rct_dataset)), 
                                               self.config.data.seed)
            self.train_rct_dataset = self.rct_dataset[self.split_ids_map['train']]
            self.valid_rct_dataset = self.rct_dataset[self.split_ids_map['valid']]
            self.train_pdt_dataset = self.pdt_dataset[self.split_ids_map['train']]
            self.valid_pdt_dataset = self.pdt_dataset[self.split_ids_map['valid']]

            
            if len(self.split_ids_map['test']) > 10:
                self.test_rct_dataset = self.rct_dataset[self.split_ids_map['test']]
                self.test_pdt_dataset = self.pdt_dataset[self.split_ids_map['test']]
            else:
                self.test_rct_dataset = self.valid_rct_dataset
                self.test_pdt_dataset = self.valid_pdt_dataset
            logging.info(f'[INFO] Split dataset into train: {len(self.train_rct_dataset)}, valid: {len(self.valid_rct_dataset)}, test: {len(self.test_rct_dataset)}. Random seed: {self.config.data.seed}')
        else:
            logging.info(f'[INFO] Load dataset {self.config.data.data_path}/{self.config.data.train_rct_data_file} and {self.config.data.train_pdt_data_file} trunck {self.config.data.data_trunck}...')
            self.train_rct_dataset = RXNDataset(root=self.config.data.data_path,name=self.config.data.train_rct_data_file,trunck=self.config.data.data_trunck)
            self.train_pdt_dataset = RXNDataset(root=self.config.data.data_path,name=self.config.data.train_pdt_data_file,trunck=self.config.data.data_trunck)
            logging.info(f'[INFO] Load dataset {self.config.data.data_path}/{self.config.data.val_rct_data_file} and {self.config.data.val_pdt_data_file} trunck {self.config.data.data_trunck}...')
            self.valid_rct_dataset = RXNDataset(root=self.config.data.data_path,name=self.config.data.val_rct_data_file,trunck=self.config.data.data_trunck)
            self.valid_pdt_dataset = RXNDataset(root=self.config.data.data_path,name=self.config.data.val_pdt_data_file,trunck=self.config.data.data_trunck)
            logging.info(f'[INFO] Load dataset {self.config.data.data_path}/{self.config.data.test_rct_data_file} and {self.config.data.test_pdt_data_file} trunck {self.config.data.data_trunck}...')
            self.test_rct_dataset = RXNDataset(root=self.config.data.data_path,name=self.config.data.test_rct_data_file,trunck=self.config.data.data_trunck)
            self.test_pdt_dataset = RXNDataset(root=self.config.data.data_path,name=self.config.data.test_pdt_data_file,trunck=self.config.data.data_trunck)
            assert len(self.train_rct_dataset) == len(self.train_pdt_dataset) and \
                   len(self.valid_rct_dataset) == len(self.valid_pdt_dataset) and \
                   len(self.test_rct_dataset) == len(self.test_pdt_dataset), 'The length of the datasets are not equal!'

        if self.use_mid_inf:
            if self.config.data.rct_data_file:
                self.mid_dataset = RXNDataset(root=self.config.data.data_path,name=self.config.data.mid_data_file,trunck=self.config.data.data_trunck)
                self.train_mid_dataset = self.mid_dataset[self.split_ids_map['train']]
                self.valid_mid_dataset = self.mid_dataset[self.split_ids_map['valid']]
                
                if len(self.split_ids_map['test']) > 10:
                    self.test_mid_dataset = self.mid_dataset[self.split_ids_map['test']]
                else:
                    self.test_mid_dataset = self.valid_mid_dataset
                
                
            else:
                logging.info(f'[INFO] Load dataset {self.config.data.data_path}/{self.config.data.train_mid_data_file} trunck {self.config.data.data_trunck}...')
                self.train_mid_dataset = RXNDataset(root=self.config.data.data_path,name=self.config.data.train_mid_data_file,trunck=self.config.data.data_trunck)
                logging.info(f'[INFO] Load dataset {self.config.data.data_path}/{self.config.data.val_mid_data_file} trunck {self.config.data.data_trunck}...')
                self.valid_mid_dataset = RXNDataset(root=self.config.data.data_path,name=self.config.data.val_mid_data_file,trunck=self.config.data.data_trunck)
                logging.info(f'[INFO] Load dataset {self.config.data.data_path}/{self.config.data.test_mid_data_file} trunck {self.config.data.data_trunck}...')
                self.test_mid_dataset = RXNDataset(root=self.config.data.data_path,name=self.config.data.test_mid_data_file,trunck=self.config.data.data_trunck)
            assert len(self.train_mid_dataset) == len(self.train_rct_dataset) == len(self.train_pdt_dataset), 'Length of train dataset is not equal'
            assert len(self.valid_mid_dataset) == len(self.valid_rct_dataset) == len(self.valid_pdt_dataset), 'Length of valid dataset is not equal'
            assert len(self.test_mid_dataset) == len(self.test_rct_dataset) == len(self.test_pdt_dataset), 'Length of test dataset is not equal'
            self.train_dataset = TripleDataset(self.train_rct_dataset,self.train_pdt_dataset,self.train_mid_dataset)
            self.valid_dataset = TripleDataset(self.valid_rct_dataset,self.valid_pdt_dataset,self.valid_mid_dataset)
            self.test_dataset = TripleDataset(self.test_rct_dataset,self.test_pdt_dataset,self.test_mid_dataset)
            
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config.data.batch_size, shuffle=True,collate_fn=triple_collate_fn)
            self.valid_dataloader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.config.data.batch_size, shuffle=False,collate_fn=triple_collate_fn)
            self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.config.data.batch_size, shuffle=False,collate_fn=triple_collate_fn)
        else:
            assert len(self.train_rct_dataset) == len(self.train_pdt_dataset), 'Length of train dataset is not equal'
            assert len(self.valid_rct_dataset) == len(self.valid_pdt_dataset), 'Length of valid dataset is not equal'
            assert len(self.test_rct_dataset) == len(self.test_pdt_dataset), 'Length of test dataset is not equal'
            self.train_dataset = PairDataset(self.train_rct_dataset,self.train_pdt_dataset)
            self.valid_dataset = PairDataset(self.valid_rct_dataset,self.valid_pdt_dataset)
            self.test_dataset = PairDataset(self.test_rct_dataset,self.test_pdt_dataset)
            
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config.data.batch_size, shuffle=True,collate_fn=pair_collate_fn)
            self.valid_dataloader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.config.data.batch_size, shuffle=False,collate_fn=pair_collate_fn)
            self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.config.data.batch_size, shuffle=False,collate_fn=pair_collate_fn)
       

        logging.info(f'[INFO] Training results will be saved in {self.save_dir}')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.log_dir = f"{self.save_dir}/log"
        self.model_save_dir = f"{self.save_dir}/model"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.config.to_json(filename=f"{self.save_dir}/parameters.json")
        
    def init_optimizer(self):
        if self.config.model.pretrained_model_path and not self.pretrained_model_freeze:
            param_lst = [{'params': filter(lambda p: p.requires_grad, self.model.rct_encoder.parameters()), 'lr': self.config.optimizer.learning_rate * self.pretrained_lr_scaled_coef},
                        {'params': filter(lambda p: p.requires_grad, self.model.pdt_encoder.parameters()), 'lr': self.config.optimizer.learning_rate * self.pretrained_lr_scaled_coef},
                        {'params': filter(lambda p: p.requires_grad, self.model.decoder.parameters()), 'lr': self.config.optimizer.learning_rate}]
            if self.use_mid_inf:
                param_lst.append({'params': filter(lambda p: p.requires_grad, self.model.mid_encoder.parameters()), 'lr': self.config.optimizer.learning_rate})
                param_lst.append({'params': filter(lambda p: p.requires_grad, self.model.mid_iteract.parameters()), 'lr': self.config.optimizer.learning_rate})
                param_lst.append({'params': filter(lambda p: p.requires_grad, self.model.mid_decoder.parameters()), 'lr': self.config.optimizer.learning_rate})
        if self.config.optimizer.optimizer.lower() == 'adam':
            if not self.config.model.pretrained_model_path or self.pretrained_model_freeze:
                self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                    lr=self.config.optimizer.learning_rate, 
                                    weight_decay=self.config.optimizer.weight_decay)
            else:
                logging.info(f'[INFO] Use different learning rate for pretrained model')

                self.optimizer = Adam(param_lst, lr=self.config.optimizer.learning_rate,weight_decay=self.config.optimizer.weight_decay)
        elif self.config.optimizer.optimizer.lower() == 'adamw':
            if not self.config.model.pretrained_model_path or self.pretrained_model_freeze:
                self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                    lr=self.config.optimizer.learning_rate, 
                                    weight_decay=self.config.optimizer.weight_decay)
            else:
                logging.info(f'[INFO] Use different learning rate for pretrained model')
                self.optimizer = AdamW(param_lst,lr=self.config.optimizer.learning_rate,weight_decay=self.config.optimizer.weight_decay)
        else:
            raise Exception(f"Unsupport optimizer: '{self.config.optimizer.optimizer.lower()}'")
    
    def init_scheduler(self):
        if self.config.scheduler.type.lower() == 'steplr':
            self.scheduler = StepLR(self.optimizer, 
                                    step_size=self.config.scheduler.lr_decay_step_size, 
                                    gamma=self.config.scheduler.lr_decay_factor)
        elif self.config.scheduler.type.lower() == 'warmup':
            self.scheduler = get_linear_scheduler_with_warmup(self.optimizer,
                                                              num_warmup_steps=self.config.scheduler.warmup_step, 
                                                              num_training_steps=self.config.training.epoch)
        elif self.config.scheduler.type.lower() == 'noamlr':
            self.scheduler = NoamLR(self.optimizer,model_size=self.config.model.emb_dim,
                                    warmup_steps=self.config.scheduler.warmup_step)
        else:
            raise Exception(f"Unsupport scheduler: '{self.config.scheduler.type.lower()}'")

    def train(self):
        self.model.train()
        loss_accum = 0
        for step, batch_data in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            if not self.use_mid_inf:
                rct_data,pdt_data = batch_data
                rct_data = rct_data.to(self.device)
                pdt_data = pdt_data.to(self.device)
                out = self.model([rct_data,pdt_data])
            else:
                rct_data,pdt_data,mid_data = batch_data
                rct_data = rct_data.to(self.device)
                pdt_data = pdt_data.to(self.device)
                mid_data = mid_data.to(self.device)
                out = self.model([rct_data,pdt_data,mid_data])

            loss = self.loss_func(out, rct_data.y.unsqueeze(1))
            loss.backward()
            if (step+1) % self.config.training.accum == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.clip_norm)
                self.optimizer.step()
                if self.config.scheduler.type.lower() == 'noamlr':
                    self.scheduler.step()
                g_norm = grad_norm(self.model)
                self.model.zero_grad()
            loss_accum += loss.detach().cpu().item()
        loss_ave = loss_accum/(step+1)
        return loss_ave
    def val(self,dataloader):
        self.model.eval()
        preds = torch.Tensor([]).to(self.device)
        targets = torch.Tensor([]).to(self.device)
        with torch.no_grad():
            for step, batch_data in enumerate(dataloader):
                if not self.use_mid_inf:
                    rct_data,pdt_data = batch_data
                    rct_data = rct_data.to(self.device)
                    pdt_data = pdt_data.to(self.device)
                    out = self.model([rct_data,pdt_data])
                else:
                    rct_data,pdt_data,mid_data = batch_data
                    rct_data = rct_data.to(self.device)
                    pdt_data = pdt_data.to(self.device)
                    mid_data = mid_data.to(self.device)
                    out = self.model([rct_data,pdt_data,mid_data])
                preds = torch.cat([preds, out.detach_()], dim=0)
                targets = torch.cat([targets, rct_data.y.unsqueeze(1)], dim=0)
        r2 = r2_score(targets.cpu(),preds.cpu())
        return torch.mean(torch.abs(targets - preds)).cpu().item(),r2
    
    def run(self):
        best_valid = float('inf')
        best_test = float('inf')
        best_valid_r2 = -float('inf')
        best_test_r2 = -float('inf')
        self.model.zero_grad()
        for epoch in range(1, self.config.training.epoch + 1):
            logging.info(f'============= Epoch {epoch} =============')
            
            logging.info('Training...')
            
            train_loss = self.train()

            logging.info('Evaluating...')
            valid_mae,valid_r2 = self.val(self.valid_dataloader)

            logging.info('Testing...')
            test_mae,test_r2 = self.val(self.test_dataloader)
            
            lr_cur = get_lr(self.optimizer)
            
            logging.info(f'Train: {train_loss:.8f}, validation mae: {valid_mae:.8f}, r2: {valid_r2}, test mae: {test_mae:.8f}, r2: {test_r2}, lr: {lr_cur}')
            

            self.writer.add_scalar('train_loss', train_loss, epoch)
            self.writer.add_scalar('valid_mae', valid_mae, epoch)
            self.writer.add_scalar('test_mae', test_mae, epoch)
            
            if valid_mae < best_valid:
                best_valid = valid_mae
                best_test = test_mae
                best_valid_r2 = valid_r2
                best_test_r2 = test_r2
                logging.info('Saving checkpoint...')
                checkpoint = {'epoch': epoch, 
                                'model_state_dict': self.model.state_dict(), 
                                'optimizer_state_dict': self.optimizer.state_dict(), 
                                'scheduler_state_dict': self.scheduler.state_dict(), 
                                'best_valid_mae': best_valid,
                                'best_valid_r2':best_valid_r2}
                torch.save(checkpoint, os.path.join(self.model_save_dir, 'valid_checkpoint.pt'))
            if self.config.scheduler.type.lower() == 'steplr':
                self.scheduler.step()

        logging.info(f'Best validation MAE so far: {best_valid}, R2: {best_valid_r2}')
        logging.info(f'Test MAE when got best validation result: {best_test}, R2: {best_test_r2}')
        

        self.writer.close()

class SequenceTrainer():
        
    def __init__(self,config):
        self.config = config
        assert self.config.model.task in ["forward_prediction","retrosynthesis"], "Task must be forward_prediction or retrosynthesis"
        self.multi_gpu = self.config.others.multi_gpu
        self.device = self.config.others.device if torch.cuda.is_available() else 'cpu'
        self.vocab = load_vocab(f'{self.config.data.data_path}/{self.config.data.vocab_file}')
        self.vocab_rev = [k for k, v in sorted(self.vocab.items(), key=lambda tup: tup[1])]
        self.save_dir = f"{self.config.model.save_dir}/seq-v2-{self.config.data.data_path.split('/')[-1]}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.topk = 10
        self.test_eval_after = 100
        self.save_improve = self.config.others.save_improve
        if hasattr(self.config.model,"pretrained_model_path") and self.config.model.pretrained_model_path:
            self.save_dir += "_ft"
        if self.multi_gpu:
            self.local_rank = self.config.others.local_rank
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend='nccl') # initialize the default distributed process group
            if dist.get_rank() == 0:
                self.device_num = dist.get_world_size()
                logger = setup_logger(f"{self.save_dir}/log")
                logging.info(self.config)
                logging.info(f"[INFO] Training results will be saved to {self.save_dir}")
                logging.info(f'[INFO] Using {self.device_num} GPUs')
                sys.stdout.flush()
            
        elif not self.multi_gpu:
            logger = setup_logger(f"{self.save_dir}/log")
            self.device_num = 1
            logging.info(self.config)
            logging.info(f"[INFO] Training results will be saved to {self.save_dir}")
            logging.info(f'[INFO] Using {self.device_num} GPU')
            sys.stdout.flush()

        self.log_dir = f"{self.save_dir}/log"
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.config.to_json(filename=f"{self.save_dir}/parameters.json")
        self.model_save_dir = f"{self.save_dir}/model"
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        
        #self.model = RXNG2Sequencer(self.config,self.vocab)
        rxng = RXNGraphormer("sequence_generation",self.config,self.vocab)
        self.model = rxng.get_model()

        for p in self.model.parameters():
            if p.dim() > 1 and p.requires_grad:
                xavier_uniform_(p)
        if hasattr(self.config.model,"pretrained_model_path") and self.config.model.pretrained_model_path:
            logging.info(f'[INFO] Load pretrained model {self.config.model.pretrained_model_path}...')
            pretrained_para_json = f"{self.config.model.pretrained_model_path}/parameters.json"
            with open(pretrained_para_json,'r') as fr:
                pretrained_config_dict = json.load(fr)
            pretrained_config = Box(pretrained_config_dict)
            ckpt_file = f"{self.config.model.pretrained_model_path}/model/valid_checkpoint.pt"
            ckpt_inf = torch.load(ckpt_file,map_location="cpu")

            '''
            pretrained_model = RXNGClassifier(emb_dim=pretrained_config.model.emb_dim,
                                            gnn_type=pretrained_config.model.gnn_type,
                                            gnn_aggr=pretrained_config.model.gnn_aggr,
                                            gnum_layer=pretrained_config.model.gnn_num_layer,
                                            node_readout=pretrained_config.model.node_readout,
                                            num_heads=pretrained_config.model.num_heads,
                                            JK=pretrained_config.model.gnn_jk,
                                            graph_pooling=pretrained_config.model.graph_pooling,
                                            tnum_layer=pretrained_config.model.trans_num_layer,
                                            trans_readout=pretrained_config.model.trans_readout,
                                            onum_layer=pretrained_config.model.output_num_layer,
                                            drop_ratio=pretrained_config.model.drop_ratio,
                                            output_size=2,split_process=True,
                                            split_merge_method=pretrained_config.model.split_merge_method,
                                            output_act_func=pretrained_config.model.output_act_func)
            '''
            
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
            pretrained_model = rxng.get_model()



            
            pretrained_model.load_state_dict(update_dict_key(ckpt_inf["model_state_dict"]))
            if self.config.model.task == "retrosynthesis":
                logging.info("[INFO] Use pdt_enocder for retrosynthesis")
                pretrained_encoder = pretrained_model.pdt_encoder.rxn_graph_encoder
            else:
                logging.info("[INFO] Use rct_encoder for forward_prediction")
                pretrained_encoder = pretrained_model.rct_encoder.rxn_graph_encoder
            logging.info("[INFO] Load pretrained model to encoder")
            self.model.encoder.load_state_dict(pretrained_encoder.state_dict())
            #for param in pretrained_encoder.parameters():
            #    param.requires_grad = False
            #self.model.encoder = pretrained_encoder
        else:
            logging.info(f'[INFO] No pretrained model is provided, train model from scratch')
            # Training from scratch
        
        total_params = sum(p.numel() for p in self.model.parameters())
        if self.multi_gpu and dist.get_rank() == 0:
            logging.info(f'[INFO] Model parameters: {int(total_params/1024/1024)} M')
            sys.stdout.flush()
        elif not self.multi_gpu:
            logging.info(f'[INFO] Model parameters: {int(total_params/1024/1024)} M')
            sys.stdout.flush()

        
        
        if hasattr(self.config.model,"pretrained_model_path") and self.config.model.pretrained_model_path and self.config.model.pretrained_model_freeze:
            ## freeze pretrained model
            logging.info("[INFO] Freeze pretrained model")
            for param in self.model.encoder.parameters():
                param.requires_grad = False
                
        if not self.multi_gpu:
            self.model.to(self.device)
        else:
            self.model.to(self.local_rank)
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank,find_unused_parameters=True)
        self.model.train()
        

        self.init_optimizer()
        self.init_scheduler()

        if self.multi_gpu and dist.get_rank() == 0:
            logging.info(f'[INFO] Load dataset {self.config.data.data_path}...')
        elif not self.multi_gpu:
            logging.info(f'[INFO] Load dataset {self.config.data.data_path}...')

        self.train_dataset = RXNG2SDataset(root=self.config.data.data_path,src_file=self.config.data.train_src_file,tgt_file=self.config.data.train_tgt_file,oh=False,multi_process=False,vocab_file=self.config.data.vocab_file,train=True)
        self.valid_dataset = RXNG2SDataset(root=self.config.data.data_path,src_file=self.config.data.valid_src_file,tgt_file=self.config.data.valid_tgt_file,oh=False,multi_process=False,vocab_file=self.config.data.vocab_file,train=False)
        self.test_dataset = RXNG2SDataset(root=self.config.data.data_path,src_file=self.config.data.test_src_file,tgt_file=self.config.data.test_tgt_file,oh=False,multi_process=False,vocab_file=self.config.data.vocab_file,train=False)
        if not self.multi_gpu:
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.config.data.batch_size, shuffle=True, num_workers=2)
            self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.config.data.batch_size, shuffle=False, num_workers=2)
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.config.data.batch_size, shuffle=False, num_workers=2)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True)
            valid_sampler = torch.utils.data.distributed.DistributedSampler(self.valid_dataset, shuffle=False)
            test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_dataset, shuffle=False)
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.config.data.batch_size//dist.get_world_size(), sampler=train_sampler,
                                            num_workers=0)
            self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.config.data.batch_size//dist.get_world_size(), sampler=valid_sampler,
                                            num_workers=0)
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.config.data.batch_size//dist.get_world_size(), sampler=test_sampler,
                                            num_workers=0)
        self.ground_truth_smiles_lst = ["".join([self.vocab_rev[idx] for idx in self.test_dataset[idx].tgt_token_ids[0][:int(self.test_dataset[idx].tgt_lens[0])-1]]) for idx in range(len(self.test_dataset))]
        self.total_step = 0
        self.accum = 0
        self.losses, self.accs = [], []
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.others.enable_amp)
        
        self.start_time = time.time()
        #self.best_val_loss = float('inf')
        self.best_val_acc = 0
        self.best_seq_acc = 0
        self.best_token_acc = 0
        if self.multi_gpu and dist.get_rank() == 0:
            logging.info("[INFO] Model initialized")
            sys.stdout.flush()
        elif not self.multi_gpu:
            logging.info("[INFO] Model initialized")
            sys.stdout.flush()
    
    def train(self):
        self.model.train()
        self.model.zero_grad()
        losses = []
        accs = []
        if self.multi_gpu:
            self.train_dataloader.sampler.set_epoch(self.epoch)
        for step, batch in enumerate(self.train_dataloader):
            if self.config.model.add_empty_node:
                batch = add_dense_empty_node_edge(batch)
            
            if not self.multi_gpu:
                batch.to(self.device)
            else:
                batch.to(self.local_rank)
            with torch.autograd.profiler.profile(enabled=False,
                                                record_shapes=False,
                                                use_cuda=torch.cuda.is_available()) as prof:
                with torch.cuda.amp.autocast(enabled=self.config.others.enable_amp):
                    loss, acc = self.model(batch)
                self.scaler.scale(loss).backward()
                losses.append(loss.item())
                accs.append(acc.item() * 100)
                self.accum += 1
                if self.accum == self.config.training.accum:
                    # Unscales the gradients of optimizer's assigned params in-place
                    self.scaler.unscale_(self.optimizer)

                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.clip_norm)

                    # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                    self.scaler.step(self.optimizer)

                    # Updates the scale for next iteration.
                    self.scaler.update()

                    self.scheduler.step()

                    g_norm = grad_norm(self.model)
                    self.model.zero_grad()
                    self.accum = 0
            if step % self.config.others.log_step == 0 and step != 0:
                if self.multi_gpu and dist.get_rank() == 0:
                    logging.info(f"Epoch {self.epoch}: step {step}, loss: {np.mean(losses)}, acc: {np.mean(accs)}, p_norm: {param_norm(self.model)}, g_norm: {g_norm}, lr: {get_lr(self.optimizer)}, time duration: {time.time() - self.start_time: .2f} s")
                    sys.stdout.flush()
                elif not self.multi_gpu:
                    logging.info(f"Epoch {self.epoch}: step {step}, loss: {np.mean(losses)}, acc: {np.mean(accs)}, p_norm: {param_norm(self.model)}, g_norm: {g_norm}, lr: {get_lr(self.optimizer)}, time duration: {time.time() - self.start_time: .2f} s")
                    sys.stdout.flush()
        
        if self.accum != self.config.training.accum:

            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.model.zero_grad()
        if self.multi_gpu and dist.get_rank() == 0:
            logging.info(f"Epoch {self.epoch} / {self.config.training.epoch}, loss: {np.mean(losses)}, acc: {np.mean(accs)}, "
                        f"p_norm: {param_norm(self.model)}, g_norm: {g_norm}, "
                        f"lr: {get_lr(self.optimizer)}, time duration: {time.time() - self.start_time: .2f} s")
            sys.stdout.flush()
        elif not self.multi_gpu:
            logging.info(f"Epoch {self.epoch} / {self.config.training.epoch}, loss: {np.mean(losses)}, acc: {np.mean(accs)}, "
                        f"p_norm: {param_norm(self.model)}, g_norm: {g_norm}, "
                        f"lr: {get_lr(self.optimizer)}, time duration: {time.time() - self.start_time: .2f} s")
            sys.stdout.flush()
        return np.mean(losses), np.mean(accs)

    def eval(self):

        self.model.eval()
        eval_losses = []
        eval_accs = []
        if self.multi_gpu:
            self.valid_dataloader.sampler.set_epoch(self.epoch)
        with torch.no_grad():
            for eval_idx, eval_batch in enumerate(self.valid_dataloader):
                if self.config.model.add_empty_node:
                    eval_batch = add_dense_empty_node_edge(eval_batch)
                if not self.multi_gpu:
                    eval_batch.to(self.device)
                else:
                    eval_batch.to(self.local_rank)

                eval_loss, eval_acc = self.model(eval_batch)
                eval_losses.append(eval_loss.item())
                eval_accs.append(eval_acc.item() * 100)
        if self.multi_gpu and dist.get_rank() == 0:
            logging.info(f"Evaluation (with teacher) at epoch {self.epoch}, eval loss: {np.mean(eval_losses)}, "
                        f"eval acc: {np.mean(eval_accs)}")
            sys.stdout.flush()
        elif not self.multi_gpu:
            logging.info(f"Evaluation (with teacher) at epoch {self.epoch}, eval loss: {np.mean(eval_losses)}, "
                        f"eval acc: {np.mean(eval_accs)}")
            sys.stdout.flush()
        return np.mean(eval_losses), np.mean(eval_accs)
                    
    def infer(self):

        self.model.eval()
        accs_token = []
        accs_seq = []
        if self.multi_gpu:
            self.valid_dataloader.sampler.set_epoch(self.epoch)
        with torch.no_grad():
            for eval_idx, eval_batch in enumerate(self.valid_dataloader):
                if self.config.model.add_empty_node:
                    eval_batch = add_dense_empty_node_edge(eval_batch)
                if not self.multi_gpu:
                    eval_batch.to(self.device)
                    results = self.model.infer(
                                    reaction_batch=eval_batch,
                                    batch_size=len(eval_batch.tgt_lens),
                                    beam_size=self.config.infer.beam_size,
                                    n_best=1,
                                    temperature=1.0,
                                    min_length=self.config.infer.min_length,
                                    max_length=self.config.infer.max_length
                                    )
                else:
                    eval_batch.to(self.local_rank)
                
                    results = self.model.module.infer(
                        reaction_batch=eval_batch,
                        batch_size=len(eval_batch.tgt_lens),
                        beam_size=self.config.infer.beam_size,
                        n_best=1,
                        temperature=1.0,
                        min_length=self.config.infer.min_length,
                        max_length=self.config.infer.max_length
                    )
                predictions = [t[0].cpu().numpy() for t in results["predictions"]]

                for i, prediction in enumerate(predictions):
                    tgt_length = eval_batch.tgt_lens[i].item()
                    tgt_token_ids = eval_batch.tgt_token_ids[i].cpu().numpy()[:tgt_length]

                    acc_seq = np.array_equal(tgt_token_ids, prediction)
                    while len(prediction) < tgt_length:
                        prediction = np.append(prediction, self.vocab["_PAD"])

                    acc_token = np.mean(tgt_token_ids == prediction[:tgt_length])

                    accs_token.append(acc_token)
                    accs_seq.append(acc_seq)

                    if eval_idx % self.config.infer.print_iterval == 0 and i == 0:
                        if self.multi_gpu and dist.get_rank() == 0:
                            logging.info(f"Target text: {' '.join([self.vocab_rev[idx] for idx in tgt_token_ids])}")
                            logging.info(f"Predicted text: {' '.join([self.vocab_rev[idx] for idx in prediction])}")
                            logging.info(f"acc_token: {acc_token}, acc_seq: {acc_seq}")
                        elif not self.multi_gpu:
                            logging.info(f"Target text: {' '.join([self.vocab_rev[idx] for idx in tgt_token_ids])}")
                            logging.info(f"Predicted text: {' '.join([self.vocab_rev[idx] for idx in prediction])}")
                            logging.info(f"acc_token: {acc_token}, acc_seq: {acc_seq}")
        if self.multi_gpu and dist.get_rank() == 0:
            logging.info(f"Evaluation (without teacher) at epoch {self.epoch}, "
                        f"eval acc (token): {np.mean(accs_token)}, "
                        f"eval acc (sequence): {np.mean(accs_seq)}")
            sys.stdout.flush()
        elif not self.multi_gpu:
            logging.info(f"Evaluation (without teacher) at epoch {self.epoch}, "
                        f"eval acc (token): {np.mean(accs_token)}, "
                        f"eval acc (sequence): {np.mean(accs_seq)}")
            sys.stdout.flush()
            
        return np.mean(accs_token), np.mean(accs_seq)
    
    def run(self):
        logging.info("[INFO] Start training...")
        sys.stdout.flush()
        for epoch in range(self.config.training.epoch):
            epoch_start_time = time.time()
            self.epoch = epoch + 1
            if self.multi_gpu and dist.get_rank() == 0:
                logging.info(f'============= Epoch {self.epoch} =============')
                logging.info("Training....")
            elif not self.multi_gpu:
                logging.info(f'============= Epoch {self.epoch} =============')
                logging.info("Training....")
            train_loss, train_acc = self.train()
            if self.multi_gpu and dist.get_rank() == 0:
                logging.info("Evaluating....")
            elif not self.multi_gpu:
                logging.info("Evaluating....")
            valid_loss, valid_acc = self.eval()
            if self.multi_gpu and dist.get_rank() == 0:
                logging.info("Inferencing....")
            elif not self.multi_gpu:
                logging.info("Inferencing....")

            valid_token_acc_wo_teach, valid_seq_acc_wo_teach = self.infer()
            if valid_acc > self.best_val_acc:
                if self.multi_gpu and dist.get_rank() == 0:
                    logging.info("[INFO] ~~~~ Better validation accuracy, saving model checkpoint... ~~~~")
                    self.best_val_acc = valid_acc
                    checkpoint = {'epoch': self.epoch,
                                'model_state_dict': self.model.module.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'scheduler_state_dict': self.scheduler.state_dict(),
                                'best_val_acc': self.best_val_acc
                                            }
                    if not self.save_improve:
                        torch.save(checkpoint, os.path.join(self.model_save_dir, f'valid_acc_checkpoint.pt'))
                    else:
                        
                        torch.save(checkpoint, os.path.join(self.model_save_dir, f'valid_acc_checkpoint_{self.epoch}.pt'))
                elif not self.multi_gpu:
                    logging.info("[INFO] ~~~~ Better validation accuracy, saving model checkpoint... ~~~~")
                    self.best_val_acc = valid_acc
                    checkpoint = {'epoch': self.epoch,
                                    'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                    'scheduler_state_dict': self.scheduler.state_dict(),
                                    'best_val_acc': self.best_val_acc
                                            }
                    if not self.save_improve:
                        torch.save(checkpoint, os.path.join(self.model_save_dir, f'valid_acc_checkpoint.pt'))
                    else:
                        torch.save(checkpoint, os.path.join(self.model_save_dir, f'valid_acc_checkpoint_{self.epoch}.pt'))
            
            
            if valid_seq_acc_wo_teach > self.best_seq_acc:
                
                
                if self.multi_gpu and dist.get_rank() == 0:
                    logging.info("[INFO] !!!! Better sequence accuracy without teacher, saving model checkpoint... !!!!")
                    self.best_seq_acc = valid_seq_acc_wo_teach
                    self.best_token_acc = valid_token_acc_wo_teach
                    checkpoint = {'epoch': self.epoch, 
                                'model_state_dict': self.model.module.state_dict(), 
                                'optimizer_state_dict': self.optimizer.state_dict(), 
                                'scheduler_state_dict': self.scheduler.state_dict(), 
                                'best_seq_acc': self.best_seq_acc, 
                                'best_token_acc': self.best_token_acc
                                            }
                    if not self.save_improve:
                        torch.save(checkpoint, os.path.join(self.model_save_dir, f'valid_checkpoint.pt'))
                    else:
                        torch.save(checkpoint, os.path.join(self.model_save_dir, f'valid_checkpoint_{self.epoch}.pt'))
                elif not self.multi_gpu:
                    logging.info("[INFO] !!!! Better sequence accuracy without teacher, saving model checkpoint... !!!!")
                    self.best_seq_acc = valid_seq_acc_wo_teach
                    self.best_token_acc = valid_token_acc_wo_teach
                    checkpoint = {'epoch': self.epoch, 
                                    'model_state_dict': self.model.state_dict(), 
                                    'optimizer_state_dict': self.optimizer.state_dict(), 
                                    'scheduler_state_dict': self.scheduler.state_dict(), 
                                    'best_seq_acc': self.best_seq_acc, 
                                    'best_token_acc': self.best_token_acc
                                            }
                    if not self.save_improve:
                        torch.save(checkpoint, os.path.join(self.model_save_dir, f'valid_checkpoint.pt'))
                    else:
                        torch.save(checkpoint, os.path.join(self.model_save_dir, f'valid_checkpoint_{self.epoch}.pt'))
                #if self.epoch >= self.test_eval_after:
                    #if self.multi_gpu and dist.get_rank() == 0:
                    #    logging.info("[INFO] Testing....")
                    #elif not self.multi_gpu:
                    #    logging.info("[INFO] Testing....")
                    #self.test()
            
            if self.multi_gpu and dist.get_rank() == 0:
                self.writer.add_scalar('train_loss', train_loss, self.epoch)
                self.writer.add_scalar('train_acc', train_acc, self.epoch)
                self.writer.add_scalar('valid_loss', valid_loss, self.epoch)
                self.writer.add_scalar('valid_acc', valid_acc, self.epoch)
                self.writer.add_scalar('valid_token_acc_w/o_teacher', valid_token_acc_wo_teach, self.epoch)
                self.writer.add_scalar('valid_seq_acc_w/o_teacher', valid_seq_acc_wo_teach, self.epoch)
            elif not self.multi_gpu:
                self.writer.add_scalar('train_loss', train_loss, self.epoch)
                self.writer.add_scalar('train_acc', train_acc, self.epoch)
                self.writer.add_scalar('valid_loss', valid_loss, self.epoch)
                self.writer.add_scalar('valid_acc', valid_acc, self.epoch)
                self.writer.add_scalar('valid_token_acc_w/o_teacher', valid_token_acc_wo_teach, self.epoch)
                self.writer.add_scalar('valid_seq_acc_w/o_teacher', valid_seq_acc_wo_teach, self.epoch)
            epoch_end_time = time.time()
            if self.multi_gpu and dist.get_rank() == 0:
                logging.info(f'Epoch {self.epoch} took {epoch_end_time - epoch_start_time:.2f} seconds\n')
            elif not self.multi_gpu:
                logging.info(f'Epoch {self.epoch} took {epoch_end_time - epoch_start_time:.2f} seconds\n')

        if self.multi_gpu and dist.get_rank() == 0:
            #logging.info(f'Best validation acc. so far: {self.best_val_acc}')
            self.writer.close()
        elif not self.multi_gpu:
            #logging.info(f'Best validation acc. so far: {self.best_val_acc}')
            self.writer.close()
            
    def init_optimizer(self):
        
        if hasattr(self.config.model,"pretrained_model_path") and self.config.model.pretrained_model_path and not self.config.model.pretrained_model_freeze:
            if not self.multi_gpu:
                param_lst = [{'params': filter(lambda p: p.requires_grad, self.model.encoder.parameters()), 'lr': self.config.optimizer.learning_rate * self.config.model.pretrained_lr_scaled_coef},
                            {'params': filter(lambda p: p.requires_grad, self.model.attention_encoder.parameters()), 'lr': self.config.optimizer.learning_rate},
                            {'params': filter(lambda p: p.requires_grad, self.model.decoder.parameters()), 'lr': self.config.optimizer.learning_rate},
                            {'params': filter(lambda p: p.requires_grad, self.model.output_layer.parameters()), 'lr': self.config.optimizer.learning_rate}]
            else:
                param_lst = [{'params': filter(lambda p: p.requires_grad, self.model.module.encoder.parameters()), 'lr': self.config.optimizer.learning_rate * self.config.model.pretrained_lr_scaled_coef},
                            {'params': filter(lambda p: p.requires_grad, self.model.module.attention_encoder.parameters()), 'lr': self.config.optimizer.learning_rate},
                            {'params': filter(lambda p: p.requires_grad, self.model.module.decoder.parameters()), 'lr': self.config.optimizer.learning_rate},
                            {'params': filter(lambda p: p.requires_grad, self.model.module.output_layer.parameters()), 'lr': self.config.optimizer.learning_rate}]
            
        if self.config.optimizer.optimizer.lower() == 'adamw':
            if hasattr(self.config.model,"pretrained_model_path") and self.config.model.pretrained_model_path and not self.config.model.pretrained_model_freeze:
                logging.info(f'[INFO] Use different learning rate for pretrained model')
                self.optimizer = AdamW(param_lst, 
                                    lr=self.config.optimizer.learning_rate, 
                                    weight_decay=self.config.optimizer.weight_decay,
                                    betas=(self.config.optimizer.beta1, self.config.optimizer.beta2),
                                    eps=self.config.optimizer.eps,
                                    )
            else:
                self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                    lr=self.config.optimizer.learning_rate, 
                                    weight_decay=self.config.optimizer.weight_decay,
                                    betas=(self.config.optimizer.beta1, self.config.optimizer.beta2),
                                    eps=self.config.optimizer.eps,
                                    )
        elif self.config.optimizer.optimizer.lower() == 'adam':

            if hasattr(self.config.model,"pretrained_model_path") and self.config.model.pretrained_model_path and not self.config.model.pretrained_model_freeze:
                logging.info(f'[INFO] Use different learning rate for pretrained model')
                self.optimizer = Adam(param_lst, 
                                    lr=self.config.optimizer.learning_rate, 
                                    weight_decay=self.config.optimizer.weight_decay,
                                    betas=(self.config.optimizer.beta1, self.config.optimizer.beta2),
                                    eps=self.config.optimizer.eps,
                                    )
            else:
                self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                    lr=self.config.optimizer.learning_rate, 
                                    weight_decay=self.config.optimizer.weight_decay,
                                    betas=(self.config.optimizer.beta1, self.config.optimizer.beta2),
                                    eps=self.config.optimizer.eps,
                                    )
        else:
            raise Exception(f"Unsupport optimizer: '{self.config.optimizer.optimizer.lower()}'")

    def test(self):
        # There are some bug
        self.model.eval()
        n_best = 10
        beam_size = 10
        temperature = 1.5
        all_predictions = []
        if self.multi_gpu:
            self.test_dataloader.sampler.set_epoch(self.epoch)
        with torch.no_grad():
            for batch_data in self.test_dataloader:
                if self.config.model.add_empty_node:
                    batch_data = add_dense_empty_node_edge(batch_data)
                if not self.multi_gpu:
                    batch_data.to(self.device)
                    results = self.model.infer(reaction_batch=batch_data,
                                            batch_size=len(batch_data.tgt_lens),
                                                beam_size=beam_size,
                                                n_best=n_best,
                                                temperature=temperature,
                                                min_length=self.config.infer.min_length,
                                                max_length=self.config.infer.max_length)
                else:
                    batch_data.to(self.local_rank)
                    results = self.model.module.infer(reaction_batch=batch_data,
                                            batch_size=len(batch_data.tgt_lens),
                                                beam_size=beam_size,
                                                n_best=n_best,
                                                temperature=temperature,
                                                min_length=self.config.infer.min_length,
                                                max_length=self.config.infer.max_length)
                for predictions in results["predictions"]:
                    smis = []
                    for prediction in predictions:
                        predicted_idx = prediction.detach().cpu().numpy()
                        predicted_tokens = [self.vocab_rev[idx] for idx in predicted_idx[:-1]]
                        smi = " ".join(predicted_tokens)
                        smis.append(smi)
                    smis = ",".join(smis)
                    all_predictions.append(smis)
        

        accuracies = np.zeros([len(self.ground_truth_smiles_lst), n_best], dtype=np.float32)
        for i in range(len(self.ground_truth_smiles_lst)):
            smi_tgt = self.ground_truth_smiles_lst[i]
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
            if self.multi_gpu and dist.get_rank() == 0:
                logging.info(f"Top-{i+1} Accuracy: {np.mean(self.accuracies[:, i])}")
            elif not self.multi_gpu:
                logging.info(f"Top-{i+1} Accuracy: {np.mean(self.accuracies[:, i])}")


        return accuracies

    def init_scheduler(self):

        if self.config.scheduler.type.lower() == 'noamlr':
            self.scheduler = NoamLR(self.optimizer,model_size=self.config.model.emb_dim,
                                    warmup_steps=self.config.scheduler.warmup_step)

        else:
            raise Exception(f"Unsupport scheduler: '{self.config.scheduler.type.lower()}'")