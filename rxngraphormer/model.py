import torch,math,json
from box import Box
import numpy as np
from torch import nn
from torch_geometric.nn import GlobalAttention
import torch.nn.functional as F
from onmt.modules.embeddings import Embeddings
from onmt.decoders import TransformerDecoder
from onmt.translate import BeamSearch, GNMTGlobalScorer, GreedySearch
from onmt.modules.embeddings import PositionalEncoding
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask
from .data import calc_batch_graph_distance,NUM_ATOM_TYPE,NUM_DEGRESS_TYPE,\
                   NUM_FORMCHRG_TYPE,NUM_HYBRIDTYPE,NUM_CHIRAL_TYPE,NUM_AROMATIC_NUM,\
                   NUM_VALENCE_TYPE,NUM_Hs_TYPE,NUM_RS_TPYE
from .utils import pad_feat,update_batch_idx,get_sin_encodings,update_dict_key
from .layer import GCNConv,GINConv,GATConv,MultiHeadAttention,FeedForward
                   

class RegressorLayer(nn.Module):
    def __init__(self,hidden_size,output_size,layer_num=3,batch_norm=False,act_func='relu'):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(layer_num-1)])
        self.batch_norm = batch_norm
        if act_func == 'relu':
            self.act_func = F.relu
        elif act_func == 'tanh':
            self.act_func = F.tanh
        if self.batch_norm:
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_size) for i in range(layer_num-1)])
        self.projection = nn.Linear(hidden_size, output_size,bias=False)
    def forward(self,x):
        if self.batch_norm:
            for layer,batch in zip(self.layers,self.batch_norms):
                x = self.act_func(batch(layer(x)))
        else:
            for layer in self.layers:
                x = self.act_func(layer(x))
        return self.projection(x)

class ClassifierLayer(nn.Module):
    def __init__(self,hidden_size,output_size=2,layer_num=3,batch_norm=True,act_func='relu'):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(layer_num-1)])
        self.batch_norm = batch_norm
        if act_func == 'relu':
            self.act_func = F.relu
        elif act_func == 'tanh':
            self.act_func = F.tanh
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_size) for i in range(layer_num-1)])
        self.projection = nn.Linear(hidden_size, output_size,bias=False)
        
    def forward(self,x):
        if self.batch_norm:
            for layer,batch in zip(self.layers,self.batch_norms):
                x = self.act_func(batch(layer(x)))
        else:
            for layer in self.layers:
                x = self.act_func(layer(x))
        
        return F.softmax(self.projection(x), dim=-1)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size,intermediate_size,num_heads,hidden_dropout_prob):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.attention = MultiHeadAttention(hidden_size,num_heads)
        self.feed_forward = FeedForward(hidden_size,intermediate_size,hidden_dropout_prob)

    def forward(self, x, mask=None):
        # Apply layer normalization and then copy input into query, key, value
        hidden_state = self.layer_norm_1(x)
        # Apply attention with a skip connection
        x = x + self.attention(hidden_state, hidden_state, hidden_state, mask=mask)
        # Apply feed-forward layer with a skip connection
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x    

class TransformerEncoder(nn.Module):
    def __init__(self, num_layer,hidden_size,intermediate_size,num_heads,hidden_dropout_prob):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(hidden_size=hidden_size,intermediate_size=intermediate_size,
                                    num_heads=num_heads,hidden_dropout_prob=hidden_dropout_prob) for _ in range(num_layer)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class EXTFeatEncoder(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,hidden_layer_num=3,batch_norm=True):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(hidden_layer_num)])
        self.projection = nn.Linear(hidden_size, output_size)
        
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.input_batch_norm = nn.BatchNorm1d(hidden_size)
            self.hidden_batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_size) for i in range(hidden_layer_num)])
            self.proj_batch_norm = nn.BatchNorm1d(output_size)
        
    
    def forward(self,x):
        if self.batch_norm:
            x = F.relu(self.input_batch_norm(self.input_layer(x)))

            for layer,batch in zip(self.hidden_layers,self.hidden_batch_norms):
                x = F.relu(batch(layer(x)))
            return self.proj_batch_norm(self.projection(x))
        else:
            x = F.relu(self.input_layer(x))
            for layer in self.hidden_layers:
                x = F.relu(layer(x))
            return self.projection(x)

class RXNGRegressor(torch.nn.Module):
    def __init__(self,gnum_layer,tnum_layer,onum_layer,emb_dim,JK="last",output_size=1,drop_ratio=0.0,
                 num_heads=4,gnn_type="gcn",bond_feat_red="mean",gnn_aggr='add',node_readout='sum',
                 trans_readout='mean',graph_pooling='attention',attn_drop_ratio=0.0,encoder_filter_size=2048,
                 rel_pos_buckets=11,rel_pos="emb_only",pretrained_encoder=None,pretrained_rct_encoder=None,pretrained_pdt_encoder=None,
                 output_norm=False,split_process=False,use_mid_inf=False,interaction=False,interaction_layer_num=3,pretrained_mid_encoder=None,mid_iteract_method="attention",
                 split_merge_method="all",output_act_func='relu',rct_batch_norm=True,pdt_batch_norm=True,mid_batch_norm=True,mid_layer_num=1):
        super().__init__()
        self.emb_dim = emb_dim
        self.trans_readout = trans_readout
        self.split_process = split_process
        self.use_mid_inf = use_mid_inf
        self.interaction = interaction                      
        self.interaction_layer_num = interaction_layer_num  
        self.split_merge_method = split_merge_method
        self.mid_iteract_method = mid_iteract_method
        self.output_act_func = output_act_func
        self.rct_batch_norm = rct_batch_norm
        self.pdt_batch_norm = pdt_batch_norm
        self.mid_batch_norm = mid_batch_norm
        self.mid_layer_num = mid_layer_num
        assert self.split_merge_method in ["only_diff","all","rct_pdt"], "split_merge_method must be one of ['only_diff','all','rct_pdt']"
        assert self.mid_iteract_method in ["fc", "1dconv", "attention"], "mid_merge_method must be one of ['fc','1dconv','attention']"
        if not self.split_process:
            ## not be used
            if pretrained_encoder is None:
                self.encoder = RXNGEncoder(gnum_layer,tnum_layer,emb_dim,JK=JK,drop_ratio=drop_ratio,attn_drop_ratio=attn_drop_ratio,
                    num_heads=num_heads,gnn_type=gnn_type,bond_feat_red=bond_feat_red,gnn_aggr=gnn_aggr,node_readout=node_readout,
                    graph_pooling=graph_pooling,encoder_filter_size=encoder_filter_size,rel_pos_buckets=rel_pos_buckets,enc_pos_encoding=None,
                    rel_pos=rel_pos,task="retrosynthesis")
            else:
                self.encoder = pretrained_encoder

            self.decoder = RegressorLayer(hidden_size=self.emb_dim,output_size=output_size,layer_num=onum_layer,
                                        batch_norm=output_norm,act_func=self.output_act_func)
        else:
            if pretrained_rct_encoder is None:
                self.rct_encoder = RXNGEncoder(gnum_layer,tnum_layer,emb_dim,JK=JK,drop_ratio=drop_ratio,attn_drop_ratio=attn_drop_ratio,
                                                num_heads=num_heads,gnn_type=gnn_type,bond_feat_red=bond_feat_red,gnn_aggr=gnn_aggr,node_readout=node_readout,
                                                graph_pooling=graph_pooling,encoder_filter_size=encoder_filter_size,rel_pos_buckets=rel_pos_buckets,enc_pos_encoding=None,
                                                rel_pos=rel_pos,task="retrosynthesis")
            else:
                self.rct_encoder = pretrained_rct_encoder

            if pretrained_pdt_encoder is None:
                self.pdt_encoder = RXNGEncoder(gnum_layer,tnum_layer,emb_dim,JK=JK,drop_ratio=drop_ratio,attn_drop_ratio=attn_drop_ratio,
                                                num_heads=num_heads,gnn_type=gnn_type,bond_feat_red=bond_feat_red,gnn_aggr=gnn_aggr,node_readout=node_readout,
                                                graph_pooling=graph_pooling,encoder_filter_size=encoder_filter_size,rel_pos_buckets=rel_pos_buckets,enc_pos_encoding=None,
                                                rel_pos=rel_pos,task="retrosynthesis")
            else:
                self.pdt_encoder = pretrained_pdt_encoder

            if self.use_mid_inf:
                if self.split_merge_method == "only_diff":
                    mid_emb_dim = 1 * emb_dim
                elif self.split_merge_method == "all":
                    mid_emb_dim = 3 * emb_dim
                elif self.split_merge_method == "rct_pdt":
                    mid_emb_dim = 2 * emb_dim
                else:
                    raise ValueError(f"Unknown split_merge_method: {self.split_merge_method}")
                
                if pretrained_mid_encoder is None:
                    self.mid_encoder = RXNGEncoder(gnum_layer,tnum_layer,emb_dim=mid_emb_dim,JK=JK,drop_ratio=drop_ratio,attn_drop_ratio=attn_drop_ratio,
                                                    num_heads=num_heads,gnn_type=gnn_type,bond_feat_red=bond_feat_red,gnn_aggr=gnn_aggr,node_readout=node_readout,
                                                    graph_pooling=graph_pooling,encoder_filter_size=encoder_filter_size,rel_pos_buckets=rel_pos_buckets,enc_pos_encoding=None,
                                                    rel_pos=rel_pos,task="retrosynthesis")
                else:
                    self.mid_encoder = pretrained_mid_encoder

                if self.mid_iteract_method == "fc":
                    layers = [nn.Linear(2 * mid_emb_dim, mid_emb_dim), nn.ReLU()]
                    for i in range(self.mid_layer_num-1):
                        layers.append(nn.Linear(mid_emb_dim, mid_emb_dim))
                        layers.append(nn.ReLU())
                    self.mid_iteract = nn.Sequential(*layers)
                elif self.mid_iteract_method == "1dconv":
                    self.mid_iteract = nn.Conv1d(2, 1, kernel_size=1)
                elif self.mid_iteract_method == "attention":
                    self.mid_iteract = nn.MultiheadAttention(mid_emb_dim, num_heads=1)
                else:
                    raise ValueError(f"Unknown mid_iteract_method: {self.mid_iteract_method}")

                self.mid_decoder = RegressorLayer(hidden_size=mid_emb_dim,output_size=output_size,layer_num=onum_layer,
                                                batch_norm=output_norm,act_func=self.output_act_func) ## TODO
                #raise NotImplemented(f"Not implemented pretrain ability!")
            if self.split_merge_method == "only_diff":
                dec_emb_dim = 1 * emb_dim
            elif self.split_merge_method == "all":
                dec_emb_dim = 3 * emb_dim
            elif self.split_merge_method == "rct_pdt":
                dec_emb_dim = 2 * emb_dim


            self.decoder = RegressorLayer(hidden_size=dec_emb_dim,output_size=output_size,layer_num=onum_layer,
                                            batch_norm=output_norm,act_func=self.output_act_func)

    def forward(self,data):
        if not self.split_process:
            padded_memory_bank,batch,memory_lengths = self.encoder(data)
            rxn_transf_emb = padded_memory_bank.transpose(0,1)
            if self.trans_readout == 'mean':
                rxn_transf_emb_merg = rxn_transf_emb.mean(dim=1) #### super para

            output = self.decoder(rxn_transf_emb_merg)

        else:
            if not self.use_mid_inf:
                rct_data,pdt_data = data
                rct_padded_memory_bank,rct_batch,rct_memory_lengths = self.rct_encoder(rct_data)
                pdt_padded_memory_bank,pdt_batch,pdt_memory_lengths = self.pdt_encoder(pdt_data)
                rct_rxn_transf_emb = rct_padded_memory_bank.transpose(0,1)
                pdt_rxn_transf_emb = pdt_padded_memory_bank.transpose(0,1)
                if self.trans_readout == 'mean':
                    rct_rxn_transf_emb_merg = rct_rxn_transf_emb.mean(dim=1) #### super para  shape: (batch_size, emb_dim) eg. 32, 256
                    pdt_rxn_transf_emb_merg = pdt_rxn_transf_emb.mean(dim=1) #### super para  shape: (batch_size, emb_dim) eg. 32, 256

                diff_emb = torch.abs(rct_rxn_transf_emb_merg - pdt_rxn_transf_emb_merg)
                cat_emb = torch.cat([rct_rxn_transf_emb_merg,pdt_rxn_transf_emb_merg,diff_emb],dim=-1)
                rct_pdt_cat_emb = torch.cat([rct_rxn_transf_emb_merg,pdt_rxn_transf_emb_merg],dim=-1)
                if self.split_merge_method == "only_diff":
                    output = self.decoder(diff_emb)
                elif self.split_merge_method == "all":
                    output = self.decoder(cat_emb)
                elif self.split_merge_method == "rct_pdt":
                    output = self.decoder(rct_pdt_cat_emb)

            else:
                rct_data,pdt_data,mid_data = data
                rct_padded_memory_bank,rct_batch,rct_memory_lengths = self.rct_encoder(rct_data)
                pdt_padded_memory_bank,pdt_batch,pdt_memory_lengths = self.pdt_encoder(pdt_data)
                mid_padded_memory_bank,mid_batch,mid_memory_lengths = self.mid_encoder(mid_data)
                rct_rxn_transf_emb = rct_padded_memory_bank.transpose(0,1)
                pdt_rxn_transf_emb = pdt_padded_memory_bank.transpose(0,1)
                mid_rxn_transf_emb = mid_padded_memory_bank.transpose(0,1)
                if self.trans_readout == 'mean':
                    rct_rxn_transf_emb_merg = rct_rxn_transf_emb.mean(dim=1)  #### super para  shape: (batch_size, emb_dim) eg. 32, 256
                    pdt_rxn_transf_emb_merg = pdt_rxn_transf_emb.mean(dim=1)  #### super para  shape: (batch_size, emb_dim) eg. 32, 256
                    mid_rxn_transf_emb_merg = mid_rxn_transf_emb.mean(dim=1)  #### super para  shape: (batch_size, emb_dim) eg. 32, 256
                

                diff_emb = torch.abs(rct_rxn_transf_emb_merg - pdt_rxn_transf_emb_merg)                     ## shape: (batch_size, emb_dim) eg. 32, 256
                cat_emb = torch.cat([rct_rxn_transf_emb_merg,pdt_rxn_transf_emb_merg,diff_emb],dim=-1)      ## shape: (batch_size, 3 * emb_dim) eg. 32, 3 * 256
                rct_pdt_cat_emb = torch.cat([rct_rxn_transf_emb_merg,pdt_rxn_transf_emb_merg],dim=-1)       ## shape: (batch_size, 2 * emb_dim) eg. 32, 2 * 256
                if self.split_merge_method == "only_diff":
                    stack_emb = torch.stack([diff_emb,mid_rxn_transf_emb_merg],dim=1)      ## shape: (batch_size, 2, emb_dim) eg. 32, 2, 256
                    #output = self.decoder(diff_emb)
                elif self.split_merge_method == "all":
                    stack_emb = torch.stack([cat_emb,mid_rxn_transf_emb_merg],dim=1)      ## shape: (batch_size, 2, 3 * emb_dim) eg. 32, 2, 3 * 256
                    #output = self.decoder(cat_emb)
                elif self.split_merge_method == "rct_pdt":
                    stack_emb = torch.stack([rct_pdt_cat_emb,mid_rxn_transf_emb_merg],dim=1)      ## shape: (batch_size, 2, 2 * emb_dim) eg. 32, 2, 2 * 256
                    #output = self.decoder(rct_pdt_cat_emb)

                if self.mid_iteract_method == "fc":                        
                    stack_emb_flat = stack_emb.view(stack_emb.shape[0],-1)           ## batch_size, 2, mid_emb_dim -> batch_size, 2 * mid_emb_dim
                    stack_emb_output = self.mid_iteract(stack_emb_flat)  ## fc = nn.Linear(2 * mid_emb_dim , mid_emb_dim) , (batch_size, 2 * mid_emb_dim) -> (batch_size, mid_emb_dim)
                elif self.mid_iteract_method == "1dconv":
                    stack_emb_output = self.mid_iteract(stack_emb).squeeze(1)       ##   nn.Conv1d(2, 1, kernel_size=1), (batch_size, 2, mid_emb_dim) -> (batch_size, 1, mid_emb_dim) -> (batch_size, mid_emb_dim)
                elif self.mid_iteract_method == "attention":
                    stack_emb_perm = stack_emb.permute(1,0,2)                      ##  (batch_size, 2, mid_emb_dim) -> (2, batch_size, mid_emb_dim)
                    stack_emb_perm_att,_ = self.mid_iteract(stack_emb_perm,stack_emb_perm,stack_emb_perm) ##   nn.MultiheadAttention(mid_emb_dim, num_heads=1)
                    stack_emb_perm_att = stack_emb_perm_att.permute(1, 0, 2)
                    stack_emb_output = stack_emb_perm_att.mean(dim=1)              ##  (batch_size, 2, mid_emb_dim) -> (batch_size, mid_emb_dim)

                output = self.mid_decoder(stack_emb_output)
                    

        return output
    
class RXNGClassifier(torch.nn.Module):
    def __init__(self,gnum_layer,tnum_layer,onum_layer,emb_dim=256,JK="last",output_size=2,drop_ratio=0.0,
                 num_heads=4,gnn_type="gcn",bond_feat_red="mean",gnn_aggr='add',node_readout='sum',
                 trans_readout='mean',graph_pooling='attention',attn_drop_ratio=0.0,encoder_filter_size=2048,
                 rel_pos_buckets=11,rel_pos="emb_only",split_process=False,split_merge_method="all",output_act_func='relu'):
        super().__init__()
        self.emb_dim = emb_dim
        self.trans_readout = trans_readout
        self.split_process = split_process
        self.split_merge_method = split_merge_method.lower()
        assert self.split_merge_method in ["only_diff","all","rct_pdt"]
        self.output_act_func = output_act_func
        if not self.split_process:
            ## This option will be removed in the future
            self.encoder = RXNGEncoder(gnum_layer,tnum_layer,emb_dim,JK=JK,drop_ratio=drop_ratio,attn_drop_ratio=attn_drop_ratio,
                    num_heads=num_heads,gnn_type=gnn_type,bond_feat_red=bond_feat_red,gnn_aggr=gnn_aggr,node_readout=node_readout,
                    graph_pooling=graph_pooling,encoder_filter_size=encoder_filter_size,rel_pos_buckets=rel_pos_buckets,enc_pos_encoding=None,
                    rel_pos=rel_pos,task="retrosynthesis")
            self.decoder = ClassifierLayer(hidden_size=self.emb_dim,output_size=output_size,layer_num=onum_layer,act_func=self.output_act_func)
        else:
            self.rct_encoder = RXNGEncoder(gnum_layer,tnum_layer,emb_dim,JK=JK,drop_ratio=drop_ratio,attn_drop_ratio=attn_drop_ratio,
                    num_heads=num_heads,gnn_type=gnn_type,bond_feat_red=bond_feat_red,gnn_aggr=gnn_aggr,node_readout=node_readout,
                    graph_pooling=graph_pooling,encoder_filter_size=encoder_filter_size,rel_pos_buckets=rel_pos_buckets,enc_pos_encoding=None,
                    rel_pos=rel_pos,task="retrosynthesis")
            self.pdt_encoder = RXNGEncoder(gnum_layer,tnum_layer,emb_dim,JK=JK,drop_ratio=drop_ratio,attn_drop_ratio=attn_drop_ratio,
                    num_heads=num_heads,gnn_type=gnn_type,bond_feat_red=bond_feat_red,gnn_aggr=gnn_aggr,node_readout=node_readout,
                    graph_pooling=graph_pooling,encoder_filter_size=encoder_filter_size,rel_pos_buckets=rel_pos_buckets,enc_pos_encoding=None,
                    rel_pos=rel_pos,task="retrosynthesis")

            if self.split_merge_method == "all":
                self.decoder = ClassifierLayer(hidden_size=self.emb_dim*3,output_size=output_size,layer_num=onum_layer,act_func=self.output_act_func) # *3 -> rct, pdt, diff
            elif self.split_merge_method == "only_diff":
                self.decoder = ClassifierLayer(hidden_size=self.emb_dim*1,output_size=output_size,layer_num=onum_layer,act_func=self.output_act_func) # *1 -> diff
            elif self.split_merge_method == "rct_pdt":
                self.decoder = ClassifierLayer(hidden_size=self.emb_dim*2,output_size=output_size,layer_num=onum_layer,act_func=self.output_act_func) # *2 -> rct, pdt
        
    def forward(self,data):
        if not self.split_process:
            padded_memory_bank,batch,memory_lengths = self.encoder(data)
            rxn_transf_emb = padded_memory_bank.transpose(0,1)
            if self.trans_readout == 'mean':
                rxn_transf_emb_merg = rxn_transf_emb.mean(dim=1) #### super para
            output = self.decoder(rxn_transf_emb_merg)
        else:
            rct_data,pdt_data = data
            rct_padded_memory_bank,rct_batch,rct_memory_lengths = self.rct_encoder(rct_data)
            pdt_padded_memory_bank,pdt_batch,pdt_memory_lengths = self.pdt_encoder(pdt_data)
            rct_rxn_transf_emb = rct_padded_memory_bank.transpose(0,1)
            pdt_rxn_transf_emb = pdt_padded_memory_bank.transpose(0,1)
            if self.trans_readout == 'mean':
                rct_rxn_transf_emb_merg = rct_rxn_transf_emb.mean(dim=1) #### super para
                pdt_rxn_transf_emb_merg = pdt_rxn_transf_emb.mean(dim=1) #### super para
            
            diff_emb = torch.abs(rct_rxn_transf_emb_merg - pdt_rxn_transf_emb_merg)
            if self.split_merge_method == "all":
                cat_emb = torch.cat([rct_rxn_transf_emb_merg,pdt_rxn_transf_emb_merg,diff_emb],dim=-1)
                output = self.decoder(cat_emb)
            elif self.split_merge_method == "only_diff":
                output = self.decoder(diff_emb)
            elif self.split_merge_method == "rct_pdt":
                rct_pdt_cat_emb = torch.cat([rct_rxn_transf_emb_merg,pdt_rxn_transf_emb_merg],dim=-1)
                output = self.decoder(rct_pdt_cat_emb)
        return output

class RXNGraphEncoder(nn.Module):
    def __init__(self, gnum_layer, emb_dim, gnn_aggr="add", bond_feat_red="mean", gnn_type='gcn', JK="last", drop_ratio=0.0, node_readout="sum"):
        super().__init__()
        self.gnum_layer = gnum_layer
        self.emb_dim = emb_dim
        self.gnn_aggr = gnn_aggr
        self.gnn_type = gnn_type
        self.JK = JK
        self.drop_ratio = drop_ratio
        self.node_readout = node_readout
        assert self.gnum_layer >= 2, "Number of RXNGraphEncoder layers must be greater than 1."

        self.x_embedding1 = torch.nn.Embedding(NUM_ATOM_TYPE, self.emb_dim)     ## atom type
        self.x_embedding2 = torch.nn.Embedding(NUM_DEGRESS_TYPE, self.emb_dim)  ## atom degree
        self.x_embedding3 = torch.nn.Embedding(NUM_FORMCHRG_TYPE, self.emb_dim) ## formal charge
        self.x_embedding4 = torch.nn.Embedding(NUM_HYBRIDTYPE, self.emb_dim)    ## hybrid type
        self.x_embedding5 = torch.nn.Embedding(NUM_CHIRAL_TYPE, self.emb_dim)   ## chiral type
        self.x_embedding6 = torch.nn.Embedding(NUM_AROMATIC_NUM, self.emb_dim)  ## aromatic or not
        self.x_embedding7 = torch.nn.Embedding(NUM_VALENCE_TYPE, self.emb_dim)  ## valence
        self.x_embedding8 = torch.nn.Embedding(NUM_Hs_TYPE, self.emb_dim)       ## number of Hs
        self.x_embedding9 = torch.nn.Embedding(NUM_RS_TPYE, self.emb_dim)       ## R or S

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding5.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding6.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding7.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding8.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding9.weight.data)
        
        self.x_emedding_lst = [self.x_embedding1,self.x_embedding2,self.x_embedding3,
                               self.x_embedding4,self.x_embedding5,self.x_embedding6,
                               self.x_embedding7,self.x_embedding8,self.x_embedding9]

        ## List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(self.gnum_layer):
            if self.gnn_type.lower() == 'gcn':
                self.gnns.append(GCNConv(self.emb_dim,aggr=self.gnn_aggr,bond_feat_red=bond_feat_red))
            elif self.gnn_type.lower() == 'gin':
                self.gnns.append(GINConv(self.emb_dim,self.emb_dim, aggr=self.gnn_aggr,bond_feat_red=bond_feat_red))
            elif self.gnn_type.lower() == 'gat':
                self.gnns.append(GATConv(self.emb_dim, aggr=self.gnn_aggr,bond_feat_red=bond_feat_red))
            else:
                raise ValueError(f"Unknown GNN type: {self.gnn_type.lower()}")
                
        ## List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.gnum_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(self.emb_dim))
    def forward(self, x, mol_index, edge_index, edge_attr):
        mol_index,batch = update_batch_idx(mol_index,device=x.device)
        mol_index = mol_index.to(x.device)
        batch = batch.to(x.device)
        x_emb_lst = []
        for i in range(x.shape[1]):
            _x_emb = self.x_emedding_lst[i](x[:,i])
            x_emb_lst.append(_x_emb)
        if self.node_readout == 'sum':
            x_emb = torch.stack(x_emb_lst).sum(dim=0)
        elif self.node_readout == 'mean':
            x_emb = torch.stack(x_emb_lst).mean(dim=0)
        h_list = [x_emb]
        for layer in range(self.gnum_layer):

            h = self.gnns[layer](h_list[layer],edge_index=edge_index,edge_attr=edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.gnum_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=True)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=True)
            h_list.append(h)
        if self.JK == 'last':
            node_representation = h_list[-1]
        elif self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "max":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)
        elif self.JK == "mean":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.mean(torch.cat(h_list, dim = 0), dim = 0)
        elif self.JK == 'last+first':
            node_representation = h_list[-1] + h_list[0]
        else:
            raise NotImplementedError
        
        return node_representation, mol_index, batch

class RXNGEncoder(torch.nn.Module):
    def __init__(self,gnum_layer,tnum_layer,emb_dim,JK="last",drop_ratio=0.0,attn_drop_ratio=0.0,
                 num_heads=4,gnn_type="gcn",bond_feat_red="mean",gnn_aggr='add',node_readout='sum',
                 graph_pooling='attention',encoder_filter_size=2048,rel_pos_buckets=11,enc_pos_encoding=None,
                 rel_pos="emb_only",task="retrosynthesis",add_empty_node=False):
        super().__init__()
        self.rxn_graph_encoder = RXNGraphEncoder(gnum_layer=gnum_layer, emb_dim=emb_dim, gnn_aggr=gnn_aggr, 
                                                 bond_feat_red=bond_feat_red, gnn_type=gnn_type, JK=JK, 
                                                 drop_ratio=drop_ratio, node_readout=node_readout)
        
        #self.gnum_layer = gnum_layer
        self.tnum_layer = tnum_layer
        self.drop_ratio = drop_ratio
        self.attn_drop_ratio = attn_drop_ratio
        # self.JK = JK
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        # self.node_readout = node_readout
        self.graph_pooling = graph_pooling
        self.encoder_filter_size = encoder_filter_size  ## attention_xl
        self.rel_pos_buckets = rel_pos_buckets
        self.enc_pos_encoding = enc_pos_encoding
        self.rel_pos = rel_pos
        self.task = task
        # self.gnn_type = gnn_type
        self.add_empty_node = add_empty_node


        
        if self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(self.emb_dim, 1))
            
        elif self.graph_pooling == "attentionxl":
            self.pool = AttnEncoderXL(self.tnum_layer, d_model=self.emb_dim, 
                                      heads=self.num_heads, d_ff=self.encoder_filter_size, 
                                      dropout=self.drop_ratio, attention_dropout=self.attn_drop_ratio, 
                                      rel_pos_buckets=self.rel_pos_buckets, enc_pos_encoding=self.enc_pos_encoding,
                                      rel_pos=self.rel_pos)
        
        self.t_encoder = TransformerEncoder(num_layer=self.tnum_layer,hidden_size=self.emb_dim,
                                           intermediate_size=self.emb_dim,num_heads=num_heads,hidden_dropout_prob=self.drop_ratio)
    def forward(self,data):
        x = data.x
        mol_index = data.mol_index
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        node_representation, mol_index,batch = self.rxn_graph_encoder(x, mol_index, edge_index, edge_attr)
        
        if self.graph_pooling == "attention":
            memory_lengths = torch.bincount(batch).long().to(device=node_representation.device)
            rxn_representation = self.pool(node_representation,mol_index)  ## node_representation is equal to hatom
            padded_feat = pad_feat(rxn_representation,batch,self.emb_dim)
            rxn_transf_emb = self.t_encoder(padded_feat)
            padded_memory_bank = rxn_transf_emb.transpose(1,0)

        elif self.graph_pooling == "attentionxl":  ## TODO name it 
            memory_lengths = torch.bincount(data.batch).long().to(device=node_representation.device)
            assert sum(memory_lengths) == node_representation.size(0), \
                f"Memory lengths calculation error, encoder output: {node_representation.size(0)}, memory_lengths: {memory_lengths}"
            ## add an empty node in original paper
            memory_bank = torch.split(node_representation, memory_lengths.cpu().tolist(), dim=0)   # [n_atoms, h] => 1+b tup of (t, h)
            padded_memory_bank = []
            max_length = max(memory_lengths)
            for length, h in zip(memory_lengths, memory_bank):
                m = nn.ZeroPad2d((0, 0, 0, max_length - length))
                padded_memory_bank.append(m(h))
            
            padded_memory_bank = torch.stack(padded_memory_bank, dim=1)     # list of b (max_t, h) => [max_t, b, h]
            distances = calc_batch_graph_distance(batch=data.batch,
                                                edge_index=data.edge_index,
                                                task=self.task)
            padded_memory_bank = self.pool(
                padded_memory_bank,
                memory_lengths,
                distances
            )

        else:
            raise NotImplementedError

        return padded_memory_bank,batch,memory_lengths

class SeqDecoder(nn.Module):
    def __init__(self, word_vec_size, vocab,decoder_num_layers,decoder_hidden_size,
                 decoder_attn_heads,decoder_filter_size,max_relative_positions,
                 pad_token="_PAD",sos_token="_SOS",eos_token="_EOS",
                 copy_attn=False,self_attn_type="scaled-dot",position_encoding=True,
                 aan_useffn=False,full_context_alignment=False,alignment_layer=-3,
                 alignment_heads=0,dropout=0.0):
        super().__init__()
        self.vocab = vocab
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.word_vocab_size = len(self.vocab)
        self.word_padding_idx = self.vocab[self.pad_token]
        self.decoder_embeddings = Embeddings(
                word_vec_size=word_vec_size,
                word_vocab_size=self.word_vocab_size,
                word_padding_idx=self.word_padding_idx,
                position_encoding=position_encoding,
                dropout=dropout)
        self.tdecoder = TransformerDecoder(
            num_layers=decoder_num_layers,
            d_model=decoder_hidden_size,
            heads=decoder_attn_heads,
            d_ff=decoder_filter_size,
            copy_attn=copy_attn,
            self_attn_type=self_attn_type,
            dropout=dropout,
            attention_dropout=dropout,
            embeddings=self.decoder_embeddings,
            max_relative_positions=max_relative_positions,
            aan_useffn=aan_useffn,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads)
        self.output_layer = nn.Linear(decoder_hidden_size, self.word_vocab_size, bias=True)
    
    def forward(self,padded_memory_bank, tgt_token_ids, memory_lengths):
        # tgt_token_ids is shorten version  tgt_token_ids = data.tgt_token_ids[:,:data.tgt_lens.max()]
        #memory_lengths = torch.bincount(batch).long().to(device=rxn_emb_enc.device)
        # padded_memory_bank = rxn_emb_enc.transpose(1,0)
        # print(f"padded_memory_bank.shape: {padded_memory_bank.shape}, memory_lengths.shape: {memory_lengths.shape}")
        self.tdecoder.state["src"] = torch.zeros(max(memory_lengths))
        
        dec_in = tgt_token_ids[:, :-1] ## pop last, insert SOS for decoder input
        m = nn.ConstantPad1d((1, 0), self.vocab[self.sos_token])
        dec_in = m(dec_in)
        dec_in = dec_in.transpose(0, 1).unsqueeze(-1)
        #print("!!!!!!!",dec_in.shape,padded_memory_bank.shape,memory_lengths.shape)

        dec_outs, _ = self.tdecoder(tgt=dec_in,
                                    memory_bank=padded_memory_bank,
                                    memory_lengths=memory_lengths)
        dec_outs = self.output_layer(dec_outs)
        dec_outs = dec_outs.permute(1, 2, 0)
        predictions = torch.argmax(dec_outs, dim=1)
        return dec_outs, predictions

class G2STransformer(nn.Module):
    def __init__(self,tencoder,tdecoder,vocab):
        super().__init__()
        self.tdecoder = tdecoder  ## SeqDecoder
        self.tencoder = tencoder  ## RXNGEncoder
        self.vocab = vocab
    def forward(self,data):
        padded_memory_bank,batch,memory_lengths = self.tencoder(data)
        """
        print(padded_memory_bank.shape,
              data.tgt_token_ids.shape,
              data.tgt_token_ids[:data.tgt_lens.max()].shape, old version data.tgt_token_ids[:data.tgt_lens.max()] instead of data.tgt_token_ids
              memory_lengths.shape)
        """
        dec_outs, predictions = self.tdecoder(padded_memory_bank,data.tgt_token_ids,memory_lengths)
        return dec_outs,predictions
    def infer(self,data,batch_size,beam_size,n_best=10,temperature=1.0,min_length=0,max_length=512):
        padded_memory_bank,batch,memory_lengths = self.tencoder(data)
        #memory_lengths = torch.bincount(batch).long().to(device=rxn_emb_enc.device)
        # padded_memory_bank = rxn_emb_enc.transpose(1,0)
        self.tdecoder.tdecoder.state["src"] = torch.zeros(max(memory_lengths))
        if beam_size == 1:
            decode_strategy = GreedySearch(pad=self.vocab["_PAD"],
                                            bos=self.vocab["_SOS"],
                                            eos=self.vocab["_EOS"],
                                            batch_size=batch_size,
                                            min_length=min_length,
                                            max_length=max_length,
                                            block_ngram_repeat=0,
                                            exclusion_tokens=set(),
                                            return_attention=False,
                                            sampling_temp=0.0,
                                            keep_topk=1)
        else:
            global_scorer = GNMTGlobalScorer(alpha=0.0,beta=0.0,length_penalty="none",coverage_penalty="none")
            decode_strategy = BeamSearch(
                beam_size=beam_size,
                batch_size=batch_size,
                pad=self.vocab["_PAD"],
                bos=self.vocab["_SOS"],
                eos=self.vocab["_EOS"],
                n_best=n_best,
                global_scorer=global_scorer,
                min_length=min_length,
                max_length=max_length,
                return_attention=False,
                block_ngram_repeat=0,
                exclusion_tokens=set(),
                stepwise_penalty=None,
                ratio=0.0)

        #padded_memory_bank, memory_lengths = self.encode_and_reshape(reaction_batch=reaction_batch)
        # adapted from onmt.translate.translator
        results = {
            "predictions": None,
            "scores": None,
            "attention": None}

        # (2) prep decode_strategy. Possibly repeat src objects.
        src_map = None
        target_prefix = None
        fn_map_state, memory_bank, memory_lengths, src_map = decode_strategy.initialize(
            memory_bank=padded_memory_bank,
            src_lengths=memory_lengths,
            src_map=src_map,
            target_prefix=target_prefix)

        # (3) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            decoder_input = decode_strategy.current_predictions.view(1, -1, 1)

            dec_out, dec_attn = self.tdecoder.tdecoder(tgt=decoder_input,
                                            memory_bank=memory_bank,
                                            memory_lengths=memory_lengths,
                                            step=step)

            if "std" in dec_attn:
                attn = dec_attn["std"]
            else:
                attn = None

            dec_out = self.tdecoder.output_layer(dec_out)            # [t, b, h] => [t, b, v]
            dec_out = dec_out / temperature
            dec_out = dec_out.squeeze(0)                    # [t, b, v] => [b, v]
            log_probs = F.log_softmax(dec_out, dim=-1)

            # log_probs = self.model.generator(dec_out.squeeze(0))

            decode_strategy.advance(log_probs, attn)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices

            if any_finished:
                # Reorder states.
                if isinstance(memory_bank, tuple):
                    memory_bank = tuple(x.index_select(1, select_indices)
                                        for x in memory_bank)
                else:
                    memory_bank = memory_bank.index_select(1, select_indices)

                memory_lengths = memory_lengths.index_select(0, select_indices)

                if src_map is not None:
                    src_map = src_map.index_select(1, select_indices)

            if any_finished:
                self.map_state(
                    lambda state, dim: state.index_select(dim, select_indices))

        results["scores"] = decode_strategy.scores
        results["predictions"] = decode_strategy.predictions
        results["attention"] = decode_strategy.attention
        results["alignment"] = [[] for _ in range(batch_size)]

        return results

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        if self.tdecoder.tdecoder.state["cache"] is not None:
            _recursive_map(self.tdecoder.tdecoder.state["cache"])
    
class MultiHeadedRelAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout, rel_pos_buckets, u, v, rel_pos="emb_only"):
        """
        rel_pos : "emb_only" or "enc_only"
        """
        super().__init__()
        assert model_dim % head_count == 0, "model_dim must be divisible by head_count (ERROR from MultiHeadedRelAttention)"
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim, model_dim)
        self.linear_values = nn.Linear(model_dim, model_dim)
        self.linear_query = nn.Linear(model_dim, model_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

        self.rel_pos_buckets = rel_pos_buckets
        self.rel_pos = rel_pos
        if self.rel_pos == "enc_only":
            self.relative_pe = nn.Embedding.from_pretrained(
                embeddings=get_sin_encodings(rel_pos_buckets, model_dim),
                freeze=True,
                padding_idx=rel_pos_buckets
            )
            # self.W_kR = nn.Parameter(
            #     torch.Tensor(self.head_count, self.dim_per_head, self.dim_per_head), requires_grad=True)
            # self.b_kR = nn.Parameter(
            #     torch.Tensor(self.head_count, self.dim_per_head), requires_grad=True)

        elif self.rel_pos == "emb_only":
            self.relative_pe = nn.Embedding(
                rel_pos_buckets + 1,
                model_dim,
                padding_idx=rel_pos_buckets
            )
            # self.W_kR = nn.Parameter(
            #     torch.Tensor(self.head_count, self.dim_per_head, self.dim_per_head), requires_grad=True)
            # self.b_kR = nn.Parameter(
            #     torch.Tensor(self.head_count, self.dim_per_head), requires_grad=True)

        else:
            self.relative_pe = None
            self.W_kR = None
            self.b_kR = None

        self.u = u
        self.v = v

    def forward(self, inputs, mask, distances):
        """
        Compute the context vector and the attention vectors.

        Args:
           inputs (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
           distances: graph distance matrix (BUCKETED), ``(batch, key_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):

           * output context vectors ``(batch, query_len, dim)``
           * Attention vector in heads ``(batch, head, query_len, key_len)``.
        """

        batch_size = inputs.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query. Seems that we don't need layer_cache here
        query = self.linear_query(inputs)
        key = self.linear_keys(inputs)
        value = self.linear_values(inputs)

        key = shape(key)                # (b, t_k, h) -> (b, head, t_k, h/head)
        value = shape(value)
        query = shape(query)            # (b, t_q, h) -> (b, head, t_q, h/head)

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)

        if self.relative_pe is None:
            scores = torch.matmul(
                query, key.transpose(2, 3))                 # (b, head, t_q, t_k)

        else:
            # a + c
            u = self.u.reshape(1, head_count, 1, dim_per_head)
            a_c = torch.matmul(query + u, key.transpose(2, 3))

            rel_emb = self.relative_pe(distances)           # (b, t_q, t_k) -> (b, t_q, t_k, h)
            rel_emb = rel_emb.reshape(                      # (b, t_q, t_k, h) -> (b, t_q, t_k, head, h/head)
                batch_size, query_len, key_len, head_count, dim_per_head)

            # W_kR = self.W_kR.reshape(1, 1, 1, head_count, dim_per_head, dim_per_head)
            # rel_emb = torch.matmul(rel_emb, W_kR)           # (b, t_q, t_k, head, 1, h/head)
            # rel_emb = rel_emb.squeeze(-2)                   # (b, t_q, t_k, head, h/head)
            #
            # b_kR = self.b_kR.reshape(1, 1, 1, head_count, dim_per_head)
            # rel_emb = rel_emb + b_kR                        # (b, t_q, t_k, head, h/head)

            # b + d
            query = query.unsqueeze(-2)                     # (b, head, t_q, h/head) -> (b, head, t_q, 1, h/head)
            rel_emb_t = rel_emb.permute(0, 3, 1, 4, 2)      # (b, t_q, t_k, head, h/head) -> (b, head, t_q, h/head, t_k)

            v = self.v.reshape(1, head_count, 1, 1, dim_per_head)
            b_d = torch.matmul(query + v, rel_emb_t
                               ).squeeze(-2)                # (b, head, t_q, 1, t_k) -> (b, head, t_q, t_k)

            scores = a_c + b_d

        scores = scores.float()

        mask = mask.unsqueeze(1)                            # (B, 1, 1, T_values)
        scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)

        context_original = torch.matmul(drop_attn, value)   # -> (b, head, t_q, h/head)
        context = unshape(context_original)                 # -> (b, t_q, h)

        output = self.final_linear(context)
        attns = attn.view(batch_size, head_count, query_len, key_len)

        return output, attns

class SALayerXL(nn.Module):
    """
    A single layer of the self-attention encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout: dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout, rel_pos_buckets, u, v, rel_pos="emb_only"):
        super().__init__()

        self.self_attn = MultiHeadedRelAttention(
            heads, d_model, dropout=attention_dropout,
            rel_pos_buckets=rel_pos_buckets,
            u=u,
            v=v,
            rel_pos=rel_pos
        )
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask, distances):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``
            distances (LongTensor): ``(batch_size, src_len, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, mask=mask, distances=distances)
        out = self.dropout(context) + inputs

        return self.feed_forward(out)

class AttnEncoderXL(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff, dropout, attention_dropout, rel_pos_buckets,
                 enc_pos_encoding="transformer",rel_pos="emb_only",encoder_emb_scale="sqrt"):
        super().__init__()
        #self.args = args
        """
        self.num_layers = args.attn_enc_num_layers
        self.d_model = args.attn_enc_hidden_size
        self.heads = args.attn_enc_heads
        self.d_ff = args.attn_enc_filter_size
        self.attention_dropout = args.attn_dropout
        self.rel_pos_buckets = args.rel_pos_buckets
        """
        self.num_layers = num_layers                    # attn_enc_num_layers
        self.d_model = d_model                          # attn_enc_hidden_size
        self.heads = heads                              # attn_enc_heads
        self.d_ff = d_ff                                # attn_enc_filter_size
        self.dropout = dropout                          # dropout
        self.attention_dropout = attention_dropout      # attn_dropout
        self.rel_pos_buckets = rel_pos_buckets          # rel_pos_buckets
        self.rel_pos = rel_pos
        self.enc_pos_encoding = enc_pos_encoding        # encoder_positional_encoding
        self.encoder_pe = None
        self.encoder_emb_scale = encoder_emb_scale
        if self.enc_pos_encoding == "transformer":
            self.encoder_pe = PositionalEncoding(
                dropout=self.dropout,                       
                dim=self.d_model,
                max_len=1024        # temporary hard-code. Seems that onmt fix the denominator as 10000.0
            )
        else:
            self.dropout = nn.Dropout(p=dropout)

        if self.rel_pos in ["enc_only", "emb_only"]:
            self.u = nn.Parameter(torch.randn(self.d_model), requires_grad=True)
            self.v = nn.Parameter(torch.randn(self.d_model), requires_grad=True)
        else:
            self.u = None
            self.v = None

        self.attention_layers = nn.ModuleList(
            [SALayerXL(
                self.d_model, self.heads, self.d_ff, dropout, self.attention_dropout,
                self.rel_pos_buckets, self.u, self.v, self.rel_pos)
             for i in range(self.num_layers)])
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)

    def forward(self, src, lengths, distances):
        """adapt from onmt TransformerEncoder
            src: (t, b, h)
            lengths: (b,)
            distances: (b, t, t)
        """

        if self.encoder_pe is not None:
            emb = self.encoder_pe(src)
            out = emb.transpose(0, 1).contiguous()
        else:
            out = src.transpose(0, 1).contiguous()
            if self.encoder_emb_scale == "sqrt":
                out = out * math.sqrt(self.d_model)
            out = self.dropout(out)

        mask = ~sequence_mask(lengths).unsqueeze(1)

        for layer in self.attention_layers:
            out = layer(out, mask, distances)
        out = self.layer_norm(out)

        return out.transpose(0, 1).contiguous()
    
class RXNG2Sequencer(nn.Module):
    
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.add_empty_node = self.config.model.add_empty_node
        self.encoder = RXNGraphEncoder(gnum_layer=self.config.model.gnum_layer,
                                    emb_dim=self.config.model.emb_dim,
                                    gnn_type=self.config.model.gnn_type,
                                    JK=self.config.model.JK,
                                    drop_ratio=self.config.model.drop_ratio,
                                    gnn_aggr=self.config.model.gnn_aggr,
                                    bond_feat_red=self.config.model.bond_feat_red,
                                    node_readout=self.config.model.node_readout,
                                    )
        if self.config.model.att_encoder_type == 'attxl':
            self.attention_encoder = AttnEncoderXL(num_layers=self.config.model.decoder_num_layers, 
                                                d_model=self.config.model.emb_dim, 
                                                heads=self.config.model.num_heads, 
                                                d_ff=self.config.model.encoder_filter_size, 
                                                dropout=self.config.model.drop_ratio, 
                                                attention_dropout=self.config.model.drop_ratio, 
                                                rel_pos_buckets=self.config.model.rel_pos_buckets,
                                                enc_pos_encoding=None)
        elif self.config.model.att_encoder_type == 'attn':
            self.attention_pool = GlobalAttention(gate_nn=torch.nn.Linear(self.config.model.emb_dim, 1))
            self.attention_encoder = TransformerEncoder(num_layer=self.config.model.tnum_layer,
                                                        hidden_size=self.config.model.emb_dim,
                                                        intermediate_size=self.config.model.emb_dim,
                                                        num_heads=self.config.model.num_heads,
                                                        hidden_dropout_prob=self.config.model.attn_drop_ratio)
            



        self.decoder_embeddings = Embeddings(
            word_vec_size=self.config.model.emb_dim,
            word_vocab_size=self.vocab_size,
            word_padding_idx=self.vocab["_PAD"],
            position_encoding=True,
            dropout=self.config.model.drop_ratio
        )

        self.decoder = TransformerDecoder(
            num_layers=self.config.model.decoder_num_layers,
            d_model=self.config.model.emb_dim,
            heads=self.config.model.num_heads,
            d_ff=self.config.model.filter_size,
            copy_attn=False,
            self_attn_type="scaled-dot",
            dropout=self.config.model.drop_ratio,
            attention_dropout=self.config.model.drop_ratio,
            embeddings=self.decoder_embeddings,
            max_relative_positions=self.config.model.max_rel_pos,
            aan_useffn=False,
            full_context_alignment=False,
            alignment_layer=-3,
            alignment_heads=0
        )

        self.output_layer = nn.Linear(self.config.model.emb_dim, self.vocab_size, bias=True)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.vocab["_PAD"],
            reduction="mean"
        )

    def encode_and_reshape(self, reaction_batch):
        hatom, mol_index, batch = self.encoder(reaction_batch.x,reaction_batch.mol_index, reaction_batch.edge_index, reaction_batch.edge_attr)                         # (n_atoms, h)

        if self.config.model.att_encoder_type == 'attxl':
            memory_lengths = torch.bincount(reaction_batch.batch).long()
            max_length = max(memory_lengths)
            padded_memory_bank = []
            if not self.add_empty_node:
                assert sum(memory_lengths) == hatom.size(0), \
                    f"Memory lengths calculation error, encoder output: {hatom.size(0)}, memory_lengths: {memory_lengths}"    ## V1

                memory_bank = torch.split(hatom,  memory_lengths.tolist(), dim=0)   # [n_atoms, h] => 1+b tup of (t, h)        ## V1
                for length, h in zip(memory_lengths, memory_bank):                                                              ## V1
                    m = nn.ZeroPad2d((0, 0, 0, max_length - length))
                    padded_memory_bank.append(m(h))
            else:
                assert 1 + sum(memory_lengths) == hatom.size(0), \
                f"Memory lengths calculation error, encoder output: {hatom.size(0)}, memory_lengths: {memory_lengths}"
                memory_bank = torch.split(hatom, [1] + memory_lengths.tolist(), dim=0)   # [n_atoms, h] => 1+b tup of (t, h)
                for length, h in zip(memory_lengths, memory_bank[1:]):
                    m = nn.ZeroPad2d((0, 0, 0, max_length - length))
                    padded_memory_bank.append(m(h))
            

            padded_memory_bank = torch.stack(padded_memory_bank, dim=1)     # list of b (max_t, h) => [max_t, b, h]

            memory_lengths = torch.tensor(memory_lengths,
                                        dtype=torch.long,
                                        device=padded_memory_bank.device)
            if not self.add_empty_node:
                distances = calc_batch_graph_distance(batch=reaction_batch.batch,
                                                        edge_index=reaction_batch.edge_index,
                                                        task=self.config.model.task)
            else:
                distances = calc_batch_graph_distance(batch=reaction_batch.batch,
                                                        edge_index=reaction_batch.edge_index[:, 1:]-1,
                                                        task=self.config.model.task)
            if self.attention_encoder is not None:
                padded_memory_bank = self.attention_encoder(
                    padded_memory_bank,
                    memory_lengths,
                    distances
                )
        elif self.config.model.att_encoder_type == "attn":
            memory_lengths = torch.bincount(batch).long()
            max_length = max(memory_lengths)
            rxn_representation = self.attention_pool(hatom,mol_index)  ## node_representation is equal to hatom
            padded_feat = pad_feat(rxn_representation,batch,self.config.model.emb_dim)
            rxn_transf_emb = self.attention_encoder(padded_feat)
            padded_memory_bank = rxn_transf_emb.transpose(1,0)
        else:
            raise NotImplementedError(f"Attention encoder type {self.config.model.att_encoder_type} not implemented")

        self.decoder.state["src"] = np.zeros(max_length)    # TODO: this is hardcoded to make transformer decoder work
        #print(f"padded_memory_bank.shape: {padded_memory_bank.shape}, memory_lengths.shape: {memory_lengths.shape}")
        return padded_memory_bank, memory_lengths

    def forward(self, reaction_batch):
        #print("Enter Graph2SeqSeriesRel forward")
        padded_memory_bank, memory_lengths = self.encode_and_reshape(reaction_batch)

        # adapted from onmt.models
        dec_in = reaction_batch.tgt_token_ids[:, :-1]                       # pop last, insert SOS for decoder input
        m = nn.ConstantPad1d((1, 0), self.vocab["_SOS"])
        dec_in = m(dec_in)
        dec_in = dec_in.transpose(0, 1).unsqueeze(-1)                       # [b, max_tgt_t] => [max_tgt_t, b, 1]
        #print("!!!!!!!",dec_in.shape,padded_memory_bank.shape,memory_lengths.shape)
        dec_outs, _ = self.decoder(
            tgt=dec_in,
            memory_bank=padded_memory_bank,
            memory_lengths=memory_lengths
        )

        dec_outs = self.output_layer(dec_outs)                                  # [t, b, h] => [t, b, v]
        dec_outs = dec_outs.permute(1, 2, 0)                                    # [t, b, v] => [b, v, t]

        loss = self.criterion(
            input=dec_outs,
            target=reaction_batch.tgt_token_ids
        )

        predictions = torch.argmax(dec_outs, dim=1)                             # [b, t]
        mask = (reaction_batch.tgt_token_ids != self.vocab["_PAD"]).long()
        accs = (predictions == reaction_batch.tgt_token_ids).float()
        accs = accs * mask
        acc = accs.sum() / mask.sum()

        return loss, acc

    def infer(self, reaction_batch,
                     batch_size: int, beam_size: int, n_best: int, temperature: float,
                     min_length: int, max_length: int):
        if beam_size == 1:
            decode_strategy = GreedySearch(
                pad=self.vocab["_PAD"],
                bos=self.vocab["_SOS"],
                eos=self.vocab["_EOS"],
                batch_size=batch_size,
                min_length=min_length,
                max_length=max_length,
                block_ngram_repeat=0,
                exclusion_tokens=set(),
                return_attention=False,
                sampling_temp=0.0,
                keep_topk=1
            )
        else:
            global_scorer = GNMTGlobalScorer(alpha=0.0,
                                             beta=0.0,
                                             length_penalty="none",
                                             coverage_penalty="none")
            decode_strategy = BeamSearch(
                beam_size=beam_size,
                batch_size=batch_size,
                pad=self.vocab["_PAD"],
                bos=self.vocab["_SOS"],
                eos=self.vocab["_EOS"],
                n_best=n_best,
                global_scorer=global_scorer,
                min_length=min_length,
                max_length=max_length,
                return_attention=False,
                block_ngram_repeat=0,
                exclusion_tokens=set(),
                stepwise_penalty=None,
                ratio=0.0
            )

        padded_memory_bank, memory_lengths = self.encode_and_reshape(reaction_batch=reaction_batch)
        # adapted from onmt.translate.translator
        results = {
            "predictions": None,
            "scores": None,
            "attention": None
        }

        # (2) prep decode_strategy. Possibly repeat src objects.
        src_map = None
        target_prefix = None
        fn_map_state, memory_bank, memory_lengths, src_map = decode_strategy.initialize(
            memory_bank=padded_memory_bank,
            src_lengths=memory_lengths,
            src_map=src_map,
            target_prefix=target_prefix
        )

        # (3) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            decoder_input = decode_strategy.current_predictions.view(1, -1, 1)
            # print(f"[{step}] decoder_input.shape: {decoder_input.shape}, memory_bank.shape: {len(memory_bank)}, memory_lengths: {len(memory_lengths)}")
            # print(f"[{step}] memory_bank: {memory_bank[0]}, memory_lengths: {memory_lengths[0]}")
            dec_out, dec_attn = self.decoder(tgt=decoder_input,
                                             memory_bank=memory_bank,
                                             memory_lengths=memory_lengths,
                                             step=step)

            if "std" in dec_attn:
                attn = dec_attn["std"]
            else:
                attn = None

            dec_out = self.output_layer(dec_out)            # [t, b, h] => [t, b, v]
            dec_out = dec_out / temperature
            dec_out = dec_out.squeeze(0)                    # [t, b, v] => [b, v]
            log_probs = F.log_softmax(dec_out, dim=-1)

            # log_probs = self.model.generator(dec_out.squeeze(0))

            decode_strategy.advance(log_probs, attn)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices

            if any_finished:
                # Reorder states.
                if isinstance(memory_bank, tuple):
                    memory_bank = tuple(x.index_select(1, select_indices)
                                        for x in memory_bank)
                else:
                    memory_bank = memory_bank.index_select(1, select_indices)

                memory_lengths = memory_lengths.index_select(0, select_indices)

                if src_map is not None:
                    src_map = src_map.index_select(1, select_indices)

            if any_finished:
                self.map_state(
                    lambda state, dim: state.index_select(dim, select_indices))

        results["scores"] = decode_strategy.scores
        results["predictions"] = decode_strategy.predictions
        results["attention"] = decode_strategy.attention
        results["alignment"] = [[] for _ in range(4096)]

        return results

    # adapted from onmt.decoders.transformer
    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        # self.decoder.state["src"] = fn(self.decoder.state["src"], 1)
        # => self.state["src"] = self.state["src"].index_select(1, select_indices)

        if self.decoder.state["cache"] is not None:
            _recursive_map(self.decoder.state["cache"])

class RXNGraphormer():
    def __init__(self, task_type, config, vocab, pretrained_ensemble={"pretrained_encoder":None,"pretrained_rct_encoder":None,
                                                                      "pretrained_pdt_encoder":None,"pretrained_mid_encoder":None}):
        assert task_type in ["regression","classification","sequence_generation"]
        self.task_type = task_type
        self.config = config
        self.vocab = vocab
        self.pretrained_ensemble = pretrained_ensemble
    def get_model(self, task_type=None, config=None, vocab=None, pretrained_ensemble={"pretrained_encoder":None,"pretrained_rct_encoder":None,
                                                                      "pretrained_pdt_encoder":None,"pretrained_mid_encoder":None}):
        if task_type is None:
            task_type = self.task_type
            config = self.config
            vocab = self.vocab
            pretrained_ensemble = self.pretrained_ensemble
        else:
            assert task_type is not None and config is not None

        if task_type == 'regression':
            model = RXNGRegressor(gnum_layer=config.gnum_layer,
                                  tnum_layer=config.tnum_layer,
                                  onum_layer=config.onum_layer,
                                  emb_dim=config.emb_dim,
                                  JK=config.JK,
                                  output_size=config.output_size,
                                  drop_ratio=config.drop_ratio,
                                  num_heads=config.num_heads,
                                  gnn_type=config.gnn_type,
                                  bond_feat_red=config.bond_feat_red,
                                  gnn_aggr=config.gnn_aggr,
                                  node_readout=config.node_readout,
                                  trans_readout=config.trans_readout,
                                  graph_pooling=config.graph_pooling,
                                  attn_drop_ratio=config.attn_drop_ratio,
                                  encoder_filter_size=config.encoder_filter_size,
                                  rel_pos_buckets=config.rel_pos_buckets,
                                  rel_pos=config.rel_pos,
                                  pretrained_encoder=pretrained_ensemble["pretrained_encoder"],
                                  pretrained_rct_encoder=pretrained_ensemble["pretrained_rct_encoder"],
                                  pretrained_pdt_encoder=pretrained_ensemble["pretrained_pdt_encoder"],
                                  output_norm=config.output_norm,
                                  split_process=config.split_process,
                                  use_mid_inf=config.use_mid_inf,
                                  interaction=config.interaction,
                                  interaction_layer_num=config.interaction_layer_num,
                                  pretrained_mid_encoder=pretrained_ensemble["pretrained_mid_encoder"],
                                  mid_iteract_method=config.mid_iteract_method,
                                  split_merge_method=config.split_merge_method,
                                  output_act_func=config.output_act_func,
                                  rct_batch_norm=config.rct_batch_norm,
                                  pdt_batch_norm=config.pdt_batch_norm,
                                  mid_batch_norm=config.mid_batch_norm,
                                  mid_layer_num=config.mid_layer_num)
        
        elif task_type == 'classification':
            model = RXNGClassifier(gnum_layer=config.gnum_layer,
                                   tnum_layer=config.tnum_layer,
                                   onum_layer=config.onum_layer,
                                   emb_dim=config.emb_dim,
                                   JK=config.JK,
                                   output_size=config.output_size,
                                   drop_ratio=config.drop_ratio,
                                   num_heads=config.num_heads,
                                   gnn_type=config.gnn_type,
                                   bond_feat_red=config.bond_feat_red,
                                   gnn_aggr=config.gnn_aggr,
                                   node_readout=config.node_readout,
                                   trans_readout=config.trans_readout,
                                   graph_pooling=config.graph_pooling,
                                   attn_drop_ratio=config.attn_drop_ratio,
                                   encoder_filter_size=config.encoder_filter_size,
                                   rel_pos_buckets=config.rel_pos_buckets,
                                   rel_pos=config.rel_pos,
                                   split_process=config.split_process,
                                   split_merge_method=config.split_merge_method,
                                   output_act_func=config.output_act_func)
        
        elif task_type == "sequence_generation":
            model = RXNG2Sequencer(config, vocab)
        else:
            raise NotImplementedError
        return model