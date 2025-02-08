import torch
from torch import nn
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops,softmax
from torch_geometric.nn.inits import glorot, zeros
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from .utils import scaled_dot_product_attention,index_select_ND
from .data import NUM_BOND_TYPE,NUM_BOND_DIRECTION,NUM_BOND_STEREO,\
                   NUM_BOND_INRING,NUM_BOND_ISCONJ

def get_mess_around_edge(edge_index,mess):
    # create node to edge map
    num_nodes = torch.max(edge_index) + 1
    node_to_edges = [[] for _ in range(num_nodes)]
    for edge_idx, (src, dst) in enumerate(edge_index.t()):
        node_to_edges[src.item()].append(edge_idx)
        node_to_edges[dst.item()].append(edge_idx)

    # abstract node-edge features
    node_edge_features = []
    for edges in node_to_edges:
        if edges:
            edge_features = mess[edges]
            node_edge_features.append(edge_features)
        else:
            node_edge_features.append(torch.zeros(1, mess.size(1)))

    node_edge_features_padded = pad_sequence(node_edge_features, batch_first=True, padding_value=0)
    return node_edge_features_padded

class GCNConv(MessagePassing):
    # adapted from https://github.com/junxia97/Mole-BERT
    def __init__(self, emb_dim, aggr = "add", bond_feat_red="mean"):
        super().__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        self.edge_embedding3 = torch.nn.Embedding(NUM_BOND_STEREO, emb_dim)
        self.edge_embedding4 = torch.nn.Embedding(NUM_BOND_INRING, emb_dim)
        self.edge_embedding5 = torch.nn.Embedding(NUM_BOND_ISCONJ, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding5.weight.data)
        
        self.edge_embedding_lst = [self.edge_embedding1, self.edge_embedding2, self.edge_embedding3, self.edge_embedding4, self.edge_embedding5]

        self.aggr = aggr
        self.bond_feat_red = bond_feat_red

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, edge_attr):
        edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))
        self_loop_attr = torch.zeros(x.size(0), len(self.edge_embedding_lst))
        self_loop_attr[:,0] = NUM_BOND_TYPE - 1  #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)
        edge_embeddings = []
        for i in range(edge_attr.shape[1]):
            edge_embeddings.append(self.edge_embedding_lst[i](edge_attr[:,i]))
        if self.bond_feat_red == "mean":
            edge_embeddings = torch.stack(edge_embeddings).mean(dim=0)
        elif self.bond_feat_red == "sum":
            edge_embeddings = torch.stack(edge_embeddings).sum(dim=0)
        else:
            raise ValueError("Invalid bond feature reduction method. Please choose from 'mean' or 'sum'")
        
        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)
        return self.propagate(edge_index=edge_index, aggr=self.aggr, x=x, edge_attr=edge_embeddings, norm = norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)

class GCNEdgeConv(MessagePassing):
    # adapted from https://github.com/junxia97/Mole-BERT
    def __init__(self, emb_dim, aggr = "add", bond_feat_red="mean"):
        super().__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        self.edge_embedding3 = torch.nn.Embedding(NUM_BOND_STEREO, emb_dim)
        self.edge_embedding4 = torch.nn.Embedding(NUM_BOND_INRING, emb_dim)
        self.edge_embedding5 = torch.nn.Embedding(NUM_BOND_ISCONJ, emb_dim)
        
        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding5.weight.data)
        
        self.edge_embedding_lst = [self.edge_embedding1, self.edge_embedding2, self.edge_embedding3, self.edge_embedding4, self.edge_embedding5]
        self.bridge_layer = torch.nn.Linear(emb_dim*3, emb_dim)
        self.aggr = aggr
        self.bond_feat_red = bond_feat_red

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, edge_attr):
        edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))
        self_loop_attr = torch.zeros(x.size(0), len(self.edge_embedding_lst))
        self_loop_attr[:,0] = NUM_BOND_TYPE - 1  #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)
        edge_embeddings = []
        for i in range(edge_attr.shape[1]):
            edge_embeddings.append(self.edge_embedding_lst[i](edge_attr[:,i]))
        if self.bond_feat_red == "mean":
            edge_embeddings = torch.stack(edge_embeddings).mean(dim=0)
        elif self.bond_feat_red == "sum":
            edge_embeddings = torch.stack(edge_embeddings).sum(dim=0)
        else:
            raise ValueError("Invalid bond feature reduction method. Please choose from 'mean' or 'sum'")
        
        mess1 = x.index_select(index=edge_index[0],dim=0)        #mess1 = h.index_select(index=edge_index[0], dim=0)
        mess2 = edge_embeddings.clone()        #mess2 = edge_attr
        mess = torch.cat([mess1,mess2],dim=-1)        #mess = torch.cat([mess1,mess2],dim=-1)
        nei_mess = get_mess_around_edge(edge_index, mess)
        nei_mess = nei_mess.sum(dim=1)
        node_emb = torch.cat([x, nei_mess], dim=1)
        x = self.bridge_layer(node_emb)
        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)
        return self.propagate(edge_index=edge_index, aggr=self.aggr, x=x, edge_attr=edge_embeddings, norm = norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)

class SimpGCNConv(MessagePassing):
    # adapted from https://github.com/junxia97/Mole-BERT
    def __init__(self, emb_dim, aggr = "add", bond_feat_red="mean"):
        super().__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding = torch.nn.Embedding(max([NUM_BOND_TYPE,NUM_BOND_DIRECTION,NUM_BOND_STEREO,NUM_BOND_INRING,NUM_BOND_ISCONJ]), emb_dim)
        self.edge_dim_size = len([NUM_BOND_TYPE,NUM_BOND_DIRECTION,NUM_BOND_STEREO,NUM_BOND_INRING,NUM_BOND_ISCONJ])
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        self.aggr = aggr
        self.bond_feat_red = bond_feat_red

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))
        self_loop_attr = torch.zeros(x.size(0), self.edge_dim_size)
        self_loop_attr[:,0] = NUM_BOND_TYPE - 1  #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        if self.bond_feat_red == "mean":
            edge_embeddings = self.edge_embedding(edge_attr).mean(dim=1)
        elif self.bond_feat_red == "sum":
            edge_embeddings = self.edge_embedding(edge_attr).sum(dim=1)
        else:
            raise ValueError("Invalid bond feature reduction method. Please choose from 'mean' or 'sum'")
        
        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)
        return self.propagate(edge_index=edge_index, aggr=self.aggr, x=x, edge_attr=edge_embeddings, norm = norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)

class RGCNConv(MessagePassing):
    # adapted from https://github.com/junxia97/Mole-BERT
    def __init__(self, emb_dim, aggr = "add", bond_feat_red="mean"):
        super().__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.rnn = nn.GRUCell(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        self.edge_embedding3 = torch.nn.Embedding(NUM_BOND_STEREO, emb_dim)
        self.edge_embedding4 = torch.nn.Embedding(NUM_BOND_INRING, emb_dim)
        self.edge_embedding5 = torch.nn.Embedding(NUM_BOND_ISCONJ, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding5.weight.data)
        
        self.edge_embedding_lst = [self.edge_embedding1, self.edge_embedding2, self.edge_embedding3, self.edge_embedding4, self.edge_embedding5]

        self.aggr = aggr
        self.bond_feat_red = bond_feat_red

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, edge_attr):
        edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))
        self_loop_attr = torch.zeros(x.size(0), len(self.edge_embedding_lst))
        self_loop_attr[:,0] = NUM_BOND_TYPE - 1  #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)
        edge_embeddings = []
        for i in range(edge_attr.shape[1]):
            edge_embeddings.append(self.edge_embedding_lst[i](edge_attr[:,i]))
        if self.bond_feat_red == "mean":
            edge_embeddings = torch.stack(edge_embeddings).mean(dim=0)
        elif self.bond_feat_red == "sum":
            edge_embeddings = torch.stack(edge_embeddings).sum(dim=0)
        else:
            raise ValueError("Invalid bond feature reduction method. Please choose from 'mean' or 'sum'")
        
        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)
        x = self.propagate(edge_index=edge_index, aggr=self.aggr, x=x, edge_attr=edge_embeddings, norm=norm)
        
        # 初始化隐藏状态
        hidden_state = torch.zeros_like(x)
        new_hidden_state = torch.zeros_like(x)

        # 更新节点状态，避免原地操作
        for node_idx in range(x.size(0)):
            new_hidden_state[node_idx] = self.rnn(x[node_idx], hidden_state[node_idx])

        #hidden_state = torch.zeros_like(x)
        #for node_idx in range(x.size(0)):
        #    hidden_state[node_idx] = self.rnn(x[node_idx], hidden_state[node_idx])
        return new_hidden_state

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)
    
class GINConv(MessagePassing):
    """
    Adapted from https://github.com/junxia97/Mole-BERT
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, out_dim, aggr = "add", bond_feat_red="mean"):
        self.aggr = aggr
        super().__init__()
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, out_dim))
        self.edge_embedding1 = torch.nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        self.edge_embedding3 = torch.nn.Embedding(NUM_BOND_STEREO, emb_dim)
        self.edge_embedding4 = torch.nn.Embedding(NUM_BOND_INRING, emb_dim)
        self.edge_embedding5 = torch.nn.Embedding(NUM_BOND_ISCONJ, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding5.weight.data)
        
        self.edge_embedding_lst = [self.edge_embedding1, self.edge_embedding2, self.edge_embedding3, self.edge_embedding4, self.edge_embedding5]
        self.bond_feat_red = bond_feat_red
    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), len(self.edge_embedding_lst))
        self_loop_attr[:,0] = NUM_BOND_TYPE - 1 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = []
        for i in range(edge_attr.shape[1]):
            edge_embeddings.append(self.edge_embedding_lst[i](edge_attr[:,i]))
        if self.bond_feat_red == "mean":
            edge_embeddings = torch.stack(edge_embeddings).mean(dim=0)
            #edge_embeddings = (self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1]) + self.edge_embedding3(edge_attr[:,2]) + self.edge_embedding4(edge_attr[:,3]) + self.edge_embedding5(edge_attr[:,4]))/5
        elif self.bond_feat_red == "sum":
            edge_embeddings = torch.stack(edge_embeddings).sum(dim=0)
            #edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1]) + self.edge_embedding3(edge_attr[:,2]) + self.edge_embedding4(edge_attr[:,3]) + self.edge_embedding5(edge_attr[:,4])
        else:
            raise ValueError("Invalid bond feature reduction method. Please choose from 'mean' or 'sum'")
        
        return self.propagate(edge_index=edge_index, aggr=self.aggr, x=x, edge_attr=edge_embeddings)
    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class SimpGINConv(MessagePassing):
    """
    Adapted from https://github.com/junxia97/Mole-BERT
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, out_dim, aggr = "add", bond_feat_red="mean"):
        self.aggr = aggr
        super().__init__()
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, out_dim))
        self.edge_embedding = torch.nn.Embedding(max([NUM_BOND_TYPE,NUM_BOND_DIRECTION,NUM_BOND_STEREO,NUM_BOND_INRING,NUM_BOND_ISCONJ]), emb_dim)
        self.edge_dim_size = len([NUM_BOND_TYPE,NUM_BOND_DIRECTION,NUM_BOND_STEREO,NUM_BOND_INRING,NUM_BOND_ISCONJ])
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        self.bond_feat_red = bond_feat_red
        
    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), self.edge_dim_size)
        self_loop_attr[:,0] = NUM_BOND_TYPE - 1 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        if self.bond_feat_red == "mean":
            edge_embeddings = self.edge_embedding(edge_attr).mean(dim=1)
            
        elif self.bond_feat_red == "sum":
            edge_embeddings = self.edge_embedding(edge_attr).sum(dim=1)
        else:
            raise ValueError("Invalid bond feature reduction method. Please choose from 'mean' or 'sum'")
        
        return self.propagate(edge_index=edge_index, aggr=self.aggr, x=x, edge_attr=edge_embeddings)
    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GATConv(MessagePassing):
    # TODO
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add",bond_feat_red="mean"):
        super().__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))
        
        self.edge_embedding1 = torch.nn.Embedding(NUM_BOND_TYPE, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(NUM_BOND_DIRECTION, heads * emb_dim)
        self.edge_embedding3 = torch.nn.Embedding(NUM_BOND_STEREO, heads * emb_dim)
        self.edge_embedding4 = torch.nn.Embedding(NUM_BOND_INRING, heads * emb_dim)
        self.edge_embedding5 = torch.nn.Embedding(NUM_BOND_ISCONJ, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding5.weight.data)
        
        self.edge_embedding_lst = [self.edge_embedding1, self.edge_embedding2, self.edge_embedding3, self.edge_embedding4, self.edge_embedding5]

        self.reset_parameters()
        self.bond_feat_red = bond_feat_red
    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):

        #add self loops in the edge space
        edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), len(self.edge_embedding_lst))
        self_loop_attr[:,0] = NUM_BOND_TYPE - 1 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = []
        for i in range(edge_attr.shape[1]):
            edge_embeddings.append(self.edge_embedding_lst[i](edge_attr[:,i]))
        if self.bond_feat_red == "mean":
            edge_embeddings = torch.stack(edge_embeddings).mean(dim=0)
            #edge_embeddings = (self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1]) + self.edge_embedding3(edge_attr[:,2]) + self.edge_embedding4(edge_attr[:,3]) + self.edge_embedding5(edge_attr[:,4]))/5
        elif self.bond_feat_red == "sum":
            edge_embeddings = torch.stack(edge_embeddings).sum(dim=0)
            #edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1]) + self.edge_embedding3(edge_attr[:,2]) + self.edge_embedding4(edge_attr[:,3]) + self.edge_embedding5(edge_attr[:,4])
        else:
            raise ValueError("Invalid bond feature reduction method. Please choose from 'mean' or 'sum'")
        
        #x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        x = self.weight_linear(x).view(-1, self.heads * self.emb_dim)
        return self.propagate(edge_index=edge_index, aggr=self.aggr, x=x, edge_attr=edge_embeddings)
    
    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j = x_j.view(-1, self.heads, self.emb_dim)
        x_i = x_i.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])
        return (x_j * alpha.view(-1, self.heads, 1)).mean(dim=1)

    def update(self, aggr_out):
        #aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out

class GraphSAGEConv(MessagePassing):
    # TODO 
    def __init__(self, emb_dim, aggr="mean", bond_feat_red="mean"):
        super().__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        
        self.edge_embedding1 = torch.nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        self.edge_embedding3 = torch.nn.Embedding(NUM_BOND_STEREO, emb_dim)
        self.edge_embedding4 = torch.nn.Embedding(NUM_BOND_INRING, emb_dim)
        self.edge_embedding5 = torch.nn.Embedding(NUM_BOND_ISCONJ, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding5.weight.data)

        self.aggr = aggr
        
        self.edge_embedding_lst = [self.edge_embedding1, self.edge_embedding2, self.edge_embedding3, self.edge_embedding4, self.edge_embedding5]
        self.bond_feat_red = bond_feat_red
    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), len(self.edge_embedding_lst))
        self_loop_attr[:,0] = NUM_BOND_TYPE - 1 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = []
        for i in range(edge_attr.shape[1]):
            edge_embeddings.append(self.edge_embedding_lst[i](edge_attr[:,i]))
        if self.bond_feat_red == "mean":
            edge_embeddings = torch.stack(edge_embeddings).mean(dim=0)

        elif self.bond_feat_red == "sum":
            edge_embeddings = torch.stack(edge_embeddings).sum(dim=0)

        else:
            raise ValueError("Invalid bond feature reduction method. Please choose from 'mean' or 'sum'")
        
        x = self.linear(x)

        return self.propagate(edge_index=edge_index, aggr=self.aggr, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p = 2, dim = -1)

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):
        attn_outputs = scaled_dot_product_attention(
            self.q(query), self.k(key), self.v(value), query_mask, key_mask, mask)
        return attn_outputs

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim,num_heads):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):
        x = torch.cat([
            h(query, key, value, query_mask, key_mask, mask) for h in self.heads
        ], dim=-1)
        x = self.output_linear(x)
        return x
    
class FeedForward(nn.Module):
    def __init__(self, hidden_size,intermediate_size,hidden_dropout_prob):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size, intermediate_size)
        self.linear_2 = nn.Linear(intermediate_size, hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x
    
class DGCNGRU(nn.Module):
    """GRU Message Passing layer."""
    def __init__(self, input_size: int, h_size: int, depth: int):
        super().__init__()
        self.input_size = input_size
        self.h_size = h_size
        self.depth = depth

        self._build_layer_components()

    def _build_layer_components(self) -> None:
        """Build layer components."""
        self.W_z = nn.Linear(self.input_size + self.h_size, self.h_size)
        self.W_r = nn.Linear(self.input_size, self.h_size, bias=False)
        self.U_r = nn.Linear(self.h_size, self.h_size)
        self.W_h = nn.Linear(self.input_size + self.h_size, self.h_size)

    def GRU(self, x: torch.Tensor, h_nei: torch.Tensor) -> torch.Tensor:
        """Implements the GRU gating equations.

        Parameters
        ----------
            x: torch.Tensor, input tensor
            h_nei: torch.Tensor, hidden states of the neighbors
        """
        sum_h = h_nei.sum(dim=1)                        # (9)
        z_input = torch.cat([x, sum_h], dim=1)          # x = [x_u; x_uv]
        z = torch.sigmoid(self.W_z(z_input))            # (10)

        r_1 = self.W_r(x).view(-1, 1, self.h_size)
        r_2 = self.U_r(h_nei)
        r = torch.sigmoid(r_1 + r_2)                    # (11) r_ku = f_r(x; m_ku) = W_r(x) + U_r(m_ku)

        gated_h = r * h_nei
        sum_gated_h = gated_h.sum(dim=1)                # (12)
        h_input = torch.cat([x, sum_gated_h], dim=1)
        pre_h = torch.tanh(self.W_h(h_input))           # (13)
        new_h = (1.0 - z) * sum_h + z * pre_h           # (14)

        return new_h

    def forward(self, fmess: torch.Tensor, bgraph: torch.Tensor) -> torch.Tensor:
        """Forward pass of the RNN

        Parameters
        ----------
            fmess: torch.Tensor, contains the initial features passed as messages
            bgraph: torch.Tensor, bond graph tensor. Contains who passes messages to whom.
        """
        h = torch.zeros(fmess.size()[0], self.h_size, device=fmess.device)
        mask = torch.ones(h.size()[0], 1, device=h.device)
        mask[0, 0] = 0      # first message is padding

        for i in range(self.depth):
            h_nei = index_select_ND(h, 0, bgraph)
            h = self.GRU(fmess, h_nei)
            h = h * mask
        return h

class DGCNEncoder(nn.Module):
    """MessagePassing Network based encoder. Messages are updated using an RNN
    and the final message is used to update atom embeddings.
    https://github.com/coleygroup/Graph2SMILES/blob/main/models/dgcn.py
    """
    def __init__(self, enc_hidden_size, enc_attn_heads, enc_num_layers, input_size, node_fdim,
                 drop_ratio=0.0,att_drop_ratio=0.0):
        super().__init__()
        self.h_size = enc_hidden_size
        self.enc_attn_heads = enc_attn_heads
        self.depth = enc_num_layers
        self.input_size = input_size
        self.node_fdim = node_fdim
        self.drop_ratio = drop_ratio
        self.att_drop_ratio = att_drop_ratio
        self._build_layers()

    def _build_layers(self) -> None:
        """Build layers associated with the MPNEncoder."""
        self.W_o = nn.Sequential(nn.Linear(self.node_fdim + self.h_size, self.h_size), nn.GELU())
        self.rnn = DGCNGRU(self.input_size, self.h_size, self.depth)

    def forward(self, fnode, fmess,
                agraph, bgraph, mask):
        """Forward pass of the MPNEncoder.

        Parameters
        ----------
            fnode: torch.Tensor, node feature tensor
            fmess: torch.Tensor, message features
            agraph: torch.Tensor, neighborhood of an atom
            bgraph: torch.Tensor, neighborhood of a bond,
                except the directed bond from the destination node to the source node
            mask: torch.Tensor, masks on nodes
        """
        h = self.rnn(fmess, bgraph)
        nei_message = index_select_ND(h, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        node_hiddens = torch.cat([fnode, nei_message], dim=1)
        node_hiddens = self.W_o(node_hiddens)

        if mask is None:
            mask = torch.ones(node_hiddens.size(0), 1, device=fnode.device)
            mask[0, 0] = 0      # first node is padding

        return node_hiddens * mask, h
    
class DGATGRU(nn.Module):
    """GRU Message Passing layer."""
    def __init__(self, enc_attn_heads, input_size, h_size, depth, drop_ratio, att_drop_ratio):
        super().__init__()
        self.enc_attn_heads = enc_attn_heads
        self.input_size = input_size
        self.h_size = h_size
        self.depth = depth
        self.drop_ratio = drop_ratio
        self.att_drop_ratio = att_drop_ratio

        self._build_layer_components()
        self._build_attention()

    def _build_layer_components(self) -> None:
        """Build layer components."""
        self.W_z = nn.Linear(self.input_size + self.h_size, self.h_size)
        self.W_r = nn.Linear(self.input_size, self.h_size, bias=False)
        self.U_r = nn.Linear(self.h_size, self.h_size)
        self.W_h = nn.Linear(self.input_size + self.h_size, self.h_size)

    def _build_attention(self) -> None:
        self.leaky_relu = nn.LeakyReLU()
        self.head_count = self.enc_attn_heads
        self.dim_per_head = self.h_size // self.head_count

        self.attn_alpha = nn.Parameter(
            torch.Tensor(1, 1, self.head_count, 2 * self.dim_per_head), requires_grad=True)
        self.attn_bias = nn.Parameter(
            torch.Tensor(1, 1, self.head_count), requires_grad=True)

        self.attn_W_q = nn.Linear(self.input_size, self.h_size, bias=True)
        self.attn_W_k = nn.Linear(self.h_size, self.h_size, bias=True)
        self.attn_W_v = nn.Linear(self.h_size, self.h_size, bias=True)

        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(self.drop_ratio)
        self.attn_dropout = nn.Dropout(self.att_drop_ratio)

    def GRU(self, x: torch.Tensor, h_nei: torch.Tensor) -> torch.Tensor:
        """Implements the GRU gating equations.

        Parameters
        ----------
            x: torch.Tensor, input tensor
            h_nei: torch.Tensor, hidden states of the neighbors
        """
        # attention-based aggregation
        n_node, max_nn, h_size = h_nei.size()
        head_count = self.head_count
        dim_per_head = self.dim_per_head

        q = self.attn_W_q(x)                            # (n_node, input) -> (n_node, h)
        q = q.unsqueeze(1).repeat(1, max_nn, 1)         # -> (n_node, max_nn, h)
        q = q.reshape(
            n_node, max_nn, head_count, dim_per_head)   # -> (n_node, max_nn, head, h/head)

        k = self.attn_W_k(h_nei)                        # (n_node, max_nn, h)
        k = k.reshape(
            n_node, max_nn, head_count, dim_per_head)   # -> (n_node, max_nn, head, h/head)

        v = self.attn_W_v(h_nei)                        # (n_node, max_nn, h)
        v = v.reshape(
            n_node, max_nn, head_count, dim_per_head)   # -> (n_node, max_nn, head, h/head)

        qk = torch.cat([q, k], dim=-1)                  # -> (n_node, max_nn, head, 2*h/head)
        qk = self.leaky_relu(qk)

        attn_score = qk * self.attn_alpha               # (n_node, max_nn, head, 2*h/head)
        attn_score = torch.sum(attn_score, dim=-1)      # (n_node, max_nn, head, 2*h/head) -> (n_node, max_nn, head)
        attn_score = attn_score + self.attn_bias        # (n_node, max_nn, head)

        attn_mask = (h_nei.sum(dim=2) == 0
                     ).unsqueeze(2)                     # (n_node, max_nn, h) -> (n_node, max_nn, 1)
        attn_score = attn_score.masked_fill(attn_mask, -1e18)

        attn_weight = self.softmax(attn_score)          # (n_node, max_nn, head), softmax over dim=1
        attn_weight = attn_weight.unsqueeze(3)          # -> (n_node, max_nn, head, 1)

        attn_context = attn_weight * v                  # -> (n_node, max_nn, head, h/head)
        attn_context = attn_context.reshape(
            n_node, max_nn, h_size)                     # -> (n_node, max_nn, h)

        sum_h = attn_context.sum(dim=1)                 # -> (n_node, h)

        # GRU
        z_input = torch.cat([x, sum_h], dim=1)          # x = [x_u; x_uv]
        z = torch.sigmoid(self.W_z(z_input))            # (10)

        r_1 = self.W_r(x)                               # (n_node, h) -> (n_node, h)
        r_2 = self.U_r(sum_h)                           # (n_node, h) -> (n_node, h)
        r = torch.sigmoid(r_1 + r_2)                    # (11) r_ku = f_r(x; m_ku) = W_r(x) + U_r(m_ku)

        sum_gated_h = r * sum_h                         # (n_node, h)
        h_input = torch.cat([x, sum_gated_h], dim=1)
        pre_h = torch.tanh(self.W_h(h_input))           # (13)
        new_h = (1.0 - z) * sum_h + z * pre_h           # (14)

        return new_h

    def forward(self, fmess: torch.Tensor, bgraph: torch.Tensor) -> torch.Tensor:
        """Forward pass of the RNN

        Parameters
        ----------
            fmess: torch.Tensor, contains the initial features passed as messages
            bgraph: torch.Tensor, bond graph tensor. Contains who passes messages to whom.
        """
        h = torch.zeros(fmess.size()[0], self.h_size, device=fmess.device)
        mask = torch.ones(h.size()[0], 1, device=h.device)
        mask[0, 0] = 0      # first message is padding

        for i in range(self.depth):
            h_nei = index_select_ND(h, 0, bgraph)
            h = self.GRU(fmess, h_nei)
            h = h * mask
        return h

class DGATEncoder(nn.Module):
    """MessagePassing Network based encoder. Messages are updated using an RNN
    and the final message is used to update atom embeddings.
    https://github.com/coleygroup/Graph2SMILES/blob/main/models/dgat.py
    """
    def __init__(self, enc_hidden_size, enc_attn_heads, enc_num_layers,input_size, node_fdim,
                 drop_ratio, att_drop_ratio):
        super().__init__()
    
        self.h_size = enc_hidden_size
        self.depth = enc_num_layers
        self.input_size = input_size
        self.node_fdim = node_fdim
        self.enc_attn_heads = enc_attn_heads
        self.dim_per_head = self.h_size // self.head_count
        self.drop_ratio = drop_ratio
        self.att_drop_ratio = att_drop_ratio
        self.leaky_relu = nn.LeakyReLU()

        self._build_layers()
        self._build_attention()

    def _build_layers(self) -> None:
        """Build layers associated with the MPNEncoder."""
        self.W_o = nn.Sequential(nn.Linear(self.node_fdim + self.h_size, self.h_size), nn.GELU())
        self.rnn = DGATGRU(self.enc_attn_heads, self.input_size, self.h_size, self.depth, 
                           self.drop_ratio, self.att_drop_ratio)

    def _build_attention(self) -> None:
        self.attn_alpha = nn.Parameter(
            torch.Tensor(1, 1, self.head_count, 2 * self.dim_per_head), requires_grad=True)
        self.attn_bias = nn.Parameter(
            torch.Tensor(1, 1, self.head_count), requires_grad=True)

        self.attn_W_q = nn.Linear(self.node_fdim, self.h_size, bias=True)
        self.attn_W_k = nn.Linear(self.h_size, self.h_size, bias=True)
        self.attn_W_v = nn.Linear(self.h_size, self.h_size, bias=True)

        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(self.drop_ratio)
        self.attn_dropout = nn.Dropout(self.att_drop_ratio)

    def forward(self, fnode, fmess,
                agraph, bgraph, mask):
        """Forward pass of the MPNEncoder.

        Parameters
        ----------
            fnode: torch.Tensor, node feature tensor
            fmess: torch.Tensor, message features
            agraph: torch.Tensor, neighborhood of an atom
            bgraph: torch.Tensor, neighborhood of a bond,
                except the directed bond from the destination node to the source node
            mask: torch.Tensor, masks on nodes
        """
        h = self.rnn(fmess, bgraph)
        nei_message = index_select_ND(h, 0, agraph)

        # attention-based aggregation
        n_node, max_nn, h_size = nei_message.size()
        head_count = self.head_count
        dim_per_head = self.dim_per_head

        q = self.attn_W_q(fnode)                        # (n_node, h)
        q = q.unsqueeze(1).repeat(1, max_nn, 1)         # -> (n_node, max_nn, h)
        q = q.reshape(
            n_node, max_nn, head_count, dim_per_head)   # (n_node, max_nn, h) -> (n_node, max_nn, head, h/head)

        k = self.attn_W_k(nei_message)                  # (n_node, max_nn, h)
        k = k.reshape(
            n_node, max_nn, head_count, dim_per_head)   # -> (n_node, max_nn, head, h/head)

        v = self.attn_W_v(nei_message)                  # (n_node, max_nn, h)
        v = v.reshape(
            n_node, max_nn, head_count, dim_per_head)   # -> (n_node, max_nn, head, h/head)

        qk = torch.cat([q, k], dim=-1)                  # -> (n_node, max_nn, head, 2*h/head)
        qk = self.leaky_relu(qk)

        attn_score = qk * self.attn_alpha               # (n_node, max_nn, head, 2*h/head)
        attn_score = torch.sum(attn_score, dim=-1)      # (n_node, max_nn, head, 2*h/head) -> (n_node, max_nn, head)
        attn_score = attn_score + self.attn_bias        # (n_node, max_nn, head)

        attn_mask = (nei_message.sum(dim=2) == 0
                     ).unsqueeze(2)                     # (n_node, max_nn, h) -> (n_node, max_nn, 1)
        attn_score = attn_score.masked_fill(attn_mask, -1e18)

        attn_weight = self.softmax(attn_score)          # (n_node, max_nn, head), softmax over dim=1
        attn_weight = attn_weight.unsqueeze(3)          # -> (n_node, max_nn, head, 1)

        attn_context = attn_weight * v                  # -> (n_node, max_nn, head, h/head)
        attn_context = attn_context.reshape(
            n_node, max_nn, h_size)                     # -> (n_node, max_nn, h)

        nei_message = attn_context.sum(dim=1)           # -> (n_node, h)

        # readout
        node_hiddens = torch.cat([fnode, nei_message], dim=1)
        node_hiddens = self.W_o(node_hiddens)

        if mask is None:
            mask = torch.ones(node_hiddens.size(0), 1, device=fnode.device)
            mask[0, 0] = 0      # first node is padding

        return node_hiddens * mask, h
    
    