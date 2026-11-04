import inspect
import os, pickle, sys

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

source_file = os.path.abspath(__file__)
sys.path.append(os.path.dirname(source_file))
from data_utils_cltot import auto_repr, collect_class_definitions

with open(source_file, 'rb') as fp:
    file_ = fp.read()
    network_pyfile = pickle.dumps(file_)
    
def activation_func(name='relu', alpha=1.0, negative_slope=1e-2):
    """
    Return a PyTorch activation module based on the provided name.

    Args:
        name (str): Name of the activation function.
        alpha (float): Parameter for ELU/CELU.
        negative_slope (float): Parameter for LeakyReLU.

    Returns:
        nn.Module: Corresponding activation function.
    """
    name = name.lower()
    if name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'selu':
        return nn.SELU()
    elif name == 'elu':
        return nn.ELU(alpha)
    elif name == 'leakyrelu':
        return nn.LeakyReLU(negative_slope)
    elif name == 'celu':
        return nn.CELU(alpha)
    elif name == 'gelu':
        return nn.GELU()
    elif name is None:
        return nn.Identity()
    else:
        raise ValueError(f'Unsupported activation function: "{name}"')
    

def make_mlp(list_dims, dropout=0.0, act_func='relu', norm_type='layer', alpha=1.0, negative_slope=1e-2):
    
    """
    Constructs a multi-layer perceptron (MLP) using a sequence of linear layers, each optionally followed by 
    normalization, activation, and dropout layers.

    Args:
        list_dims (List[int]): A list specifying the input, hidden, and output dimensions. 
                               For example, [128, 256, 64, 1] creates 3 layers.
        dropout (float): Dropout rate to apply after each activation (except the final layer). Default is 0.0.
        act_func (Union[str, List[str], None]): Activation function(s) to use. 
            - If a single string (e.g., 'relu', 'elu', 'leakyrelu'), it is used for all hidden layers. 
              If list_dims=[128,256,64,1], then act_func list must have length 2 (for the two hidden layers [256,64]).
            - If a list of strings, it must have `len(list_dims) - 2` entries (for hidden layers only).
            - If None, no activation is applied.
        norm_type (str): Type of normalization to apply after each linear layer:
            - 'batch' for BatchNorm1d
            - 'layer' for LayerNorm
            - 'NA' for No Normalization
            - None for no normalization.
        alpha (float): The α parameter used for ELU/CELU activations. Default is 1.0.
        negative_slope (float): The negative slope parameter used for LeakyReLU. Default is 1e-2.

    Returns:
        nn.Sequential: A fully constructed MLP as a `torch.nn.Sequential` block.
    """
    layers = nn.ModuleList([])

    num_layers = len(list_dims) - 1
    if act_func is None or not act_func:
        act_funcs = [nn.Identity()] * (num_layers - 1)
    elif isinstance(act_func, str):
        act_funcs = [activation_func(act_func, alpha=alpha, negative_slope=negative_slope)] * (num_layers - 1)
    elif isinstance(act_func, list):
        assert (len(act_func) == num_layers - 1), f"Length of act_func list {len(act_func)} must match the number of hidden layers {(len(list_dims) - 2)}."
        act_funcs = [activation_func(name, alpha=alpha, negative_slope=negative_slope) for name in act_func]
    else:
        raise TypeError("act_func must be a string or a list of strings.")

    if num_layers > 0:
        for i in range(num_layers):
            in_dim = list_dims[i]
            out_dim = list_dims[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))

            if i < num_layers - 1:
                # Normalization
                if norm_type == 'batch':
                    layers.append(nn.BatchNorm1d(out_dim))
                elif norm_type == 'layer':
                    layers.append(nn.LayerNorm(out_dim))
                elif (norm_type == 'NA') or (norm_type is None):
                    layers.append(nn.Identity())
                elif norm_type is not None:
                    raise ValueError(f"Unsupported normalization type: {norm_type}")
                
                # Activation and Dropout
                layers.append(act_funcs[i])
                layers.append(nn.Dropout(dropout))
    else:
        layers.append(nn.Identity())

    return nn.Sequential(*layers)


    
def normalize_adjacency(
    adjacency,
    degree,
    laplacian=False
):
    """
    Normalize adjacency (or Laplacian) matrices for both single and multi-relational cases.

    Args:
        adjacency (Tensor): Shape [B, N, N] or [B, R, N, N].
        degree (Tensor):    Same shape as adjacency.
        laplacian (bool):   Whether to compute normalized Laplacian instead of adjacency.

    Returns:
        Tensor: Normalized adjacency or Laplacian matrix of same shape as input.
    """
    if adjacency.dim() == 4:  # [B, R, N, N] multi-relational case
        B, R, N, _ = adjacency.shape
        normalized = torch.zeros_like(adjacency)
        
        for rel in range(R):
            adj = adjacency[:, rel]       # [B, N, N]
            deg = degree[:, rel]   if degree.dim() == 4  else degree    # [B, N, N]

            deg_inv_sqrt = deg.clone()
            mask = deg_inv_sqrt != 0
            deg_inv_sqrt[mask] = deg_inv_sqrt[mask].pow(-0.5)

            if laplacian:
                lap = deg - adj
                norm = torch.bmm(torch.bmm(deg_inv_sqrt, lap), deg_inv_sqrt)
            else:
                norm = torch.bmm(torch.bmm(deg_inv_sqrt, adj), deg_inv_sqrt)

            normalized[:, rel] = norm

        return normalized

    elif adjacency.dim() == 3:  # [B, N, N] single-relation case
        deg_inv_sqrt = degree.clone()
        mask = deg_inv_sqrt != 0
        deg_inv_sqrt[mask] = deg_inv_sqrt[mask].pow(-0.5)

        if laplacian:
            lap = degree - adjacency
            normalized = torch.bmm(torch.bmm(deg_inv_sqrt, lap), deg_inv_sqrt)
        else:
            normalized = torch.bmm(torch.bmm(deg_inv_sqrt, adjacency), deg_inv_sqrt)

        return normalized

    else:
        raise ValueError("Adjacency tensor must be 3D or 4D")
  
class FusionNet(nn.Module):
    def __init__(self,
                 input_dims,        # list or tuple of input sizes, e.g., [91, 384, 300, 256]
                 fusion_dim=256,
                 mlp_params=None):
        """
        input_dims: list of input feature dimensions for each modality.
        fusion_dim: projection size for each input modality.
        mlp_params: parameters to build final MLP (must include output dim).
        """
        super().__init__()

        self.input_dims = input_dims
        self.fusion_dim = fusion_dim
        self.mlp_params = mlp_params
        
        # Create projection layers dynamically
        self.projections = nn.ModuleList([nn.Linear(dim, fusion_dim) for dim in input_dims])

        # Final projection after concatenation
        self.final_proj = nn.Linear(fusion_dim * len(input_dims), fusion_dim)
        self.norm = nn.LayerNorm(fusion_dim * len(input_dims))
        self.dropout = nn.Dropout(p=0.3)

        # MLP for prediction
        self.mlp = make_mlp(**mlp_params)

    def __repr__(self):
        return auto_repr(self)

    def feature_repr(self, modalities):
        """
        modalities: variable number of tensors.
                    Each tensor shape -> [batch, input_dim_i]
        """
        assert len(modalities) == len(self.projections), \
            f"Expected {len(self.projections)} inputs but received {len(modalities)}."

        # Project each modality to fusion dimension
        projected = [proj(mod) for proj, mod in zip(self.projections, modalities)]

        # Concatenate all projected features
        fused = torch.cat(projected, dim=-1)
        fused = self.norm(fused)
        fused = F.tanh(fused)
        fused = self.dropout(fused)

        # Final fusion projection + dropout
        fused = self.final_proj(fused)

        return fused

    def forward(self, modalities):
        x = self.feature_repr(modalities)
        return self.mlp(x)  
    
class SMILESLinearNet(nn.Module):
    """
    Task:
        Feedforward network for regression or classification on SMILES-based vector inputs.

    Inputs:
        - list_dims (List[int]): List of dimensions for each linear layer.
        - dropout (float): Dropout probability applied after each layer except the last.

    Outputs:
        - Tensor of shape [batch_size, output_dim] after passing through linear layers.
    """
    def __init__(self, list_dims, dropout=0.2, act_func='relu', norm_type='layer'):
        super().__init__()
        
        self.class_def = inspect.getsource(self.__class__)
        self.list_dims = list_dims
        self.dropout = dropout
        self.act_func = act_func
        self.norm_type = norm_type
        self.network_pyfile = network_pyfile
        self.MLP = make_mlp(list_dims=list_dims,
                            dropout=dropout,
                            act_func=act_func,
                            norm_type=norm_type
                            )
        
        # Weight Initialization
        for layer in self.MLP:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
                
    def __repr__(self):
        return auto_repr(self)
    
    def forward(self, x):
        """
        Task:
            Forward pass through MLP with ReLU, BatchNorm, and Dropout.

        Inputs:
            - x (Tensor): Input tensor of shape [batch_size, list_dims[0]]

        Outputs:
            - Tensor: Output tensor after final linear layer.
        """
        return self.MLP(x)

## Multi Layer GCN Network
class RGCNConv(nn.Module):
    """
    Task:
        Relation-aware Graph Convolution Layer for multi-relational graphs.

    Inputs:
        - in_channels (int): Input node feature dim.
        - out_channels (int): Output node feature dim.
        - num_relations (int): Number of bond types or edge relations.

    Outputs:
        - Tensor: Transformed node features with all relation contributions.
    """
    def __init__(self, in_channels, out_channels, num_relations=4, bias=True):
        super(RGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        self.weights = nn.Parameter(torch.Tensor(num_relations, in_channels, out_channels))
        self.self_loop_weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
        self.reset_parameters()
        
    def __repr__(self):
        return auto_repr(self)

    def reset_parameters(self):
        """Task: Initialize parameters using Xavier initialization."""
        nn.init.xavier_uniform_(self.weights)
        nn.init.xavier_uniform_(self.self_loop_weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adjacency_tensor):
        """
        Task:
            Perform multi-relational message passing.

        Inputs:
            - x (Tensor): Node features [batch_size, num_nodes, in_channels]
            - adjacency_tensor (Tensor): [batch_size, num_relations, num_nodes, num_nodes]

        Outputs:
            - Tensor: Updated node features [batch_size, num_nodes, out_channels]
        """
        batch_size, num_relations, num_nodes, _ = adjacency_tensor.size()
        out = torch.zeros(batch_size, num_nodes, self.out_channels, device=x.device)

        for rel in range(num_relations):
            adj = adjacency_tensor[:, rel]  # Shape: [batch_size, num_nodes, num_nodes]

            h_rel = torch.matmul(x, self.weights[rel])  # Shape: [batch_size, num_nodes, out_channels]
            out += torch.matmul(adj, h_rel)
            
        # ********************************************************************************
        out += torch.matmul(x, self.self_loop_weight)
        # out = out/num_relations
        # ********************************************************************************

        if self.bias is not None:
            out += self.bias

        return out

class MRGCN(nn.Module):
    """
    Multi-Relational Graph Convolutional Network with optional attention-based pooling and dense prediction head.

    Args:
        list_dims_gcn (list[int]): Dimensions for GCN layers.
        list_dims_fc (list[int]): Dimensions for fully connected layers.
        dropout (float): Dropout rate.
        max_num_atom (int): Max number of atoms (used for pooling logic externally).
        num_relations (int): Number of edge types in the graph.
        act_func (str): Activation function for FC layers.
        act_func_gcn (str): Activation function for GCN layers.
        norm_type (str): Type of normalization ('layer', 'batch', etc.).
        alpha (float): Parameter for certain activation functions (e.g., ELU).
        negative_slope (float): Slope for LeakyReLU.
        use_attention (bool): Whether to use attention-based pooling.
        attn_type (str): Type of attention pooling ('self attention' or 'attention pool').
        attn_dim (int): Attention dimension.
        num_heads (int): Number of attention heads (for multi-head attention).

    Outputs:
        - Tensor: Final prediction or representation [batch_size, output_dim]
    """
    def __init__(self,
                 list_dims_gcn,
                 list_dims_fc,
                 max_num_atom,
                 dropout = 0.2,
                 num_relations=4,
                 act_func='relu',
                 act_func_gcn='relu',
                 norm_type='layer',
                 alpha=1.0,
                 negative_slope=1e-2,
                 use_attention=False,       # <--- flag for QK^T V
                 attn_type='self attention',# 'attention pool',#           
                 attn_dim=None,            # <--- dimension of attention projection
                 num_heads=1):             # <--- optional multi-head attention
        super().__init__()
        self.max_num_atom = max_num_atom
        self.list_dims_gcn = list_dims_gcn
        self.list_dims_fc = list_dims_fc
        self.num_relations = num_relations
        self.class_def = inspect.getsource(self.__class__)
        self.network_pyfile = network_pyfile
        self.act_func = act_func
        self.act_func_gcn = act_func_gcn
        self.norm_type = norm_type
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.alpha = alpha
        self.negative_slope = negative_slope
        self.use_attention = use_attention
        self.attn_type = attn_type

        # GCN Layers
        self.gcn_layers = nn.ModuleList([
            RGCNConv(list_dims_gcn[i], list_dims_gcn[i + 1], num_relations=self.num_relations)
            for i in range(len(list_dims_gcn) - 1)
        ])
        self.gcn_layer_norms = nn.ModuleList([
            nn.LayerNorm(list_dims_gcn[i + 1]) for i in range(len(list_dims_gcn) - 2)
        ])
        
        # --- QK^T V Attention Pooling ---
        hidden_dim = list_dims_gcn[-1]
        attn_dim = attn_dim or hidden_dim
        if self.use_attention and self.attn_type=='self attention':
            self.W_Q = nn.Linear(hidden_dim, attn_dim * num_heads)
            self.W_K = nn.Linear(hidden_dim, attn_dim * num_heads)
            self.W_V = nn.Linear(hidden_dim, attn_dim * num_heads)
            self.W_out = nn.Linear(attn_dim * num_heads, hidden_dim)
        if self.use_attention and self.attn_type=='attention pool':
            # self.att_pool = nn.Linear(hidden_dim, 1)
            self.att_pool = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2),
                                          nn.Tanh(), nn.Linear(hidden_dim // 2, 1)
                                          )

        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.activation_func_gcn = activation_func(self.act_func_gcn, alpha=alpha, negative_slope=negative_slope)
        if len(self.list_dims_fc) >0:
            self.fc_conn = nn.Linear(list_dims_gcn[-1], list_dims_fc[0])
            self.input2repr = make_mlp(list_dims=list_dims_fc, dropout=dropout, act_func=act_func, norm_type=norm_type)
        else:
            self.fc_conn = nn.Identity()
            self.input2repr = nn.Identity()
        
    def __repr__(self):
        return auto_repr(self)

    def get_features(self, feature_vector, adjacency_tensor, degree_tensor, feature_mask=1.0, adjacency_mask=1.0):
        """
        Task:
            Extract graph-level features from multi-relational input.

        Inputs:
            - x (Tensor): Node features
            - adjacency_tensor (Tensor): Edge-type-specific adjacency matrices
            - degree_tensor (Tensor): Node degree matrices

        Outputs:
            - Tensor: Aggregated graph features [batch_size, gcn_out_dim]
        """
        feature_vector = feature_vector * feature_mask
        adjacency_tensor = adjacency_tensor * adjacency_mask
        degree_tensor = degree_tensor * adjacency_mask
        
        adjacency_tensor = normalize_adjacency(adjacency=adjacency_tensor,
                                               degree=degree_tensor
                                               )
        x = feature_vector
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, adjacency_tensor)
            if i < len(self.gcn_layers) - 1:
                x = self.activation_func_gcn(x)
                x = self.gcn_layer_norms[i](x)
                x = self.dropout(x)

        # --- Conditional pooling ---
        if self.use_attention and self.attn_type=='self attention':
            B, N, D = x.shape  # batch, num_nodes, hidden_dim
            H = self.num_heads
            d_k = D // H

            Q = self.W_Q(x).view(B, N, H, -1).transpose(1, 2)  # [B, H, N, d_k]
            K = self.W_K(x).view(B, N, H, -1).transpose(1, 2)  # [B, H, N, d_k]
            V = self.W_V(x).view(B, N, H, -1).transpose(1, 2)  # [B, H, N, d_k]

            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # [B, H, N, N]
            attn_weights = torch.softmax(attn_scores, dim=-1)                  # attention matrix
            x = torch.matmul(attn_weights, V)                                  # [B, H, N, d_k]

            x = x.mean(dim=2)                                                  # pool over nodes
            x = x.transpose(1, 2).contiguous().view(B, -1)                     # concat heads
            x = x.reshape(B, -1)
            x = self.W_out(x)                                                  # [B, D]
        elif self.use_attention and self.attn_type=='attention pool':
            att_weights = torch.softmax(self.att_pool(x), dim=1)  # [batch, num_nodes, 1]
            x = torch.sum(att_weights * x, dim=1)
        else:
            # simple mean pooling
            x = x.mean(dim=1)
            
        self.feature_dim = x.shape[-1]
        return x

    def forward(self, feature_vector, adjacency_tensor, degree_tensor, feature_mask=1.0, adjacency_mask=1.0):
        """
        Task:
            Forward pass from graph input to final prediction via GCN and FC layers.

        Inputs:
            - x: Node features
            - adjacency_tensor: Multi-relational adjacency
            - degree_tensor: Node degree tensor
            - feature_mask (float or Tensor): Mask to apply on features
            - adjacency_mask (float or Tensor): Mask to apply on adjacency

        Outputs:
            - Tensor: Final prediction/representation
        """
        x = self.get_features(feature_vector, adjacency_tensor, degree_tensor,
                              feature_mask=feature_mask, adjacency_mask=adjacency_mask)
        x = self.fc_conn(x)
        x = self.input2repr(x)
        return x
    
    
class ChemBERTa(nn.Module):
    """
    Task:
        ChemBERTa-based regressor for molecular property prediction using AutoModel.
    """
    def __init__(self,
                 model_name: str,
                 list_dims: list[int],
                 dropout: float = 0.2,
                 act_func: str = 'relu',
                 norm_type: str = 'layer',
                 ):
        """
        Task:
            Initialize model with a pretrained transformer and a regression MLP.

        Input:
            model_name (str): Huggingface model identifier (e.g., 'seyonec/ChemBERTa').
            list_dims (list[int]): Dimensions for the MLP head.
            dropout (float): Dropout probability.

        Output:
            None
        """
        super().__init__()
        self.class_def = collect_class_definitions(self)
        self.network_pyfile=network_pyfile
        
        self.model_name = model_name
        self.list_dims = list_dims
        self.dropout = dropout
        self.act_func = act_func
        self.norm_type = norm_type
        self.config = AutoConfig.from_pretrained(model_name)
        self.chemberta = AutoModel.from_pretrained(model_name, config=self.config)
        
        self.hidden_size = self.config.hidden_size
        self.list_dims = [self.hidden_size, ] + list_dims
        self.regressor = make_mlp(list_dims=self.list_dims, dropout=self.dropout, act_func=self.act_func, norm_type=self.norm_type)

        
    def __repr__(self):
        """
        Task:
            String representation of the model including its name and configuration.

        Output:
            str: Model name and configuration.
        """
        return auto_repr(self)
    
    def token_embeddings(self, input_ids, attention_mask, **kwargs):
        
        outputs = self.chemberta(input_ids=input_ids, attention_mask=attention_mask)
        token_embed = outputs.last_hidden_state  # [B, T, H]

        return token_embed

    def embeddings(self, input_ids, attention_mask, **kwargs):
        """
        Task:
            Compute mean-pooled embedding over valid token positions.

        Input:
            input_ids (Tensor): Input token IDs of shape [B, T].
            attention_mask (Tensor): Attention mask of shape [B, T].

        Output:
            Tensor: Mean pooled embedding of shape [B, H].
        """
        token_embed = self.token_embeddings(input_ids=input_ids, attention_mask=attention_mask)
        x = token_embed[:, 0]  # CLS token
        
        return x
    
    def forward(self, input_ids, attention_mask, **kwargs):
        """
        Task:
            Perform forward pass for regression prediction.

        Input:
            input_ids (Tensor): Tokenized inputs.
            attention_mask (Tensor): Valid attention positions.

        Output:
            Tensor: Regression prediction.
        """
        x = self.embeddings(input_ids=input_ids, attention_mask=attention_mask)
        x = self.regressor(x)
        
        return x
    

class ChemBERTaRegressor(ChemBERTa):
    """
    Task:
        ChemBERTa-based regressor for molecular property prediction using AutoModel.
    """
    def __init__(self,
                 model_name: str,
                 list_dims: list[int],
                 dropout: float = 0.2,
                 act_func: str = 'relu',
                 norm_type: str = 'layer',
                 ):
        """
        Task:
            Initialize model with a pretrained transformer and a regression MLP.

        Input:
            model_name (str): Huggingface model identifier (e.g., 'seyonec/ChemBERTa').
            list_dims (list[int]): Dimensions for the MLP head.
            dropout (float): Dropout probability.

        Output:
            None
        """
        super().__init__(model_name=model_name,
                         list_dims=list_dims,
                         dropout=dropout,
                         act_func=act_func,
                         norm_type=norm_type
                         )
        self.model_name = model_name
        self.dropout = dropout
        self.act_func = act_func
        self.list_dims = list_dims
        self.norm_type = norm_type

        self.hidden_size = self.config.hidden_size
        list_dims = [self.hidden_size, ] + list_dims
        self.regressor = make_mlp(list_dims=list_dims, 
                                  dropout=self.dropout, 
                                  act_func=self.act_func, 
                                  norm_type=self.norm_type
                                  )
        
    
    def embeddings(self, input_ids, attention_mask, **kwargs):
        """
        Task:
            Compute mean-pooled embedding over valid token positions.

        Input:
            input_ids (Tensor): Input token IDs of shape [B, T].
            attention_mask (Tensor): Attention mask of shape [B, T].

        Output:
            Tensor: Mean pooled embedding of shape [B, H].
        """
        token_embed = self.token_embeddings(input_ids=input_ids, attention_mask=attention_mask)
        masked_embed = (token_embed * attention_mask.unsqueeze(-1)).sum(1)
        denom = attention_mask.sum(1, keepdim=True).clamp(min=1e-6)
        x = masked_embed / denom 
        
        return x
    

    def forward(self, input_ids, attention_mask, **kwargs):
        """
        Task:
            Perform forward pass for regression prediction.

        Input:
            input_ids (Tensor): Tokenized inputs.
            attention_mask (Tensor): Valid attention positions.

        Output:
            Tensor: Regression prediction.
        """
        x = self.embeddings(input_ids=input_ids, attention_mask=attention_mask)
        x = self.regressor(x)
        return x