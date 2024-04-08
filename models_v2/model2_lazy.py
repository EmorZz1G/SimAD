import torch
import torch.nn as nn
import math
"""
启发，加入查询嵌入是否有效果02.03、11：25.原本在PSM最多为0.518
还真是！我超了，现在350IT，F1 0.554，后面还是降到了0.545，这个是让V作为MEM
然后试了一下，把K作为MEM，350IT=F1 0.519，最多0.522，然后下降到0.515
第二次，同样V做MEM，350IT，F1 0.56，后面下降到0.54
第三次，其他不变，复原PATCH2，不降噪，似乎350IT，F1 0.556，后面下降到0.545

"""


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000, add=False):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        # L, 1
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        # 
        if d_model % 2 == 0:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.add = add

    def forward(self, x):
        if self.add:
            x = x + self.pe[:, :x.size(1)]
            return x
        else:
            return self.pe[:, :x.size(1)]
        
class PositionalEmbedding_wo_ch(nn.Module):
    def __init__(self, d_model, max_len=5000, add=False):
        super(PositionalEmbedding_wo_ch, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        # L, 1
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        # 
        if d_model % 2 == 0:
            pe[:, 0::2] = torch.sin(position)
            pe[:, 1::2] = torch.cos(position)
        else:
            pe[:, 0::2] = torch.sin(position)
            pe[:, 1::2] = torch.cos(position)[:, :-1]

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.add = add

    def forward(self, x):
        if self.add:
            x = x + self.pe[:, :x.size(1)]
            return x
        else:
            return self.pe[:, :x.size(1)]
        

class MyPositionalEmbedding_wo_ch(nn.Module):
    def __init__(self, d_model, max_len=5000, add=False):
        super(MyPositionalEmbedding_wo_ch, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.arange(0, max_len).float()/max_len
        pe.require_grad = False

        pe = torch.sin(pe)

        pe = pe.unsqueeze(0).unsqueeze(-1)
        self.register_buffer('pe', pe)
        self.add = add

    def forward(self, x):
        if self.add:
            x = x + self.pe[:, :x.size(1)]
            return x
        else:
            return self.pe[:, :x.size(1)]


    
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
    

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, max_len=9000, add=False, dropout=0.00):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model, max_len=max_len, add=add)

        self.dropout = nn.Dropout(p=dropout, inplace=False)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        # x = x + self.position_embedding(x)
        return self.dropout(x)
    
class Mlp(nn.Module):
    def __init__(self, 
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x
    

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class Attention(nn.Module):

    def __init__(self, d_model, n_head, max_len = 1000, seq_len=0, mask=False, dropout=0.) -> None:
        super().__init__()

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        
        assert seq_len>0
        self.v = nn.Embedding(max_len, d_model)
        self.v2 = nn.Sequential(Rearrange('b l d -> b d l'), nn.Linear(max_len, seq_len), Rearrange('b d l -> b l d'))
        self.out_proj = nn.Linear(d_model, d_model)

        self.n_head = n_head
        self.mask = mask
        self.mask_map = None
        self.dropout = nn.Dropout(dropout)

    def forward(self,q,k,v,return_attention=True):
        q = self.q(q)
        k = self.k(k)
        B, N, D = q.shape
        # print(self.v2[1].weight.shape)
        v = self.v.weight.unsqueeze(0).expand(B, -1, -1)
        # print(v.shape)
        # print(self.v2[1].weight.shape,'---')
        v2 = self.v2(v)
        # v = v + v2
        v = v2
        # print(q.shape, k.shape, v.shape)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.n_head)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.n_head)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.n_head)

        # attn = torch.einsum('b h i d, b h j d -> b h i j', q, k) / (k.shape[-1] ** 0.5)
        attn = torch.matmul(q, k.transpose(-1, -2)) / (k.shape[-1] ** 0.5)
        

        if self.mask:
            L1 = attn.shape[-2]
            L2 = attn.shape[-1]
            if self.mask_map is None or self.mask_map.shape[-1] != L2:
                self.mask_map = torch.zeros((L1, L2), device=attn.device)
                for i in range(L1):
                    for j in range(L2):
                        if abs(i - j) <= 3:
                            self.mask_map[i, j] = 1
            mask_map = self.mask_map.unsqueeze(0).unsqueeze(0).expand(attn.shape[0], self.n_head, -1, -1)
            attn = attn.masked_fill(mask_map.bool(), -float('inf'))


        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        # print(attn.shape, v.shape)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_proj(out)
        # out B, N, D
        # attn B, H, N, N
        if return_attention:
            return out, attn
        else:
            return out, None


class EncoderLayer(nn.Module):
    def __init__(self, 
                 self_attention,
                 d_model,
                 d_feed_foward=None,
                 dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        d_feed_foward = d_feed_foward or 4 * d_model
        self.self_attention = self_attention
        self.ffn = Mlp(d_model, d_feed_foward, drop=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        new_x, _ = self.self_attention(
            x, x, x, 
            return_attention=False
        )
        new_x = x + self.dropout(new_x)
              
        y = x = self.norm1(new_x)
        y = self.ffn(y)
        out = self.norm2(x + y)

        return out



class Encoder(nn.Module):
    def __init__(self, encode_layers, norm_layer=None):
        super(Encoder, self).__init__()
        
        self.encode_layers = nn.ModuleList(encode_layers)
        self.norm = norm_layer

    def forward(self, x):
        for layer in self.encode_layers:
            x = layer(x)

        if self.norm is not None:
            x = self.norm(x)

        return x


from einops.layers.torch import Rearrange, Reduce


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str='norm'):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = torch.ones(self.num_features)
        self.affine_bias = torch.zeros(self.num_features)
        self.affine_weight=nn.Parameter(self.affine_weight)
        self.affine_bias=nn.Parameter(self.affine_bias)
        

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
            

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x




class ContAD_wo_ci(nn.Module):

    def __init__(self, c_dim, seq_len=2048, patch_size=16, d_model=256, n_layers=3, d_feed_foward_scale=4, n_head=4, dropout=0.0, with_inter=1,with_intra=1, proj_dim=10, query_len=1000) -> None:
        super().__init__()
        assert d_model % n_head == 0
        assert seq_len % patch_size == 0, f'seq={seq_len} patch={patch_size}'
        d_feed_foward = d_model * d_feed_foward_scale

        # Embedding
        # self.embedding = nn.Sequential(RevIN(c_dim), DataEmbedding(c_dim, c_dim, dropout=dropout))
        # self.embedding = nn.Sequential(RevIN(c_dim))
        # self.embedding = nn.Sequential(DataEmbedding(c_dim, c_dim, dropout=dropout))
        self.embedding = nn.Sequential(RevIN(c_dim), MyPositionalEmbedding_wo_ch(c_dim, seq_len, add=True))
        # self.embedding = nn.Sequential(MyPositionalEmbedding_wo_ch(c_dim, seq_len, add=True))

        # self.tcn = nn.Sequential(Rearrange('b l c -> b c l'),nn.Conv1d(c_dim, c_dim, kernel_size=3, padding='same'),nn.ReLU(),nn.Conv1d(c_dim, c_dim, kernel_size=7, padding='same'),nn.ReLU(),Rearrange('b c l -> b l c'))

        # Patching
        self.to_patch_embedding = nn.Sequential(Rearrange('b (l p) c -> b l (p c)', p=patch_size),nn.LayerNorm(patch_size*c_dim), nn.Linear(patch_size*c_dim, d_model), nn.LayerNorm(d_model))


        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer( 
                    Attention(d_model, n_head, mask=False, max_len=query_len, dropout=dropout, seq_len=seq_len//patch_size),
                    d_model,
                    d_feed_foward,
                    dropout=dropout
                ) for l in range(n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, patch_size*c_dim)


        self.projection2 = nn.Sequential(nn.Linear(d_model,d_model),
                                        # nn.LayerNorm(d_model),
                                        nn.ReLU(),
                                        nn.Linear(d_model, patch_size*proj_dim))
                                        # ,
                                        # Rearrange('b n (p c) -> b (n p) c',c=proj_dim))

        self.classifer = nn.Sequential(nn.Linear(d_model,d_model),nn.ReLU(),nn.Linear(d_model, patch_size),Rearrange('b n p -> b (n p)'))
        
        self.patch_size = patch_size
        self.c_dim = c_dim

    def forward(self, x):
        # print(x.shape)
        x = self.embedding(x)
        # x = self.tcn(x)
        x = self.to_patch_embedding(x)
        x = self.encoder(x)

        # if self.training:
        #     x = x + torch.randn_like(x) * 1.
        x_out = self.projection(x)
        sim_score = self.projection2(x)
        # BC, N, P
        return x_out, sim_score
    

    

def rec_score_func(x_out, x_patch, c_dim):
    from torch.nn.functional import interpolate
    l2_loss = nn.MSELoss(reduction='none')
    cos_loss = nn.CosineSimilarity(dim=-1)
    l2_score = l2_loss(x_out, x_patch)
    l2_score = rearrange(l2_score, '(b c) l p -> b (l p) c', c=c_dim).mean(-1)
    seq_len = l2_score.shape[1]
    cos_score = 1 - cos_loss(x_out, x_patch)
    cos_score = rearrange(cos_score, '(b c) l -> b c l', c=c_dim)
    cos_score = interpolate(cos_score, size=seq_len, mode='linear', align_corners=False).mean(1)
    score = l2_score + cos_score
    return score

def rec_score_func2(x_out, x_patch, c_dim):
    from torch.nn.functional import interpolate
    l2_loss = nn.MSELoss(reduction='none')
    cos_loss = nn.CosineSimilarity(dim=-1)
    l2_score = l2_loss(x_out, x_patch)
    l2_score = rearrange(l2_score, 'b l (p c) -> b (l p) c', c=c_dim).mean(-1)
    seq_len = l2_score.shape[1]
    cos_score = 1 - cos_loss(x_out, x_patch)
    cos_score = interpolate(cos_score.unsqueeze(1), size=seq_len, mode='linear', align_corners=False).squeeze(1)
    score = l2_score + cos_score
    return score

if __name__ == '__main__':
    model = ContAD_wo_ci(15, seq_len=2048, patch_size=16, d_model=256, n_layers=3, d_feed_foward_scale=4, n_head=4, dropout=0.0, with_inter=1,with_intra=1)
    x = torch.randn(32, 2048, 15)
    x_out, intra_corrs_list, inter_corrs_list = model(x)
    l2_loss = nn.MSELoss(reduction='none')
    cos_loss = nn.CosineSimilarity(dim=-1)

    x_patch = rearrange(x,'b (l p) c -> (b c) l p', p=model.patch_size)
    x_patch = rearrange(x,'b (l p) c -> b l (p c)', p=model.patch_size)
    # rec_score = l2_loss(x_out, x_patch) + 1 - cos_loss(x_out, x_patch)
    # print(rec_score.shape)
    # print(cos_loss(x_out,x_patch).shape)
    rec_score_func2(x_out, x_patch, model.c_dim)

