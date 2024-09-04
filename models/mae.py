import torch
import timm
import numpy as np
import math
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from torch.nn import init
from einops import rearrange
from torch.nn.parameter import Parameter, UninitializedParameter

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

def take_indexes_channel_perm(sequences, indexes):
    return torch.gather(sequences, -1, repeat(indexes, 't c ->b t c',b = sequences.shape[0]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes

    
    
from torch.nn import init
from einops import rearrange
from torch.nn.parameter import Parameter, UninitializedParameter
class NonShareLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, num_groups: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NonShareLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups= num_groups
        self.weight = Parameter(torch.empty((out_features, in_features, num_groups), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        
        out_,in_,group_= self.weight.shape
        self.weight.data = self.weight.data.reshape(out_,-1)
        init.kaiming_uniform_(self.weight.data, a=math.sqrt(5),)
        self.weight.data = self.weight.data.reshape(out_,in_,group_)

#         tie_weight = torch.randn(self.out_features,self.in_features)
#         init.kaiming_uniform_(tie_weight, a=math.sqrt(5),)
#         self.weight.data = tie_weight.unsqueeze(2).repeat(1, 1, self.num_groups)
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    def forward(self,input):
        output = rearrange(input,"b (n d) f -> b n d f",d = self.num_groups)
        output = torch.einsum("ijdk,mkd -> ijdm", output, self.weight)
        if self.bias is not None:
            return rearrange(output,"b n d f -> b (n d) f")+self.bias   
        else:
            return rearrange(output,"b n d f -> b (n d) f")   
    
class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 idx,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 mlp_ratio= 4,
                 num_groups=1,
                 ) -> None:
        super().__init__()
        self.idx = idx
        
        self.patch_size = patch_size
        self.image_size = image_size
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        num_patch = (image_size // patch_size) ** 2
        self.patch_pad_area = len(idx)//num_patch
        self.pos_embedding = torch.nn.Parameter(torch.randn((num_patch, 1, emb_dim)))
#         self.pos_embedding = torch.nn.Parameter(torch.zeros(1,(image_size // patch_size) ** 2, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)
    
        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)
        
        self.patch_embs = NonShareLinear(self.patch_pad_area*3,emb_dim,num_groups,bias=True)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head,mlp_ratio=mlp_ratio) for _ in range(num_layer)])
        
        self.layer_norm = torch.nn.LayerNorm(emb_dim)
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)
        
    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
#         out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
#         out = out.reshape(x.size(0), (self.image_size // self.patch_size) ** 2 ,-1)
        x = rearrange(x,"b c h w -> b c (h w)")
        b,c,l = x.shape
        x_padded = torch.cat((x,torch.zeros(b,c,len(self.idx)-l).to(x.device)),dim=-1)
        out = torch.index_select(x_padded, 2, self.idx.to(x.device))
#         made error, swaped the color dimension and spatial dimension
#         out = rearrange(out, 'b c (l w) -> b l (c w)',w =self.patch_size*self.patch_size)
        out = rearrange(out, 'b c (l w) -> b l (w c)',w = self.patch_pad_area)
        return out
    
    def forward(self, img):
#         print(img.shape)
        patches = self._to_words(img)
#         print(patches.shape)
        patches = self.patch_embs(patches)
        patches = rearrange(patches, 'b t c -> t b c')
        patches = patches + self.pos_embedding
        if self.training:
            patches, forward_indexes, backward_indexes = self.shuffle(patches)
        else:
            backward_indexes = forward_indexes = None
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = F.normalize(features,dim=2)
#         features_t = rearrange(features, 'b t c -> t b c')
        return features, backward_indexes
    
    def feature_extract(self,img):
        patches = self._to_words(img)
        patches = self.patch_embs(patches)
        patches = rearrange(patches, 'b t c -> t b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = F.normalize(features,dim=2)
        return features
    

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 idx,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 mlp_ratio = 4,
                 num_groups=1,
                 ) -> None:
        super().__init__()
        self.idx = idx
        self.patch_size=patch_size
        self.image_size=image_size
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.num_patch = (image_size // patch_size) ** 2
        self.patch_pad_area = len(idx)//self.num_patch
        self.pos_embedding = torch.nn.Parameter(torch.randn((self.num_patch + 1, 1, emb_dim)))
#         self.pos_embedding = torch.nn.Parameter(torch.zeros(1,(image_size // patch_size) ** 2, emb_dim))
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head,mlp_ratio=mlp_ratio) for _ in range(num_layer)])
        self.head = NonShareLinear(emb_dim,self.patch_pad_area*3,num_groups,bias=True)
                                                
        self.init_weight()
        
    def _to_imgs(self, out):
        """
        (b, n, f) -> (b, c, h, w)
        """
#         made error, swaped the color dimension and spatial dimension
#         x = rearrange(x, 'b l (c w) -> b c (l w)',w =self.patch_size**2)
        x = rearrange(out, 'b l (w c) -> b c (l w)',w =self.patch_pad_area)
        x = torch.index_select(x, 2, self.idx.to(x.device))
        x_unpad = x[:,:,:self.image_size*self.image_size]
        x = rearrange(x_unpad," b c (h w) -> b c h w",h=self.image_size)
        
        return x
    
    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        features = rearrange(features, 'b t c -> t b c')
        T = features.shape[0]
        if self.training:
            backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
            features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
            features = take_indexes(features, backward_indexes)
        else:
            pass
        
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = features[:,1:]
        patches  = self.head(features)
        
#         features = rearrange(features, 'b t c -> t b c')
#         features = features[1:] # remove global feature
#         patches = self.head(features)
        
        if self.training:
            mask = torch.zeros_like(patches)
            mask = rearrange(mask, 'b t c -> t b c')
            mask[T:] = 1
            mask = take_indexes(mask, backward_indexes[1:] - 1)
            mask = rearrange(mask, 't b c -> b t c')
        
            img = self._to_imgs(patches)
            mask = self._to_imgs(mask)
        else:
            img = self._to_imgs(patches)
            mask = None
#         print(img.shape)
#         print(mask.shape)
#         img = self.patch2img(patches)
#         mask = self.patch2img(mask)

        return img, mask

class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 labels,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75,
                 mlp_ratio = 4,
                 num_groups=1,
                 ) -> None:
        super().__init__()
        labels_np = np.load(labels)
        self.labels = torch.from_numpy(labels_np).type(torch.long).view(-1)
        self.num_patch = int(max(self.labels))+1
        idx_padded_ls = []
        idx_ls=[]
        for i in range(self.num_patch): # 相同标签的放在一个list里
            idx = torch.nonzero(self.labels==i).squeeze(-1)
            idx_ls.append(idx)
        max_cluster = max([len(idx) for idx in idx_ls])
        pad_idx_start = max([max(idx) for idx in idx_ls])+1
        idx_padded_ls = []

        for idx in idx_ls:
            pad_amount = max_cluster - len(idx)
            idx_new = torch.cat((idx,torch.arange(pad_idx_start,pad_idx_start+pad_amount))) # 按照最大的cluster进行padding，顺序填充
            idx_padded_ls.append(idx_new)
            pad_idx_start+=pad_amount
        idx = torch.cat(idx_padded_ls)
        
        self.encoder = MAE_Encoder(idx, image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio, mlp_ratio,num_groups)
        self.decoder = MAE_Decoder(torch.argsort(idx), image_size, patch_size, emb_dim, decoder_layer, decoder_head, mlp_ratio,num_groups)

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features, backward_indexes)
        if self.training:         
            return predicted_img, mask
        else:
            return predicted_img

class MAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MAE_Encoder()
        self.decoder = MAE_Decoder()
    
    def forward(self, img):
        features = self.encoder(img)
        predicted_img, mask = self.decoder(features)
        return predicted_img, mask

class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder : MAE_Encoder, num_classes=10) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        logits = self.head(features[0])
        return logits