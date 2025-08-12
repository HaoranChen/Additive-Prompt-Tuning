import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from .vit import VisionTransformer
from .moco import vit_base as moco_base
import numpy as np
import copy
from timm.models.layers import trunc_normal_, DropPath
import random
import math
from operator import mul
from functools import reduce


class APT(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, ema_coeff):

        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.n_tasks = n_tasks
        self._init_smart(prompt_param)

        self.merge_flag = True

        self.ema_coeff = ema_coeff

        self.prompt_tokens = create_prompt_with_init(12*2, emb_d) 
        global_merged_prompt = torch.zeros(12*2, emb_d).cuda()
        self.register_buffer('global_merged_prompt', global_merged_prompt.clone().detach()) 

        trunc_normal_(self.prompt_tokens, std=0.02)

        for i in range(12):
            setattr(self, f'k_layer_proj{i}', nn.Linear(2, 2))
            setattr(self, f'v_layer_proj{i}', nn.Linear(2, 2))
         
   
    def merge_prompt(self, prompt1, prompt2):
        print("Merging prompt ... ")
        return prompt1*self.ema_coeff + prompt2*(1-self.ema_coeff)

    def _init_smart(self, prompt_param):
        self.prompt_dropout_ratio = float(prompt_param[0])
        self.prompt_dropout = nn.Dropout(self.prompt_dropout_ratio)

    def process_task_count(self):
        self.task_count += 1

    def forward(self, l, x_block, train=False):
        B, _, _ = x_block.shape

        prompt_groups = self.prompt_tokens
        
        if train or not self.merge_flag:
            P_root_k = prompt_groups[l*2:l*2+1].reshape(12,1,64).expand(B,12,1,64)
            P_root_v = prompt_groups[l*2+1:l*2+2].reshape(12,1,64).expand(B,12,1,64)
        elif not train and self.merge_flag:
            P_root_k = self.global_merged_prompt[l*2:l*2+1].reshape(12,1,64).expand(B,12,1,64)
            P_root_v = self.global_merged_prompt[l*2+1:l*2+2].reshape(12,1,64).expand(B,12,1,64)
        else:
            raise ValueError("merge flag and mode err")

        P_k = torch.cat((P_root_k, torch.zeros((B,12,196,64),device =x_block.device)),dim=-2)
        P_v = torch.cat((P_root_v, torch.zeros((B,12,196,64),device =x_block.device)),dim=-2)
        
        P = [P_k, P_v]    

        return P #, rpt_index

# note - ortho init has not been found to help l2p/dual prompt
def create_prompt_with_init(a, b, c=None, ortho=False, mean=None, std=None, init_ref=None):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    
    if ortho:
        nn.init.orthogonal_(p)
    elif init_ref is not None:
        p = torch.nn.Parameter(init_ref.squeeze(dim=0).expand(a, b),  requires_grad=True)
    elif mean and std:
        nn.init.normal_(p, mean=mean, std=std)
    else:
        nn.init.uniform_(p)
    return p

class ViTZoo(nn.Module):
    def __init__(self, num_classes=10, ema_coeff=0.5, pt=False, prompt_flag=False, prompt_param=None, tasks=[]):
        super(ViTZoo, self).__init__()
        self.num_classes = num_classes
        # get last layer

        self.prompt_flag = prompt_flag
        self.task_id = None
    
        self.tasks = tasks

        # get feature encoder
        if pt:
            zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                        num_heads=12, ckpt_layer=0,
                                        drop_path_rate=0
                                        )
            from timm.models import vit_base_patch16_224
            load_dict = vit_base_patch16_224(pretrained=True).state_dict()
            del load_dict['head.weight']; del load_dict['head.bias']
            zoo_model.load_state_dict(load_dict,strict=False)
        else:
            pass
        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model
     
        #classifier
        self.last = nn.Linear(768, num_classes) 
        self.clf_norm = nn.LayerNorm(768)

        # create prompting module
        if self.prompt_flag == 'apt':
            self.prompt = APT(768, prompt_param[0], prompt_param[1], ema_coeff=ema_coeff)
        else:
            self.prompt = None

        if self.prompt_flag == "apt":
            tuned_params = [
            "clf_norm.weight","clf_norm.bias",
            "prompt.prompt_tokens",
            "last.weight",
            "last.bias", 
            ] 
        else:
            tuned_params = [
            "clf_norm.weight","clf_norm.bias",
            "last.weight",
            "last.bias", 
            ]

        for name, param in self.named_parameters():
            if name in tuned_params:
                param.requires_grad = True
            else:
                param.requires_grad = False
           

    def get_attn_score_within_heads(self, attn_matrix, dim, method="mean"):
        if method == "mean":
            return attn_matrix.mean(dim=dim)

        elif method == "max":
            return attn_matrix.max(dim=dim)[0]
 
    def forward(self, x, train=False):
        if self.prompt is not None:
            if self.prompt_flag == 'apt':
                out = self.feat(x, prompt=self.prompt, train=train)
                out =  out[:,0,:]
            else: 
                raise ValueError("prompt flag not supported")
               
        else:
            out, _, _ = self.feat(x, train=train)
            out = out[:,0,:]

        out = self.clf_norm(out)
        wt_norm = F.normalize(self.last.weight, p=2, dim=1) 
        out = torch.matmul(out, wt_norm.t())

        return out
   

class MoCoZoo(ViTZoo):
    def __init__(self, num_classes=10, pt=False, prompt_flag=False, prompt_param=None, tasks=[]):
        super(MoCoZoo, self).__init__(num_classes, pt, prompt_flag, prompt_param, tasks)
       
        if pt:
            zoo_model = moco_base()#VisionTransformerMoCo(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                     #   num_heads=12,
                                    #    drop_path_rate=0
                                   #     )
            ckpt = "/share/ckpt/cgn/vpt/model/mocov3_linear-vit-b-300ep.pth.tar"

            checkpoint = torch.load(ckpt, map_location="cpu")
            load_dict = checkpoint['state_dict']
            for k in list(load_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.'):
                    # remove prefix
                    load_dict[k[len("module."):]] = load_dict[k]
                # delete renamed or unused k
                del load_dict[k]

            del load_dict['head.weight']; del load_dict['head.bias']
            zoo_model.load_state_dict(load_dict, strict=False)

        else:
            pass
        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model

def vit_pt_imnet(out_dim, ema_coeff, tasks=[], prompt_flag = 'None', prompt_param=None):
    return ViTZoo(num_classes=out_dim, ema_coeff=ema_coeff, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param, tasks=tasks)
    
def moco_pt(out_dim, tasks=[], prompt_flag = 'None', prompt_param=None):
    return MoCoZoo(num_classes=out_dim, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param, tasks=tasks)
