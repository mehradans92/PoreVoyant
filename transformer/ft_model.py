import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, Tensor

from MOFormer_modded.transformer import Transformer, TransformerRegressor
from MOFormer_modded.dataset_modded import MOF_ID_Dataset
from MOFormer_modded.tokenizer.mof_tokenizer import MOFTokenizer
import yaml
from MOFormer_modded.model.utils import *

from MOFormer_modded.transformer import PositionalEncoding
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class Transformer(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.token_encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model

        self.init_weights()

    def init_weights(self) -> None:
        # initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        nn.init.xavier_normal_(self.token_encoder.weight)

    def forward(self, src: Tensor) -> Tensor:
        src = self.token_encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output[:, 0:1, :] #this was added in by me

        return output.squeeze(dim = 1) #this was added in by me
    
class ClassificationTransformer(nn.Module):
    def __init__(self, transformer, input_dim = 512, hidden_dim = 256, output_dim = 1):
        super(ClassificationTransformer, self).__init__()
        self.model = transformer

        self.classification_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ) #needs to change

    def forward(self, x):
        x = self.model(x)
        x = x.mean(dim = 1)
        x = self.classification_head(x)

        return x

class RegressionTransformer(nn.Module):
    def __init__(self, model, mlp_hidden_dim=256):
        super(RegressionTransformer, self).__init__()
        
        #initialize model itself
        self.model = model

        #regression head
        self.regression_head = nn.Sequential(
            nn.Linear(512, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1)
        )
        
        #only updating MLP regression head
        #for params in self.model.parameters():
        #    params.requires_grad = False
                
    def forward(self, smiles):
        transformer_output = self.model(smiles)
        output = self.regression_head(transformer_output)
        
        return output

def _load_pre_trained_weights(model, mode = 'cgcnn'):
    """
    Taken from this repository: https://github.com/zcao0420/MOFormer/blob/main/finetune_transformer.py
    Edited to include other pretrained weights
    """
    try:
        # checkpoints_folder = os.path.join(self.config['fine_tune_from'], 'checkpoints')
        #checkpoints_folder = 'SSL/pretrained/transformer'
        checkpoints_folder = 'SSL/pretrained/cgcnn'
        if mode == 'geometric':
            checkpoints_folder = 'SSL/pretrained/geometric'

        elif mode == 'cgcnn':
            checkpoints_folder = 'SSL/pretrained/transformer'
        
        else:
            checkpoints_folder = 'SSL/pretrained/None'

        load_state = torch.load(os.path.join(checkpoints_folder, 'model_t_50.pth'),  map_location=config['gpu']) 
        model_state = model.state_dict()

        for name, param in load_state.items():
            if name not in model_state:
                print('NOT loaded:', name)
                continue
            else:
                print('loaded:', name)
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            model_state[name].copy_(param)
        print("Loaded pre-trained model with success.")
    except FileNotFoundError:
        print("Pre-trained weights not found. Training from scratch.")

    return model

tokenizer = MOFTokenizer("MOFormer_modded/tokenizer/vocab_full.txt")
config = yaml.load(open("MOFormer_modded/config_ft_transformer.yaml", "r"), Loader=yaml.FullLoader)
config['dataloader']['randomSeed'] = 0

if torch.cuda.is_available() and config['gpu'] != 'cpu':
    device = config['gpu']
    torch.cuda.set_device(device)
    config['cuda'] = True

else:
    device = 'cpu'
    config['cuda'] = False
print("Running on:", device)

transformer_SMILES = Transformer(**config['Transformer'])
model_pre = _load_pre_trained_weights(model = transformer_SMILES, mode = 'cgcnn')
model = RegressionTransformer(model = model_pre)

model.load_state_dict(torch.load('model_ft_bandgap.pth'))
model.to(device)
#model = torch.load('full_model_ft_bandgap.pth') #loads finetuned model on band gap for QMOF dataset (done on ~400 data points)

def predictBandGap(smiles):
    model.eval()
    token = np.array([tokenizer.encode(smiles, max_length=512, truncation=True,padding='max_length')])
    token = torch.from_numpy(np.asarray(token))

    token = token.to(device)
    return model(token)

#Example use: predictBandGap('[Zn]12.OC(=O)C1=CC=C(C=C1)C(O2)=O', model)
#should return: tensor([[4.0581]], device='cuda:0', grad_fn=<AddmmBackward0>), can just do predictBandGap(smiles, model).item() to extract band gap