from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import ipdb

# Can make things float 16 if memory issues

class Transformer_Model(nn.Module):

    def __init__(self,config:dict):
        super().__init__()
        self.config = config
        
        self.projection = nn.Linear(in_features=self.config['in_features'], 
            out_features=self.config['out_features'],bias=False)
        
        self.encoder = nn.TransformerEncoderLayer(d_model=self.config['in_features'],
            nhead=self.config['nhead'],activation=self.get_activation(),
            batch_first=True)
        
        self.encoder_stack = nn.TransformerEncoder(encoder_layer=self.encoder, 
            num_layers=self.config['num_encoder_layers'])
        
        self.classifier = nn.Linear(in_features=self.config['in_features'],
            out_features=2)
        
        self.softmax = nn.Softmax(dim=1)

        self.emb = nn.Embedding(num_embeddings=2, embedding_dim=self.config['in_features'])
        self.cls = torch.tensor(0).to("cuda")
        self.sep = torch.tensor(1).to("cuda") #This is not how this should be done 

    def get_activation(self):
        if self.config['activation'] == 'selu':
            return F.selu
        elif self.config['activation'] == 'relu':
            return F.relu
        elif self.config['activation'] == 'gelu':
            return F.gelu
        else:
            return None

    def forward(self,x1,x2,padd_mask):# The input is in a batch
        # ["CLS", X1,"SEP",X2]
        # Have to padd such that x1 and x2 have same number of paragraph 
        # X1 [N , 80 , 768] X2 [N , 80 , 768] P1 and P2 will be same for all the paragraphs in that batch and that side
        # Order is also fixed the query case will be the first and the precedent side will be the later
        # ipdb.set_trace()
        cls_emb = self.emb(self.cls)
        sep_emb = self.emb(self.sep)
        # ipdb.set_trace()
        cls_out = cls_emb.unsqueeze(0).unsqueeze(0).repeat((x1.shape[0],1,1))
        sep_emb = sep_emb.unsqueeze(0).unsqueeze(0).repeat((x2.shape[0],1,1))
        # ipdb.set_trace()
        cls_x1 = torch.cat((cls_out,x1),dim=1)
        sep_x2 = torch.cat((sep_emb,x2),dim=1)
        # ipdb.set_trace()
        x = torch.cat((cls_x1,sep_x2),dim=1)
        # ipdb.set_trace()
        x = self.projection(x)
        x = self.encoder_stack(x,src_key_padding_mask=padd_mask)
        cls_emb = x[:,0,:]
        x = self.classifier(cls_emb)
        x = self.softmax(x)
        # print(x)        
        return x

if __name__=='__main__':
    print("Testing model.py")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    config = yaml.safe_load(open('model/model_config.yaml','r'))
    print(config)

    model = Transformer_Model(config=config)
    model.to(device)

    temp_input_1 = torch.randn((config['batch_size'],20,config['in_features'])).to(device)
    temp_input_2 = torch.randn((config['batch_size'],13,config['in_features'])).to(device)
    
    # print(temp_input.shape)
    
    model(temp_input_1,temp_input_2)