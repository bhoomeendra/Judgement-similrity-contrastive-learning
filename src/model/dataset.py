
# Positive negative pairs files taking only 1:2 
# Return tensor and labels ==> ( x1, x2, label)
from torch.utils.data import Dataset
from tqdm import tqdm

import torch
import random
import pickle
import yaml
import ipdb

class JudgementDataset(Dataset):

    def __init__(self,config,which:str):
        # Load all the judgement matrix from bert
        # Load all pos and neg pair files
        # Ratio of pos and negs will come from config
        # Which embeddings to use bert sbert etc from config
        self.config = config
        random.seed(42)
        self.triplets = [ tuple(x.strip('\n').split(' , ')) for x in tqdm(open('../data/triplets.csv').readlines(),desc= 'Loading Triplets')] 
        self.which  = which
        self.embeddings = pickle.load(open(f"../data/processed/{self.config['embedding']}",'rb'))
        # ipdb.set_trace()
        self.jud_id = list(set( [x[0] for x in tqdm(self.triplets,desc='ids extractions') 
            if self.embeddings[x[0]].shape[0] <=config['limit']  ] ) )
        random.shuffle(self.jud_id)
        total_len = len(self.jud_id)
        t = int(config['train_split']*total_len)
        v =  int(config['val_split']*total_len)

        self.train_jid = set(self.jud_id[:t])
        self.val_jid = set(self.jud_id[t:t+v])
        self.test_jid = self.jud_id[t+v:]
        
        if self.which == 'train':
            self.train_triples = [ x  for x in tqdm(self.triplets,desc=f"{self.which} Split") 
            if x[0] in self.train_jid and x[1] in self.train_jid and x[2] in self.train_jid]
        elif self.which == 'valid':
            self.val_triples = [ x  for x in tqdm(self.triplets,desc=f"{self.which} Split") 
            if x[0] in self.val_jid and x[1] in self.val_jid and x[2] in self.val_jid ]
        elif self.which == 'test':
            self.test_triples = [ x for x in tqdm(self.triplets,desc=f'{self.which} Split') 
            if x[0] in self.test_jid and x[1] in self.test_jid and x[2] in self.test_jid ]
        else:
            print(f"ERROR: {self.which} is not a correct split")

        # This is not a good idea to keep this here because for each train ,vail and test the file will be loaded 3 times
        # breakpoint()


    def get_tensor(self,jid1,jid2,label):
        # breakpoint()
        # Stack should 
        return torch.tensor(self.embeddings[jid1]),torch.tensor(self.embeddings[jid2]),torch.tensor(label,dtype=torch.float)
    def __getitem__(self,idx):

        if idx%2 == 0:# Positive Pair
            idx = idx//2
            if self.which == 'train':
                return self.get_tensor(self.train_triples[idx][0] , self.train_triples[idx][1] , [0,1])
            elif self.which == 'valid':
                return self.get_tensor(self.val_triples[idx][0] , self.val_triples[idx][1] , [0,1])
            elif self.which == 'test':
                return self.get_tensor(self.test_triples[idx][0] , self.test_triples[idx][1] , [0,1])
        else: # Negative Pairs
            idx = idx//2
            if self.which == 'train':
                return self.get_tensor(self.train_triples[idx][0] , self.train_triples[idx][2] , [1,0])
            elif self.which == 'valid':
                return self.get_tensor(self.val_triples[idx][0] , self.val_triples[idx][2], [1,0])
            elif self.which == 'test':
                return self.get_tensor(self.test_triples[idx][0] , self.test_triples[idx][2], [1,0])

    def __len__(self):
        if self.which == 'train':
            return 2*len(self.train_triples)
        elif self.which == 'test':
            return 2*len(self.test_triples)
        elif self.which == 'valid':
            return 2*len(self.val_triples)


if __name__=='__main__':

    config = yaml.safe_load(open('model/dataset_config.yaml'))
    print(config)
    dataset = JudgementDataset(config=config, which="valid")
    breakpoint()