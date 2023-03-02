from tqdm import tqdm

import torch
import pickle


if __name__=='__main__':

    
    final = dict()
    out = [ tuple(x.strip('\n').split(' , ')) for x in tqdm(open('../data/triplets.csv').readlines(),desc= 'Loading Triplets')]
    
    unique_keys = set()
    for x in out:
        unique_keys.add(x[0])
        unique_keys.add(x[1])
        unique_keys.add(x[2])
    print(len(unique_keys))
    emb = pickle.load(open('../data/processed/InLegalBERT_emb.pkl','rb'))

    for key in tqdm(unique_keys,desc="Tensor Conversion"):
        if len(emb[key]) != 0:
            final[key] = torch.stack(emb[key]).squeeze(dim=1)
    pickle.dump(obj=final, file=open('../data/processed/InLegalBERT_emb_new.pkl','wb'))