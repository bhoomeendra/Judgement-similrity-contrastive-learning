from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

import torch
import numpy as np
import json
import pickle


def get_gpath_dict():
    paths = yaml.safe_load(open('/home2/sisodiya.bhoomendra/github/contrastive_learning/data/paths.yaml','r'))
    posix_path = {}
    for k,v in paths.items():
        posix_path[k] = Path(v)
    return posix_path


def get_all_paths(path):
    print("Collecting all paths")
    paths = []
    for fpath in path.iterdir():
        paths.append(fpath)
    return paths


if __name__=='__main__':

    model_name = "law-ai/InLegalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to('cuda')
    
    file = open('../data/processed/judg2para.json','r')
    judgments_para = json.load(file)
    file.close()
    final = dict()

    out = [ tuple(x.strip('\n').split(' , ')) for x in tqdm(open('../data/triplets.csv').readlines(),desc= 'Loading Triplets')]
    
    unique_keys = set()
    for x in out:
        unique_keys.add(x[0])
        unique_keys.add(x[1])
        unique_keys.add(x[2])

    for key in tqdm(unique_keys,desc=f'Embedding from {model_name}'):
        paragraphs = judgments_para[key]
        mat = []
        for paragrah in paragraphs:
            encoded_input = tokenizer(paragrah, padding=True,truncation=True, return_tensors="pt").to('cuda')
            output = model(**encoded_input)
            sentence_embb = output.pooler_output.squeeze(dim=1)# as the batch size is one
            mat.append(sentence_embb.detach())
        final[key.split('.')[0]] = torch.stack(mat)
    file = open(f"../data/processed/{model_name.split('/')[-1]}_emb.pkl",'wb')
    pickle.dump(obj=final, file=file)
    file.close()
    print("Completed")