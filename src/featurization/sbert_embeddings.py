from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import numpy as np
import json
import pickle
import ipdb


if __name__=='__main__':

    model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    print(f"Loading Model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"Loading Judgements ")
    file = open('../data/processed/judg2para.json','r')
    judgments_para = json.load(file)
    file.close()
    final = dict()
    print(f"Extracting judgment Ids")
    out = [ tuple(x.strip('\n').split(' , ')) for x in 
            tqdm(open('../data/triplets.csv').readlines(),
                desc= 'Loading Triplets')]
    
    unique_keys = set()
    for x in out:
        unique_keys.add(x[0])
        unique_keys.add(x[1])
        unique_keys.add(x[2])
    unique_keys = list(unique_keys)

    for key in tqdm(unique_keys,desc=f'Embedding from {model_name}'):
        paragraphs = judgments_para[key]
        mat = []
        for paragraph in paragraphs:
            output = model.encode(paragraph)
            mat.append(output)

        if len(mat) == 0:
            final[key.split('.')[0]] = np.zeros(shape=(1,768))
        else:
            final[key.split('.')[0]] = np.stack(mat)
    
    file = open(f"../data/processed/{model_name.split('/')[-1]}_emb.pkl",'wb')
    pickle.dump(obj=final, file=file)
    file.close()
    print("Completed")