from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import json
import numpy as np
import yaml


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

def paragraph_number(paths):
    para_nums = []
    for path in tqdm(paths,desc="Paras:"):
       para_nums.append(len(json.load(open(path,'r'))['paragraphs']))
    para_nums = np.array(para_nums)
    para_nums =  para_nums[para_nums<100]
    plt.hist(para_nums,bins=50)
    plt.savefig('para_nums')
    
    for x in range(85,101,1):
        print(f" Percentile {x} No. of paragraph {np.percentile(para_nums, q=x)}")
    # breakpoint()

if __name__=='__main__':
    gpath = get_gpath_dict()
    paths = get_all_paths(gpath['path_clean'])
    paragraph_number(paths)
    