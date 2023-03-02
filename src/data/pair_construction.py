from pathlib import Path
from tqdm import tqdm

import networkx as nx
import numpy as np
import json
import yaml
import re
import pickle
import csv


def dict2json(judg,path):
    file = open(path,'w')
    json.dump(judg,file)
    file.close()

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

def get_dist_2(source):
    out = list(nx.bfs_edges(G, source=source, depth_limit=2))
    pos = [ x for x in out if x[0]==source ]
    neg = [ (source,x[1]) for x in out if x[0]!= source ]
    return pos,neg

def get_triplet(source):
    out = list(nx.bfs_edges(G, source=source, depth_limit=2))
    dist1 =  set([ x[0] for x in out if x[0]!=source ])
    triplets = [ (source,x[0],x[1]) for x in out if x[0] in dist1 ]
    return triplets

if __name__=='__main__':
    x = get_gpath_dict()
    paths = get_all_paths(x['path_clean'])

    out = list()
    for path in tqdm(paths,desc="Link extraction"):
        file = open(path,'r')
        judg = json.load(file)
        out.append([path.name.split('.')[0],judg['citation_sc']]) 
        file.close()

    nodes = [x[0] for x in out]
    edges = []
    for x in tqdm(out,desc="edges"):
        for y in x[1]:
            edges.append((x[0],y))
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    IRLED_map = open('../data/maps.txt')
    text = IRLED_map.read()
    idx = re.findall('\d+.json',text)
    idx = set([ x.split('.')[0] for x in idx])

    AILA_2019 = open('../../para_based_legal_search/data/raw/query2indiankanoon.txt','r')
    text = AILA_2019.read()
    aila_ids = set([ x.split()[-1]  for x in  text.split('\n')])
    for x in aila_ids:
        if x not in 'NAN':
            idx.add(x)

    triplets = []
    for node in tqdm(nodes,desc='Making constrastive pairs'):
        # print(node)
        if node not in idx:
            triplets.extend(get_triplet(node))

    triplets_lines = [ x[0]+' , '+x[1]+' , '+x[2]+'\n' for x in triplets]    

    file_triplets = open('../data/triplets.csv','w')
    file_triplets.writelines(triplets_lines)
    file_triplets.close()
