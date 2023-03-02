from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from pathlib import Path

import argparse
import yaml
import json

def get_all_paths(path):
    print("Collecting all paths")
    paths = []
    for fpath in path.iterdir():
        paths.append(fpath)
    return paths

def get_gpath_dict():
    paths = yaml.safe_load(open('/home2/sisodiya.bhoomendra/github/contrastive_learning/data/paths.yaml','r'))
    posix_path = {}
    for k,v in paths.items():
        posix_path[k] = Path(v)
    return posix_path


def judg2text():
    pass

def query2text():
    pass

def main(dpaths,qpaths):

    dnames = []
    dtext = []

    for path in tqdm(dpaths,desc='All judgement Loading'):
        file = open(path,'r')
        judg = json.load(file)
        file.close()
        actual_paras = judg.pop('headnote')
        actual_paras.extend(judg['paragraphs'])
        out = ' '.join(actual_paras).strip()
        # print(out)
        dnames.append(path.name)
        dtext.append(out.lower())

    qnames = []
    qtext = []

    for path in tqdm(qpaths,desc='Current Cases Loading'):
        try:    
            file = open(path,'r')
            qtext.append(file.read().lower())
            qnames.append(path.name)
            file.close()
        except:
            print(f"Exception: {path} is opened in rb")
            file = open(path,'rb')
            qnames.append(path.name)
            qtext.append(str(file.read()).lower())
            file.close()

    vct = TfidfVectorizer()
    dout = vct.fit_transform(dtext)
    qout = vct.transform(qtext)

    print(dout.shape,qout.shape)
    simi  = qout.dot(dout.T) # 200 * x
    idxs = simi.argmax(axis=1)
    # print(idxs)
    idxs = idxs.flatten().tolist()[0]
    matching = []
    for i in range(len(qnames)):
        print(idxs[i])
        matching.append((qnames[i],dnames[idxs[i]]))
    print(matching)

    breakpoint()

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',help='Get in to debug mode',action='store_true')
    parser.add_argument('--idx',help='index of file that will be cleaned',type=int)
    args = parser.parse_args()
    gpath = get_gpath_dict()
    dpaths = get_all_paths(gpath['path_clean'])
    qpaths = get_all_paths(Path('../../Judgment_retrival_Fire_2017/data/raw/Current_Cases/'))
    if args.debug:
        # print(qpaths)
        main(dpaths, qpaths)