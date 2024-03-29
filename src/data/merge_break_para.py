from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

import argparse
import yaml
import spacy
import json

"""
This is still not perfect because some warnings are ther i.e length is exeding the maximum limit but very few
"""

def dict2json(judg,path,num):
    file = open(path,'w')
    json.dump(judg,file)
    file.close()
    print(f"Done {num} : {path}")


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

def break_para(para,nlp,limit):
    """
    Breaks the para in it has more that 400 words
    para: string with multuple sentences
    nlp: spacy en_core_web_md used for sentence segmentation
    """
    num_words = len(para.split(' '))
    
    if num_words > limit:
        # print(num_words)
        dpara = nlp(para)
        new_paras = []
        temp_para = ''
        count = 0
        for idx,sent in enumerate(dpara.sents):
            # breakpoint()
            temp_para += sent.text + ' '

            count += len(sent.text.split(' '))
            if count >= limit:
                new_paras.append(temp_para.strip())
                # print(f"{len(new_paras)} --->##{len(new_paras[-1].split(' '))}##-- {new_paras[-1]}\n\n" )
                count = 0
                temp_para = ''
        
        if count>0:
            new_paras.append(temp_para.strip())
            count = 0
            temp_para = ''
            # print(f"{len(new_paras)} --->##{len(new_paras[-1].split(' '))}##-- {new_paras[-1]}\n\n" )
        # if some paragraph are still greate then limit the we will split them 
        final_paras = []
        for para in new_paras:
            if len(para.split()) > limit:
                out = ''
                for idx,word in enumerate(para.split()):
                    out += word + ' '
                    if (idx+1)%limit == 0:
                        final_paras.append(out.strip())
                        out = ''
            else:
                final_paras.append(para)

        return final_paras
            # sents.append(sent)
        
    return [para]
    # print(sents)

def merge_paras(para_list,limit):

    lengths = [len(para.split(' ')) for para in para_list]
    pointer_one = 0
    pointer_two = 0
    temp_limit = 0
    new_paras = []

    while(pointer_one<len(lengths)):
        if lengths[pointer_one] < limit:
            pointer_two =pointer_one
            total = lengths[pointer_two]
            temp_para = ''

            while(pointer_two<len(lengths) and total < limit):
                temp_para += para_list[pointer_two] + ' '
                pointer_two+=1
                if pointer_two ==len(lengths):
                    break
                total += lengths[pointer_two]
                

            pointer_one = pointer_two
            new_paras.append(temp_para.strip())
        else:
            new_paras.append(para_list[pointer_one])
            pointer_one+=1

    return new_paras


def show_paras(para_list):

    for idx,para in enumerate(para_list):
        print(f"{idx+1} --->##{len(para.split(' '))}##-- {para}\n\n")

def check_paras(para_list,limit,num):

    for idx,para in enumerate(para_list):
        length = len(para.split(' '))
        if length > limit:
            print(f"WARNING: LENGTH EXCESSED {limit} {length}")


def paralength_fixing(x):
    """
    Need to include the paragraph lenghts
    """
    path,nlp,limit,num = x

    file = open(path,'r')
    judg = json.load(file)
    file.close()
    actual_paras = judg.pop('headnote')
    actual_paras.extend(judg['paragraphs'])
    out = dict()
    merged_para = merge_paras(para_list=actual_paras, limit=limit)
    new_paras = []
    for para in merged_para:
        new_paras.extend(break_para(para=para,nlp=nlp,limit=limit))
    # merged_para = merge_paras(para_list=new_paras, limit=limit)
    out['paragraphs'] = new_paras
    # show_paras(para_list=judg['paragraphs'])
    check_paras(para_list=out['paragraphs'],limit=limit,num=num)
    dict2json(judg=out, path=f'../data/processed/bert_ready/{path.name}',num=num)


def combine():
    final = dict()
    for path in tqdm(Path('../data/processed/bert_ready/').iterdir(),desc='Combining Different outputs'):
        file = open(path)
        out = json.load(file) 
        final[path.name.split('.')[0]] = out['paragraphs']
        file.close()
    dict2json(judg=final, path='../data/processed/judg2para.json', num="ALL DONE")

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',help='Get in to debug mode',action='store_true')
    parser.add_argument('--idx',help='index of file that will be cleaned',type=int)
    args = parser.parse_args()
    gpath = get_gpath_dict()
    paths = get_all_paths(gpath['path_clean'])
    nlp = spacy.load("en_core_web_sm")
    limit = 400

    inputs = []
    for idx,path in enumerate(paths):
        inputs.append([path,nlp,limit,idx])

    if args.debug:
        print("Started")
        paralength_fixing(x=inputs[args.idx])
    else:
        n_cores = 30
        print("Number of cpus: ",n_cores)
        print("Preparing Inputs to Map")
        
        print("Parallel processing started")

        with Pool(n_cores) as p:
            p.map(paralength_fixing,inputs)

        combine()
        print("#"*10 +" Completed "+ "#"*10)
