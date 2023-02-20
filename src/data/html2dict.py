import yaml
from bs4 import BeautifulSoup
from pathlib import Path
import re
import json
from tqdm import tqdm
from multiprocessing import Pool

# How to properly combine law data with this data

def dict2json(judg,path):
    file = open(path,'w')
    json.dump(judg,file)
    file.close()
    print(f"Done: {path}")

def get_gpath_dict():
    paths = yaml.safe_load(open('/home2/sisodiya.bhoomendra/github/contrastive_learning/data/paths.yaml','r'))
    posix_path = {}
    for k,v in paths.items():
        posix_path[k] = Path(v)
    return posix_path

def clean_text(para):
    """
    Given a string input return a clearn string 
    1. Replaces tabs next lines and as spaces
    2. Combines Multiple spaces to 1
    3. Removing Punctuation
    3. Makes everthing in lower case
    """
    para = para.replace("\t",' ')
    para = para.replace('\n',' ')# Replacing tabs and next line with spaces
    # para = re.sub('[^\w\s]','',para)# Replacing puncation with space
    para = re.sub(' +',' ',para)
    para = para.lower()
    return para

def to_text(doc):
    if doc is not None:
        return doc.text
    return ''

def get_doc_id_from_a_tag(lk):
    return lk.get('href').split('/')[-2]

def get_doc_name_from_a_tag(lk):
    # print(lk.text)
    return lk.text

def html2dict(x):
    path,name,jids = x
    soup_jud = BeautifulSoup(open(path,'r'),features="html.parser")
    title = to_text(soup_jud.find('div',{"class": "doc_title"}))
    eq_citation = to_text(soup_jud.find('div',{'class':'doc_citations'}))# may or maynot exit
    author =  to_text(soup_jud.find('div',{'class':'doc_author'}))# may or maynot exit
    bench = to_text(soup_jud.find('div',{'class':'doc_bench'}))# may or maynot exit
    meta_data = []
    # Head note should also come in paragraph so if it is pre the remove it and try to add into paragraph
    head_note = []
    paragraphs = []
    for pre in soup_jud.find_all('pre'):
        data = to_text(pre).lower()
        a = data.split('headnote')
        meta_data.append(a[0])
        if len(a)==2:
            if len(a[1]) > 100:
                head_note.append(clean_text(a[1]))

    for blockquote in soup_jud.find_all('blockquote'):
        if blockquote is not None:
            paragraphs.append(clean_text(blockquote.text))
    
    judgment_citations = set() # Preform intersection with Links
    other_citations = set()

    for para in soup_jud.find_all('p'):
        if para is not None:
            paragraphs.append(clean_text(para.text))

    for link in soup_jud.find_all('a'):
        if link is not None:
            doc_id = get_doc_id_from_a_tag(link)
            if doc_id in jids:
                judgment_citations.add(doc_id)
            else:
                other_citations.add((doc_id,clean_text(get_doc_name_from_a_tag(link))))
    other_citations = list(other_citations)
    judgment_citations = list(judgment_citations)
    
    judg = {"title":title ,
            "author":author,
            "eq_citation":eq_citation,
            "bench":bench,
            'meta_data':meta_data,
            "headnote":head_note,
            "paragraphs":paragraphs,
            "citation_sc":judgment_citations,
            "citation_others":other_citations}
    path = f'../data/processed/processed_judgements/{name}.json'
    dict2json(judg,path)

if __name__=='__main__':
    gpath = get_gpath_dict()
    jids = set()
    names = []
    jpaths = []
    n_cores = 10
    print("Number of cpus: ",n_cores)
    print("Preparing Inputs to Map")
    for paths in tqdm(gpath['path_html_judg'].iterdir(),desc="Prep"):
        name = paths.name.split('.')[0]
        jids.add(name)
        names.append(name)
        jpaths.append(paths)
    inputs = []
    for x,y in tqdm(zip(jpaths,names),desc='Combine'):
        inputs.append((x,y,jids))
    print("Input Prep Completed")
    print("Parallel processing started")
    with Pool(n_cores) as p:
        print(p.map(html2dict,inputs))

    