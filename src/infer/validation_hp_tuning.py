import ipdb
import sys
sys.path.append('/home2/sisodiya.bhoomendra/github/contrastive_learning/src')

from model.model import Transformer_Model
from model.dataset import JudgementDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryF1Score,BinaryPrecision,BinaryRecall,BinaryConfusionMatrix
import torch
import numpy as np


def pad_batch(x):
    # X is a list of datapoints what we get from Judgement Dataset
    # Size of the list is equal to batch size
    # We will just take the judgements which ha
    # Added padded
    padd = torch.zeros((1,768))
    labels = []
    d1 = []
    d2 = []
    masks = []
    limit = 80
    for v in x:
        x1,x2,l = v # X1 (C,768) X2 (M,768) should become (80,768)
        padd_mask = torch.zeros(162,dtype=torch.bool).to(x1.device)
        nums_padd_1 = limit - x1.shape[0]
        nums_padd_2 = limit - x2.shape[0]
        out1 = torch.cat((x1,padd.repeat((nums_padd_1,1)).to(x1.device)), dim=0)
        out2 = torch.cat((x2,padd.repeat((nums_padd_2,1)).to(x2.device)),dim=0)
        padd_mask[nums_padd_1:limit]  = 1
        padd_mask[limit+1+nums_padd_2:] = 1
        masks.append(padd_mask)
        d1.append(out1)
        d2.append(out2)
        labels.append(l)
    x1 = torch.stack(d1)
    x2 = torch.stack(d2)
    labels = torch.stack(labels)
    masks = torch.stack(masks)
    return x1,x2,labels,masks

def infer(model_path):
    test_dataset = JudgementDataset( config={ 'embedding': 'InLegalBERT_emb_new.pkl',
                  'limit': 80,
                  'train_split': 0.6,
                  'val_split': 0.15},which='valid')

    test_dataloader = DataLoader(dataset=test_dataset,batch_size=128,collate_fn=pad_batch)
    model_details = torch.load(model_path)
    model = Transformer_Model(model_details['config']).to('cuda')
    model.load_state_dict(model_details['model'])
    model.eval()
    
    loss = torch.nn.CrossEntropyLoss()
    f1 = BinaryF1Score()
    precision = BinaryPrecision()
    recall = BinaryRecall()
    confusion = BinaryConfusionMatrix()
    test_loss = 0

    pred_labels = []
    true_labels = []

    for batch,(x1,x2,labels,masks) in tqdm(enumerate(test_dataloader),desc="Test Desc",total=len(test_dataloader)):
        y_pred = model(x1,x2,masks)
        test_loss += loss(y_pred,labels).item()
        # ipdb.set_trace()        
        pred_labels.append(y_pred[:,1].detach().cpu())
        true_labels.append(labels[:,1].detach().cpu())
        binary_id = torch.argmax(y_pred,dim=1).to('cpu')
        groud_truth = torch.argmax(labels,dim=1).to('cpu')

        f1.update(binary_id,groud_truth)
        precision.update(binary_id,groud_truth)
        recall.update(binary_id,groud_truth)
        confusion.update(binary_id,groud_truth)
    test_f1 = f1.compute()
    test_recall = recall.compute()
    test_precision = precision.compute()
    test_loss = test_loss/len(test_dataloader)


    print(f"Avergae Train Loss {test_loss}")
    print(f"F1 train : {test_f1}")
    print(f"Precision train : {test_precision}")
    print(f"Recall train : {test_recall}")
    print(f"Confusion Matrix {confusion.compute()}")

    # ipdb.set_trace()
    pred_labels = torch.cat(pred_labels)
    true_labels = torch.tensor(torch.cat(true_labels),dtype=torch.int16)
    
    # ipdb.set_trace()
    thetas = np.array([i for i in range(1,100)])/100
    for theta in thetas:
        f1.reset()
        precision.reset()
        recall.reset()
        confusion.reset()
        f1.update(pred_labels>theta,true_labels)
        # ipdb.set_trace()
        precision.update(torch.tensor(pred_labels>theta,dtype=torch.int16),true_labels)
        recall.update(torch.tensor(pred_labels>theta,dtype=torch.int16),true_labels)
        confusion.update(torch.tensor(pred_labels>theta,dtype=torch.int16),true_labels)
        print(f"Threshold: {theta} \n F1 score : {f1.compute()} \n Precision : {precision.compute()} \n Recall : {recall.compute()} \n Confusion matrix : {confusion.compute()}")
    # ipdb.set_trace()

if __name__=='__main__':
    infer(model_path='../data/model/model_0.80.pth')