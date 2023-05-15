import ipdb
import sys
sys.path.append('/home2/sisodiya.bhoomendra/github/contrastive_learning/src')
from model.model import Transformer_Model
from model.dataset import JudgementDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryF1Score,BinaryPrecision,BinaryRecall


import torch
import yaml
import wandb


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


def train(config):
    # Which model
    # Model config
    # Traning parameters 
    # Learning rate , loss , learning rate schedular , optimizer , batch size , number of epochs
    # Early stopping and checkpointing
    # Logging
    # Define collate function in the dataloader
    device = 'cuda'
    wblog = wandb.init(project="Precedencer Classification Trasfomer",config=config)
    train_dataset = JudgementDataset(config=config['dataset'],which='train')
    valid_dataset = JudgementDataset(config=config['dataset'],which='valid')
    # test_dataset = JudgementDataset(config=config['dataset'],which='test')
    # ipdb.set_trace()
    print(f"Train size: {len(train_dataset)} , Valid size: {len(valid_dataset)} ")#, Test size: {len(test_dataset)}")
    

    train_dataloader = DataLoader(dataset=train_dataset,batch_size=config['model']['batch_size'],shuffle=True,collate_fn=pad_batch)
    # test_dataloader = DataLoader(dataset=test_dataset,batch_size=config['model']['batch_size'],collate_fn=pad_batch)
    valid_dataloader = DataLoader(dataset=valid_dataset,batch_size=config['model']['batch_size'],collate_fn=pad_batch)

    model =  Transformer_Model(config=config['model']).to(device)

    loss = torch.nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(params=model.parameters(),lr=config['train']['lr'])
    metric = BinaryF1Score()
    precision = BinaryPrecision()
    recall = BinaryRecall()
    schedular = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optim,T_0=100) #torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim,patience=1)

    # (x1,x2,labels,masks) = next(iter(valid_dataloader))
    # out = model(x1,x2,masks)
    # ipdb.set_trace()
    # print(f"Starting Learning Rate : {schedular._last_lr}")
    best_f1 = 0
    for epoch in range(config['train']['epochs']):
        avg_train_loss = 0
        for batch,(x1,x2,labels,masks) in tqdm(enumerate(train_dataloader),desc=f"Traning Epoch {epoch+1}",total=len(train_dataloader)):
            x1 = x1.to(device)
            x2 = x2.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            # ipdb.set_trace()
            y_pred = model(x1,x2,masks)
            train_loss = loss(y_pred,labels)
            optim.zero_grad()
            train_loss.backward()
            optim.step()
            avg_train_loss += train_loss.item()
            # y_pred[y_pred>0.5] = 1
            # y_pred[y_pred<=0.5] = 0
            if (batch+1)%300 == 0:
                print(f"Batch No. {batch+1}  the average train loss {avg_train_loss/(batch+1)}")

            binary_id = torch.argmax(y_pred,dim=1).to('cpu')
            groud_truth = torch.argmax(labels,dim=1).to('cpu')
            # ipdb.set_trace()
            metric.update(binary_id,groud_truth)
            precision.update(binary_id,groud_truth)
            recall.update(binary_id,groud_truth)
            schedular.step(epoch*len(train_dataloader)+batch)


        train_f1 = metric.compute()
        train_recall = recall.compute()
        train_precision = precision.compute()
        train_loss = avg_train_loss/len(train_dataloader)

        print(f"Avergae Train Loss {train_loss}")
        print(f"F1 train : {train_f1}")
        print(f"Precision train : {train_precision}")
        print(f"Recall train : {train_recall}")

        metric.reset()
        precision.reset()
        recall.reset()

        with torch.no_grad():
            val_loss = 0
            for batch,(x1,x2,labels,masks) in tqdm(enumerate(valid_dataloader),desc=f"Validation Epoch {epoch+1}",total=len(valid_dataloader)):
                x1 = x1.to(device)
                x2 = x2.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                y_pred = model(x1,x2,masks)
                val_loss += loss(y_pred,labels).item()

                binary_id = torch.argmax(y_pred,dim=1).to('cpu')
                groud_truth = torch.argmax(labels,dim=1).to('cpu')
                metric.update(binary_id,groud_truth)
                precision.update(binary_id,groud_truth)
                recall.update(binary_id,groud_truth)

            valid_f1 = metric.compute()
            valid_recall = recall.compute()
            valid_precision = precision.compute()
            valid_loss = val_loss/len(valid_dataloader)
            
            print(f"Avergae Validation Loss {val_loss/len(valid_dataloader)}")
            print(f"F1 Validation: {valid_f1}")
            print(f"Precision train : {valid_recall}")
            print(f"Recall train : {valid_loss}")

            if best_f1<valid_f1:
                torch.save(obj = {'config':config['model'], 'model':model.state_dict()}, f='../data/model/model_1.pth')
                best_f1 = valid_f1
                print(f"\n\nCHECKPOINT REACHED: BEST VALIDATION F1 SCORE: {best_f1}\n\n")
                

            


            metric.reset()
            precision.reset()
            recall.reset()
            print(f"Updated Learning Rate : {schedular._last_lr[0]}")
            
            wandb.log({
            "Train_F1":train_f1,
            "Train_Precision":train_precision,
            "Train_recall":train_recall,
            "Train_Avg_error":train_loss,
            "Validation_F1":valid_f1,
            "Validation_Precision":valid_precision,
            "Validation_recall":valid_recall,
            "Validation_Avg_error":valid_loss,
            })
    # how to get data from dataloader
    # print(len(train_dataloader))
    # for batch , (X1,X2,labels,masks) in enumerate(train_dataloader):
    #     print(batch,X1.shape,X2.shape,labels.shape,masks.shape)
    # # ipdb.set_trace()
    # print(len(valid_dataloader))
    # for batch , (X1,X2,labels,masks) in enumerate(valid_dataloader):
    #     print(batch,X1.shape,X2.shape,labels.shape,masks.shape)
    # print(len(test_dataloader))
    # for batch , (X1,X2,labels,masks) in enumerate(test_dataloader):
    #     print(batch,X1.shape,X2.shape,labels.shape,masks.shape)
if __name__=='__main__':
    config = yaml.safe_load(open('train/train_config.yaml','r'))
    print(config)
    train(config)
