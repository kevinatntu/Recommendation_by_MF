import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import sys
import pickle
import json

from model import MF
from tqdm import tqdm

def preprocessing(path):
    '''
    Load train.csv
    '''
    data = pd.read_csv(path)
    data['ItemId'] = data['ItemId'].str.split()
    maxID = data['UserId'].max()
    maxDocID = -1

    for idx, row in data.iterrows():
        lstMax = max(list(map(int, row['ItemId'])))
        maxDocID = max(maxDocID, lstMax)
    
    maxID += 1
    maxDocID += 1

    matrix = np.zeros((maxID, maxDocID), dtype=int)
    pos_item = {}

    # first try: Do not consider the 'time' factor
    for idx, row in data.iterrows():
        pos_item[row['UserId']] = []
        for item in list(map(int, row['ItemId'])):
            matrix[row['UserId'], item] = 1
            pos_item[row['UserId']].append(item)

    return maxID, maxDocID, matrix, pos_item

'''
For Training
'''
class MFDataset(Dataset):
    def __init__(self, user_idx_lst, num_items, pos_item_dic, neg_sample_ratio=0.5):
        #self.matrix = matrix
        self.data = []
        self.neg_item_dic = {}
        self.test_pos = []
        self.pos_item_dic = pos_item_dic.copy()
        self.num_items = num_items

        self.test_pos = {}

        for i in user_idx_lst:
            num_pos = len(self.pos_item_dic[i])

            # cut
            self.pos_item_dic[i] = random.sample(pos_item_dic[i], int(num_pos*0.9))
            num_pos = len(self.pos_item_dic[i])
            self.test_pos[i] = list(set(pos_item_dic[i]) - set(self.pos_item_dic[i]))
            
            # neg
            num_neg = int(neg_sample_ratio * num_pos)
            neg_item = list(set(range(num_items)) - set(self.pos_item_dic[i]))
            neg_sample_item = random.sample(neg_item, num_neg)
            self.neg_item_dic[i] = neg_sample_item
            
            if neg_sample_ratio <= 1.0:
                for idx, j in enumerate(self.pos_item_dic[i]):
                    self.data.append([i, j, neg_sample_item[idx%num_neg]])
            else:
                for idx, j in enumerate(neg_sample_item):
                    self.data.append([i, self.pos_item_dic[i][idx%num_pos], j])
            
            #self.data.append(this_user)

        #self.data = np.asarray(self.data)
        self.neg_sample_ratio = neg_sample_ratio

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    def collate_fn(self, datas):
        input_users = []
        #input_items = []
        #input_labels = []
        input_pos_items = []
        input_neg_items = []
        
        for user, pos_item, neg_item in datas:
            input_users.append(user)
            input_pos_items.append(pos_item)
            '''
            input_pos_items.append(random.sample(self.pos_item_dic[user], 1)[0])
            random_neg_item = random.sample(self.neg_item_dic[user], 1)[0]
            '''
            random_neg_item = random.sample(range(self.num_items), 1)[0]
            while random_neg_item in self.pos_item_dic[user]:
                random_neg_item = random.sample(range(self.num_items), 1)[0]
            
            input_neg_items.append(random_neg_item)

            #input_neg_items.append(neg_item)
        
        return torch.LongTensor(input_users), torch.LongTensor(input_pos_items), torch.LongTensor(input_neg_items)


def train(epochs, train_dim, loss_func, loss_type, trainData, maxID, maxDocID, test_user_lst, pos_item_dic, decay=0.01):
    model = MF(maxID, maxDocID, vec_dim=train_dim)

    device = torch.device('cpu')

    model.to(device)
    #opt = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=decay) # learning rate
    #opt = torch.optim.SGD(model.parameters(), lr=1e-3) # learning rate
    opt = torch.optim.Adam(model.parameters(), lr=1e-3) # learning rate

    #scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[150,200,250], gamma=0.5)

    history = {'train':[],'valid':[]}

    train_batch_size = 4096
    valid_batch_size = 1024

    for epoch in range(0, epochs):
        print("Epoch: ", epoch)
        # For training
        model.train(True)
        #dataset = MFDataset(train_user_lst, maxDocID, pos_item_dic, neg_sample_ratio)
        dataset = trainData
        dataloader = DataLoader(dataset,
                              batch_size=train_batch_size,
                              shuffle=True, 
                              collate_fn=dataset.collate_fn,
                              num_workers=4)
        trange = tqdm(enumerate(dataloader), total=len(dataloader), desc='Training', file=sys.stdout, position=0, leave=True)
        loss = 0
        vec_total_loss = 0

        for i, data in trange:
            '''
            input_users, input_items, input_labels = [d.to(device) for d in data]
            predict_labels = model(input_users, input_items)
            '''
            user_tensors, pos_item_tensors, neg_item_tensors = [d.to(device) for d in data]
            pos_scores, neg_scores = model(user_tensors, pos_item_tensors, neg_item_tensors)

            #print(predict_labels.shape, input_labels.shape)
            # BCELoss with L2 regularization
            if loss_type == 'BCELoss':
                #pos_scores = torch.sigmoid(pos_scores)
                #neg_scores = torch.sigmoid(neg_scores)
                one = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
                ground_truth = torch.FloatTensor(one).to(device)
                #predict = torch.cat([F.sigmoid(pos_scores), F.sigmoid(neg_scores)])
                #print(ground_truth.shape, predict.shape)
                #batch_loss = loss_func(pos_scores, neg_scores)
                #batch_loss = loss_func(predict, ground_truth)
                #batch_loss = loss_func(predict_labels, input_labels)

                reg_loss = decay * (torch.sum(model.userVec(user_tensors)**2) + torch.sum(model.docVec(pos_item_tensors)**2) + torch.sum(model.docVec(neg_item_tensors)**2))
                #reg_loss = decay * (torch.sum(model.userVec(user_tensors)**2) + torch.sum(model.docVec(pos_item_tensors)**2) + torch.sum(model.docVec(neg_item_tensors)**2) + torch.sum(model.l1(user_tensors)**2) + torch.sum(model.l2(pos_item_tensors)**2) + torch.sum(model.l2(neg_item_tensors)**2))
                #weights = torch.FloatTensor([0.025]).to(device)
                loss_function = torch.nn.BCEWithLogitsLoss(reduction='sum')
                #loss_function = torch.nn.BCELoss()
                batch_loss = loss_function(torch.cat((pos_scores, neg_scores)), ground_truth) + reg_loss

            # BPRLoss with L2 regularization
            else:
                #pos_vec = predict_labels[input_labels == 1]
                #neg_vec = predict_labels[input_labels == 0]
                #pos_scores = torch.sum(pos_scores, 1)
                #neg_scores = torch.sum(neg_scores, 1)
                #vec_loss = F.logsigmoid((pos_scores - neg_scores)).sum()
                vec_loss = (1 - torch.sigmoid(pos_scores - neg_scores)).sum()
                #vec_loss = (1 - torch.sigmoid(pos_scores - neg_scores)).mean()
                #vec_loss = loss_func(pos_scores, pos_scores)
                reg_loss = decay * (torch.sum(model.userVec(user_tensors)**2) + torch.sum(model.docVec(pos_item_tensors)**2) + torch.sum(model.docVec(neg_item_tensors)**2))
                #reg_loss = decay * (torch.sum(model.userVec(user_tensors)**2) + torch.sum(model.docVec(pos_item_tensors)**2) + torch.sum(model.docVec(neg_item_tensors)**2) + torch.sum(model.l1(user_tensors)**2) + torch.sum(model.l2(pos_item_tensors)**2) + torch.sum(model.l2(neg_item_tensors)**2))

                #batch_loss = - vec_loss + reg_loss
                batch_loss = vec_loss + reg_loss
                #batch_loss = -vec_loss
                #batch_loss = - torch.sum(loss_func(pos_scores, neg_scores))

            opt.zero_grad()
            batch_loss.backward()
            opt.step()
        
            loss += batch_loss.item()
            if loss_type == 'BCELoss':
                trange.set_postfix(loss=loss / (i+1))
            else:
                vec_total_loss = vec_loss.item()
                trange.set_postfix(loss= loss / (i+1))
        
        #scheduler.step()
        
        history['train'].append({'epoch': epoch, 'loss': loss / len(trange)})
       
        # For validation
        model.train(False)
        model.eval()
        '''
        dataset = validData
        dataloader = DataLoader(dataset, 
                              batch_size=valid_batch_size,
                              shuffle=False, 
                              collate_fn=dataset.collate_fn,
                              num_workers=4)
        '''
        #trange = tqdm(enumerate(dataloader), total=len(dataloader), desc='Validation', file=sys.stdout)
        loss = 0
        vec_total_loss = 0
        with torch.no_grad():
            #for i, data in trange:
            #    input_users, _, _ = data
            #    input_users = input_users.to(device)
            #input_users = torch.LongTensor(random.sample(range(maxID), 500)).to(device)
            #input_users = torch.LongTensor(range(maxID)).to(device)
            input_users = torch.LongTensor(test_user_lst).to(device)
            all_item_lst = torch.LongTensor(range(maxDocID)).to(device)
            predict_matrix = model.predict(input_users, all_item_lst)
            predict_matrix = predict_matrix.detach().cpu().numpy()

            input_users = input_users.detach().cpu().tolist()

            #ranking = np.argsort(predict_matrix, axis=1)[::-1] # reverse ranking - decreasing
            #print(ranking.shape)
            total_AP = []
            for i in range(len(input_users)):
                
                AP = 0
                user = input_users[i]
                labels = pos_item_dic[user]
                num_labels = len(labels)
                correct = 0
                ranking = np.argsort(predict_matrix[i])[::-1]
                for iidx, rank in enumerate(ranking[:50]):
                    if rank in labels:
                        correct += 1
                        AP += correct / (iidx+1)
                AP /= num_labels
                total_AP.append(AP)
                
            print("Valid Accuracy: ", sum(total_AP) / len(total_AP))
           
        history['valid'].append({'epoch': epoch, 'MAP': sum(total_AP) / len(total_AP)})
        
        save(epoch, model, history, loss_type, train_dim)

def save(epoch, model, history, loss_type, train_dim, cut=False):
    if not os.path.exists('model'):
        os.makedirs('model')
    if not cut:
        model_path = './model/model_{}_dim_{}.pkl.{}'.format(loss_type, train_dim, epoch)
        his_path = './model/history_{}_dim_{}.json'.format(loss_type, train_dim)
    else:
        model_path = './model/cutmodel_{}_dim_{}.pkl.{}'.format(loss_type, train_dim, epoch)
        his_path = './model/cuthistory_{}_dim_{}.json'.format(loss_type, train_dim)

    torch.save(model.state_dict(), model_path)
    with open(his_path, 'w') as f:
        json.dump(history, f, indent=4)

# BPR loss function
def BPRLoss():
    return lambda pos, neg: F.logsigmoid((pos - neg)).sum()
    #return lambda pos, neg: (1 - F.sigmoid((pos - neg))).mean()

def BCELoss():
    #return lambda pos, neg: - ( torch.sum(F.logsigmoid(pos)) + torch.sum(torch.log(max(1 - torch.sigmoid(neg), 1e-10))) )
    return lambda pos, neg: torch.cat([1 - F.sigmoid(pos), F.sigmoid(neg)]).mean()

'''
    WARP Loss part
'''
def num_tries_gt_zero(scores, batch_size, max_trials, max_num, device):
    tmp = torch.nonzero(scores.gt(0)).t()
    # We offset these values by 1 to look for unset values (zeros) later
    values = tmp[1] + 1
    if device.type == "cuda":
        t = torch.cuda.sparse.LongTensor(
            tmp, values, torch.Size((batch_size, max_trials + 1))
        ).to_dense()
    else:
        t = torch.sparse.LongTensor(
            tmp, values, torch.Size((batch_size, max_trials + 1))
        ).to_dense()
    t[(t == 0)] += max_num
    # set all unused indices to be max possible number so its not picked by min() call
    tries = torch.min(t, dim=1)[0]
    return tries

def warp_loss(positive_predictions, negative_predictions, num_labels, device):
    batch_size, max_trials = negative_predictions.size(0), negative_predictions.size(1)
    offsets, ones, max_num = (
        torch.arange(0, batch_size, 1).long().to(device) * (max_trials + 1),
        torch.ones(batch_size, 1).float().to(device),
        batch_size * (max_trials + 1),
    )
    sample_scores = 1 + negative_predictions - positive_predictions
    # Add column of ones so we know when we used all our attempts,
    # This is used for indexing and computing should_count_loss if no real value is above 0
    sample_scores, negative_predictions = (
        torch.cat([sample_scores, ones], dim=1),
        torch.cat([negative_predictions, ones], dim=1),
    )
    tries = num_tries_gt_zero(sample_scores, batch_size, max_trials, max_num, device)
    attempts, trial_offset = tries.float(), (tries - 1) + offsets
    loss_weights, should_count_loss = (
        torch.log(torch.floor((num_labels - 1) / attempts)),
        (attempts <= max_trials).float(),
    )  # Don't count loss if we used max number of attempts
    losses = (
        loss_weights
        * ((1 - positive_predictions.view(-1)) + negative_predictions.view(-1)[trial_offset])
        * should_count_loss
    )
    return losses.unsqueeze(1)
