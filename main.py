'''
Reference: 
https://blog.fastforwardlabs.com/2018/04/10/pytorch-for-recommenders-101.html
https://www.ethanrosenthal.com/2017/06/20/matrix-factorization-in-pytorch/

'''
import logging
import argparse
import torch
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn

import random
from tqdm import tqdm
#from tqdm import tqdm_notebook as tqdm
import os
import sys
import json

# utils
from utils import preprocessing, MFDataset, train, BPRLoss, BCELoss
from model import MF

torch.manual_seed(666) # cpu
np.random.seed(666) #numpy
random.seed(666)

# global variable
device = torch.device('cpu') # use cpu
maxID = 4454
maxDocID = 3260

#import matplotlib.pyplot as plt

def predict(dim, pos_item_dic, show_MAP=False):
    '''
    Predict the ranking matrix
    '''
    model = MF(maxID, maxDocID, vec_dim=dim)
    model_path = './model.pkl'
    #model_path = './cutmodel_BPRLoss_dim_256.pkl.120'
    model.load_state_dict(torch.load(model_path, map_location='cpu')) # use cpu
    model.to(device)
    model.eval()

    # predict the matrix
    all_user_lst = torch.LongTensor(range(maxID)).to(device)
    all_item_lst = torch.LongTensor(range(maxDocID)).to(device)

    predict_matrix = model.predict(all_user_lst, all_item_lst)
    predict_matrix = predict_matrix.detach().cpu().numpy()

    #print(predict_matrix.shape)

    prediction = []
    origin_prediction = []

    for i in range(predict_matrix.shape[0]):
        labels = pos_item_dic[i]
        this_user = []
        num = 0
        cur_idx = 0
        this_ranking = np.argsort(predict_matrix[i])[::-1]
        origin_prediction.append(this_ranking[:50])
        while num < 50 and cur_idx < maxDocID:
            if this_ranking[cur_idx] not in labels:
                this_user.append(this_ranking[cur_idx])
                num += 1
            cur_idx += 1
        prediction.append(this_user)

    if show_MAP:
        total_AP = []
        for i in range(predict_matrix.shape[0]):
            AP = 0
            labels = pos_item_dic[i]
            correct = 0
            for iidx, rank in enumerate(origin_prediction[i][:50]):
                if rank in labels:
                    correct += 1
                    AP += correct / (iidx+1)
            AP /= 50
            total_AP.append(AP)

        print(sum(total_AP) / len(total_AP))
    
    return prediction

def SubmitGenerator(prediction, filename='./prediction.csv'):
    """
    Output the prediction to csv

    Args:
        prediction (List)
        filename (str)
    """
    submit = {}
    submit['UserId'] = range(maxID)
    
    docs = []
    for p in prediction:
        docs.append(' '.join(map(str,p)))

    submit['ItemId'] = list(docs)

    df = pd.DataFrame.from_dict(submit)
    df.to_csv(filename,index=False)

def _parse_args():
    parser = argparse.ArgumentParser(
        description="IR Coding-HW2."
    )
    parser.add_argument('--output', type=Path, required=True, help="Output file path")
    parser.add_argument('--train', action="store_true", help="Do training") # if -r specified, then turn on the feedback
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _parse_args()
    loglevel = os.environ.get('LOGLEVEL', 'DEBUG').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    
    logging.debug("START")
    
    # Load and pre-processing
    logging.debug("Loading input and do preprocessing")
    train_path = './train.csv'
    maxID, maxDocID, matrix, pos_item_dic = preprocessing(train_path)
    '''
    Model setting
    '''
    model_dim = 256
    loss_type = 'BPRLoss'

    if args.train:
        logging.debug("Loading dataset")
        total_user_lst = range(maxID)
        #train_user_lst, test_user_lst = train_test_split(total_user_lst, test_size=0.1, random_state=666)
        train_user_lst = total_user_lst
        test_user_lst = random.sample(range(maxID), 500)
        
        neg_sample_ratio = 1.5
        trainData = MFDataset(train_user_lst, maxDocID, pos_item_dic, neg_sample_ratio)

        logging.debug("START training")
        max_epoch = 400
        loss_func = BPRLoss()
        train(max_epoch, model_dim, loss_func, "BPRLoss", trainData, maxID, maxDocID, test_user_lst, pos_item_dic, decay=0.005)

    else:
        logging.debug("Start predicting")
        prediction = predict(model_dim, pos_item_dic)

        # Output
        logging.debug("Output csv")
        SubmitGenerator(prediction, args.output)

    logging.debug("END")
