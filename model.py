import torch
import torch.nn.functional as F
import torch.nn as nn

class MF(nn.Module):
    def __init__(self, num_user, num_doc, vec_dim):
        '''
        Two embedding: one for user vector, one for doc vec
        '''
        super(MF, self).__init__()
        self.userVec = nn.Embedding(num_user, vec_dim) # sparse = True
        self.docVec = nn.Embedding(num_doc, vec_dim)

        #self.userVec.weight.data.normal_(0, 1.0 / vec_dim)
        #self.docVec.weight.data.normal_(0, 1.0 / vec_dim)
        
        #self.userBias = torch.nn.Embedding(num_user, 1)
        #self.itemBias = torch.nn.Embedding(num_doc, 1)

        #self.userBias.weight.data.zero_()
        #self.itemBias.weight.data.zero_()
        
        #self.l1 = nn.Linear(vec_dim, vec_dim)
        #self.l2 = nn.Linear(vec_dim, vec_dim)
    
    def forward(self, input_users, input_pos_items, input_neg_items):
        # elementwise multiplication
        #pos_scores = self.userVec(input_users) * self.docVec(input_pos_items)
        #neg_scores = self.userVec(input_users) * self.docVec(input_neg_items)
        #x1 = F.relu(self.l1(self.userVec(input_users)))
        #x2 = F.relu(self.l2(self.docVec(input_pos_items)))
        #x3 = F.relu(self.l2(self.docVec(input_neg_items)))
        #pos_scores = torch.mul(x1, x2)
        #neg_scores = torch.mul(x1, x3)

        pos_scores = torch.mul(self.userVec(input_users), self.docVec(input_pos_items))
        neg_scores = torch.mul(self.userVec(input_users), self.docVec(input_neg_items))

        #pos_bias = self.userBias(input_users) + self.itemBias(input_pos_items)
        #neg_bias = self.userBias(input_users) + self.itemBias(input_neg_items)
        
        #print(pos_scores.sum(1).shape, pos_bias.shape)

        #return pos_scores.sum(1) + pos_bias.squeeze(), neg_scores.sum(1) + neg_bias.squeeze()
        return pos_scores.sum(1), neg_scores.sum(1)
        

        #pos_scores += (self.userVec(input_users) * self.docVec(input_pos_items)).sum(dim=1, keepdim=True)
        #neg_scores += (self.userVec(input_users) * self.docVec(input_neg_items)).sum(dim=1, keepdim=True)
        #print(scores.shape)
        #return torch.sum(scores, 1)
        #return pos_scores.squeeze(), neg_scores.squeeze()

    def predict(self, input_users, input_items):
        #x1 = F.relu(self.l1(self.userVec(input_users)))
        #x2 = F.relu(self.l2(self.docVec(input_items)))
        #return torch.mm(x1, x2.t())
        #user_bias = self.userBias(input_users).view(-1, 1).expand(input_users.shape[0], input_items.shape[0])
        #item_bias = self.itemBias(input_items).view(1, -1).expand(input_users.shape[0], input_items.shape[0])

        #return torch.mm(self.userVec(input_users), self.docVec(input_items).t()) + user_bias + item_bias
        return torch.mm(self.userVec(input_users), self.docVec(input_items).t())