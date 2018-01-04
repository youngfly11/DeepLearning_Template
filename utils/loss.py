import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class RewardLoss(nn.Module):

    def __init__(self, num_glimpse):
        super(RewardLoss, self).__init__()
        self.num_glimpse = num_glimpse

    def forward(self, value, reward, log_prob_all):

        rewards = [reward.unsqueeze(1) for i in range(self.num_glimpse)]
        rewards = torch.cat(tuple(rewards), 1) # batch*time_step
        values = torch.cat(tuple(value), 1)  # batch*time_step
        log_probs = torch.cat(tuple(log_prob_all), 1)
        advs = rewards-values.detach()

        # loss for action networks
        loss1 = - torch.mean(advs*log_probs)

        # loss for target network, the parameter update action network and target networks
        loss2 = torch.mean((rewards-values)**2)

        return loss1, loss2


class ClassifierLoss(nn.Module):

    def __init__(self, num_glimpse):
        super(ClassifierLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, score, prob, label):

        label = label.long()
        _, pred = torch.max(prob, 1)
        pred_np = pred.data.cpu().numpy()
        label_np = label.data.cpu().numpy()
        acc = np.mean(pred_np == label_np)
        reward = (pred==label).float()
        loss = self.loss(score, label)

        return loss, acc, reward


if __name__ == '__main__':

    # TODO
    # test different modules
    score = torch.FloatTensor(4, 10).uniform_(-5, 5).cuda()
    score = Variable(score)
    label = torch.FloatTensor([2,3,5,7]).cuda()
    label = Variable(label)
    prob = F.softmax(score, dim=1)
    criterion2 = RewardLoss(num_glimpse=10)
    log_prob_all = [Variable(torch.FloatTensor(4, 1).uniform_(-1, 1).cuda()) for i in range(10)]
    value = [Variable(torch.FloatTensor(4, 1).uniform_(-1, 1).cuda()) for i in range(10)]
    criterion = ClassifierLoss(num_glimpse=10)
    loss, acc, reward = criterion(score=score, prob=prob,label=label)
    loss = criterion2(value=value, reward=reward, log_prob_all=log_prob_all)
    print loss, acc, reward,

