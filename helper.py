import matplotlib.pyplot as plt
import numpy as np
from random import  random

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms

from utils.parse_config import *

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net_config = parse_model_config('cfg/yolov3.cfg')[0]

net = Net().cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

net_config['burn_in'] = 5
net_config['policy'] = '222'

# float get_current_rate(network *net)
# {
#     size_t batch_num = get_current_batch(net);
#     int i;
#     float rate;
#     if (batch_num < net->burn_in) return net->learning_rate * pow((float)batch_num / net->burn_in, net->power);
#     switch (net->policy) {
#         case CONSTANT:
#             return net->learning_rate;
#         case STEP:
#             return net->learning_rate * pow(net->scale, batch_num/net->step);
#         case STEPS:
#             rate = net->learning_rate;
#             for(i = 0; i < net->num_steps; ++i){
#                 if(net->steps[i] > batch_num) return rate;
#                 rate *= net->scales[i];
#             }
#             return rate;
#         case EXP:
#             return net->learning_rate * pow(net->gamma, batch_num);
#         case POLY:
#             return net->learning_rate * pow(1 - (float)batch_num / net->max_batches, net->power);
#         case RANDOM:
#             return net->learning_rate * pow(rand_uniform(0,1), net->power);
#         case SIG:
#             return net->learning_rate * (1./(1.+exp(net->gamma*(batch_num - net->step))));
#         default:
#             fprintf(stderr, "Policy is weird!\n");
#             return net->learning_rate;
#     }
# }


def get_current_rate(batch_num):
    global net_config

    if batch_num < net_config['burn_in']:
        return (batch_num / net_config['burn_in']) ** net_config['power']

    if net_config['policy'] == 'constant':
        return 1
    elif net_config['policy'] == 'step':
        return net_config['scale'] ** (batch_num / net_config['step'])
    elif net_config['policy'] == 'steps':
        rate = 1
        for step in net_config['steps']:
            if batch_num < step:
                return rate
            rate *= step
        return rate
    elif net_config['policy'] == 'exp':
        return net_config['gamma'] ** batch_num
    elif net_config['policy'] == 'poly':
        return (1 - batch_num/net_config['max_batches']) ** net_config['power']
    elif net_config['policy'] == 'random':
        return random() ** net_config['power']
    elif net_config['policy'] == 'sigmoid':
        return 1 / (1 + np.exp(net_config['gamma'] * (batch_num - net_config['step'])))
    else:
        print("{} policy is not implemented. Using constant policy".format(net_config['policy']))
        net_config['policy'] = 'constant'
        return 1


scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

scheduler_burn_in = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_current_rate)

for i in range(10):
    print(optimizer.param_groups[0]['lr'], i)

    scheduler_burn_in.step(i)

    # if i == 30 or i == 50:
    #     scheduler.step()


# for epoch in range(10):  # loop over the dataset multiple times
#     print(optimizer.param_groups[0]['lr'])
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#
#         # get the inputs
#         inputs, labels = data
#
#         # wrap them in Variable
#         inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#
#         # print statistics
#         running_loss += loss.data[0]
#         if i % 2000 == 1999:  # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0
#
# print('Finished Training')