from torch import nn
import torch.nn.functional as F
class LeNet(nn.Module):
    def __init__(self, hidden_channel=16, hidden_linear=120):
        super(LeNet, self).__init__()
        self.hidden_channel = hidden_channel
        self.hidden_linear = hidden_linear
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=2) # [1,1,28,28] ->[1,6,32,32]-> [1,6,28,28]
        self.pool = nn.MaxPool2d(2, 2) # [1,6,28,28] -> [1,6,14,14]
        self.conv2 = nn.Conv2d(6, hidden_channel, 5) # [1,6,14,14] -> [1,hidden_channel,10,10]
        self.fc1 = nn.Linear(hidden_channel * 5 * 5, hidden_linear) # [1,hidden_channel,10,10] -> [1,hidden_channel * 5 * 5] -> [1,hidden_linear]
        self.fc2 = nn.Linear(hidden_linear, 84) # [1,hidden_linear] -> [1,84]
        self.fc3 = nn.Linear(84, 10) # [1,84] -> [1,10]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.hidden_channel* 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=1)
        return x
    
