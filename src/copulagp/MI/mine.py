import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

def train_MINE(y, x=None, H=20, lr=0.01, batches=10, n_epoch=2000, device = torch.device("cuda:1")):

    data_size = y.shape[0]
    if x is None:
        x = torch.linspace(0.,1.,data_size)
    else:
        assert x.shape[0]==data_size
    
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(1, H)
            self.fc2 = nn.Linear(y.shape[-1], H)
            self.fc3 = nn.Linear(H, 1)

        def forward(self, x, y):
            h1 = F.relu(self.fc1(x)+self.fc2(y))
            h2 = self.fc3(h1)
            return h2  
    
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    plot_loss = []
    x_sample = Variable(torch.cat(batches*[x.unsqueeze(-1).to(device)]), 
                        requires_grad = True)   
    y_sample = Variable(torch.cat(batches*[torch.from_numpy(y).type(torch.FloatTensor).to(device)]), 
                       requires_grad = True)
    for epoch in range(n_epoch):
        y_shuffle = y_sample[torch.cat([torch.randperm(data_size) for _ in range(batches)])]

        pred_xy = model(x_sample, y_sample)
        pred_x_y = model(x_sample, y_shuffle)

        ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
        loss = - ret  # maximize
        plot_loss.append(-loss.data.cpu().numpy())
        model.zero_grad()
        loss.backward()
        optimizer.step()
    return torch.tensor(plot_loss[-50:]).mean().numpy()