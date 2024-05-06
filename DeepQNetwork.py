import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims

        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3= nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()



        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state.to(self.device)

        # Input Layer
        x = state.to(T.float32)

        # Layer 1
        x = self.fc1(x)
        x = F.relu6(x)
        if len(x.size()) > 1:
            x = nn.BatchNorm1d(x.size()[1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)(x)


        # Layer 2
        x = self.fc2(x)
        x = F.relu6(x)

        # Layer 3
  
        x = self.fc3(x)
        x = F.relu6(x)
        x = nn.Dropout(0.2)(x)



        # Ouput layer
        actions = self.fc4(x)
        return actions