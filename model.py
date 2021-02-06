import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # first shared mlp
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)

        # second shared mlp
        self.conv3 = nn.Conv1d(512, 512, 1)
        self.conv4 = nn.Conv1d(512, 1024, 1)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(1024)
    
    def forward(self, x):
        n = x.size()[2]

        # first shared mlp
        x = F.relu(self.bn1(self.conv1(x)))           # (B, 128, N)
        f = self.bn2(self.conv2(x))                   # (B, 256, N)
        
        # point-wise maxpool
        g = torch.max(f, dim=2, keepdim=True)[0]      # (B, 256, 1)
        
        # expand and concat
        x = torch.cat([g.repeat(1, 1, n), f], dim=1)  # (B, 512, N)

        # second shared mlp
        x = F.relu(self.bn3(self.conv3(x)))           # (B, 512, N)
        x = self.bn4(self.conv4(x))                   # (B, 1024, N)
        
        # point-wise maxpool
        v = torch.max(x, dim=-1)[0]                   # (B, 1024)
        
        return v


class Decoder(nn.Module):
    def __init__(self, num_coarse=1024, num_dense=16384):
        super(Decoder, self).__init__()

        self.num_coarse = num_coarse
        
        # fully connected layers
        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 3 * num_coarse)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)

        # shared mlp
        self.conv1 = nn.Conv1d(3+2+1024, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, 3, 1)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(512)

        # 2D grid
        grids = np.meshgrid(np.linspace(-0.05, 0.05, 4, dtype=np.float32),
                            np.linspace(-0.05, 0.05, 4, dtype=np.float32))                               # (2, 4, 44)
        self.grids = torch.Tensor(grids).view(2, -1)  # (2, 4, 4) -> (2, 16)
    
    def forward(self, x):
        b = x.size()[0]
        # global features
        v = x  # (B, 1024)

        # fully connected layers to generate the coarse output
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        y_coarse = x.view(-1, 3, self.num_coarse)  # (B, 3, 1024)

        repeated_centers = y_coarse.unsqueeze(3).repeat(1, 1, 1, 16).view(b, 3, -1)  # (B, 3, 16x1024)
        repeated_v = v.unsqueeze(2).repeat(1, 1, 16 * self.num_coarse)               # (B, 1024, 16x1024)
        grids = self.grids.to(x.device)  # (2, 16)
        grids = grids.unsqueeze(0).repeat(b, 1, self.num_coarse)                     # (B, 2, 16x1024)

        x = torch.cat([repeated_v, grids, repeated_centers], dim=1)                  # (B, 2+3+1024, 16x1024)
        x = F.relu(self.bn3(self.conv1(x)))
        x = F.relu(self.bn4(self.conv2(x)))
        x = self.conv3(x)                # (B, 3, 16x1024)
        y_detail = x + repeated_centers  # (B, 3, 16x1024)

        return y_coarse, y_detail


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        v = self.encoder(x)
        y_coarse, y_detail = self.decoder(v)
        return v, y_coarse, y_detail


if __name__ == "__main__":
    pcs = torch.rand(16, 3, 2048)
    encoder = Encoder()
    v = encoder(pcs)
    print(v.size())

    decoder = Decoder()
    decoder(v)
    y_c, y_d = decoder(v)
    print(y_c.size(), y_d.size())

    ae = AutoEncoder()
    v, y_coarse, y_detail = ae(pcs)
    print(v.size(), y_coarse.size(), y_detail.size())
