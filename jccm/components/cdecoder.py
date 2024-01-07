import torch
import torch.nn as nn

class cDecoder(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.decoder_dim = 32
        self.seq = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 3)),
            
            nn.Conv2d(4, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 3)),

            nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 3)),
        )
        
    def forward(self, x):
        x = x.view(-1, 1, 11, self.decoder_dim)
        x = self.seq(x)
        return x