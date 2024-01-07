import torch
import torch.nn as nn

class SignalPreprocess(nn.Module):
    def __init__(self, blocks=4):
        nn.Module.__init__(self)
        self.poolinglayers = nn.ModuleList([SignalPooling() for _ in range(blocks)])
        
    def forward(self, x):
        x = x.view(-1, 5000).to(torch.float32)
        for poolinglayer in self.poolinglayers:
            x = poolinglayer(x)
        x = x.view(-1, 1, 50, 100).to(torch.float32)
        return x
    
class SignalPooling(nn.Module):
    def __init__(self, minpool_k=3, minpool_s=1, avgpool_k=3, avgpool_s=1):
        super(SignalPooling, self).__init__()
        self.minpool = MinPool1d(kernel_size=minpool_k, stride=minpool_s)
        self.avgpool = nn.AvgPool1d(kernel_size=avgpool_k, stride=avgpool_s, padding=1)
        
    def forward(self, x):
        x = self.minpool(x)
        x = self.avgpool(x)
        return x
        

class MinPool1d(nn.Module):
    def __init__(self, kernel_size=3, stride=1):
        super(MinPool1d, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.w_width = kernel_size
        
    def forward(self, x):
        bs = x.size(0)
        in_width = x.size(1)
        out_width = int((in_width - self.w_width) / self.stride) + 1
        out = torch.Tensor([])
        for j in range(out_width):
            start_j = j * self.stride
            end_j = start_j + self.w_width
            if out.numel() > 0:
                out = torch.vstack((out, torch.min(x[:, start_j: end_j], dim=1)[0]))  
            else:
                out = torch.min(x[:, start_j: end_j], dim=1)[0] 
        out = out.T        
        padding = torch.zeros(bs * (in_width - out_width)).view(bs, -1)
        for i in range(bs):
            p = out[i]
            pmin, pmax = min(p), max(p)
            out[i] = (p - pmin) / (pmax - pmin) + 1e-9
        res = torch.concat((out, padding), 1)
        return res