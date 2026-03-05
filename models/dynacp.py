import torch
import torch.nn as nn

class SelectiveConv(nn.Module):
    def __init__(self, ch, stride=(1,1), dilation=1):
        super().__init__()
        self.a = nn.Sequential(
            nn.Conv2d(ch, ch, 3, stride=stride, padding=dilation, dilation=dilation, groups=ch, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        self.b = nn.Sequential(
            nn.Conv2d(ch, ch, 3, stride=stride, padding=1, groups=ch, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(ch, 2))

    def forward(self, x):
        w = torch.softmax(self.gate(x), dim=1)
        ya = self.a(x); yb = self.b(x)
        wa = w[:,0].view(-1,1,1,1); wb = w[:,1].view(-1,1,1,1)
        return wa*ya + wb*yb

class DynaCPBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=(1,1), dilation=1, expand=2):
        super().__init__()
        mid = in_ch*expand
        self.pw1 = nn.Sequential(nn.Conv2d(in_ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.sel = SelectiveConv(mid, stride=stride, dilation=dilation)
        self.pw2 = nn.Sequential(nn.Conv2d(mid, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.act = nn.ReLU(inplace=True)
        self.use_res = (stride==(1,1) and in_ch==out_ch)

    def forward(self, x):
        y = self.pw1(x)
        y = self.sel(y)
        y = self.pw2(y)
        if self.use_res:
            y = y + x
        return self.act(y)

class DynaCP(nn.Module):
    def __init__(self, num_outputs: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=(2,2), padding=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, 32, 3, stride=(2,2), padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            DynaCPBlock(32, 32, (1,1), dilation=2),
            DynaCPBlock(32, 32, (2,1), dilation=1),
            DynaCPBlock(32, 48, (1,2), dilation=1),
            DynaCPBlock(48, 72, (2,1), dilation=1),
            DynaCPBlock(72, 72, (1,1), dilation=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Linear(72, num_outputs)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.head(x)
