import torch.nn as nn

class DWConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=(1,1), expand=2):
        super().__init__()
        mid = in_ch*expand
        self.pw1 = nn.Sequential(nn.Conv2d(in_ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.dw  = nn.Sequential(nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.pw2 = nn.Sequential(nn.Conv2d(mid, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.act = nn.ReLU(inplace=True)
        self.use_res = (stride==(1,1) and in_ch==out_ch)

    def forward(self, x):
        y = self.pw1(x)
        y = self.dw(y)
        y = self.pw2(y)
        if self.use_res:
            y = y + x
        return self.act(y)

class CPMobile(nn.Module):
    def __init__(self, num_outputs: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=(2,2), padding=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, 32, 3, stride=(2,2), padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            DWConvBlock(32, 32, (1,1)),
            DWConvBlock(32, 32, (2,1)),
            DWConvBlock(32, 32, (1,1)),
            DWConvBlock(32, 48, (1,2)),
            DWConvBlock(48, 48, (1,1)),
            DWConvBlock(48, 72, (2,1)),
            DWConvBlock(72, 72, (1,1)),
        )
        self.tail = nn.Sequential(nn.Conv2d(72,72,1,bias=False), nn.BatchNorm2d(72), nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Linear(72, num_outputs)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.tail(x)
        x = self.pool(x).flatten(1)
        return self.head(x)
