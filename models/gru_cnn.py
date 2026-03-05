import torch
import torch.nn as nn

class GRUCNN(nn.Module):
    """A compact CNN -> GRU -> classifier model (often described as CNN-GRU / GRU-CNN).

    Input: (B, 1, F, T)
    We do small 2D convs to reduce frequency axis, then treat time as sequence for a GRU.
    """
    def __init__(self, num_outputs: int, feat_dim: int = 64, gru_hidden: int = 96, gru_layers: int = 1, bidir: bool = True):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=(2,2), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=(2,1), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, feat_dim, kernel_size=3, stride=(2,1), padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

        self.pool_f = nn.AdaptiveAvgPool2d((1, None))  # pool over frequency only -> (B,C,1,T')
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=bidir,
        )
        out_gru = gru_hidden * (2 if bidir else 1)

        self.head = nn.Sequential(
            nn.LayerNorm(out_gru),
            nn.Linear(out_gru, num_outputs),
        )

    def forward(self, x):
        # x: (B,1,F,T)
        x = self.cnn(x)               # (B,C,F',T')
        x = self.pool_f(x).squeeze(2) # (B,C,T')
        x = x.transpose(1,2)          # (B,T',C)
        y, _ = self.gru(x)            # (B,T',H)
        y = y.mean(dim=1)             # time-avg pooling
        return self.head(y)
