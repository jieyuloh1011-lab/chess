import torch, torch.nn as nn, torch.nn.functional as F
from .encoding import N_PLANES, N_MOVES


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)
        self.c2, self.b2 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        return F.relu(x + self.b2(self.c2(y)))


class ChessNet(nn.Module):
    """ResNet backbone -> policy head (4096 logits) + value head (tanh in [-1,1])."""
    def __init__(self, ch=64, blocks=6):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(N_PLANES, ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(ch), nn.ReLU())
        self.body = nn.Sequential(*[Block(ch) for _ in range(blocks)])
        self.p_conv = nn.Sequential(nn.Conv2d(ch, 32, 1), nn.BatchNorm2d(32), nn.ReLU())
        self.p_fc = nn.Linear(32 * 64, N_MOVES)
        self.v_conv = nn.Sequential(nn.Conv2d(ch, 8, 1), nn.BatchNorm2d(8), nn.ReLU())
        self.v_fc = nn.Sequential(nn.Linear(8 * 64, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh())

    def forward(self, x):
        h = self.body(self.stem(x))
        return self.p_fc(self.p_conv(h).flatten(1)), self.v_fc(self.v_conv(h).flatten(1)).squeeze(1)
