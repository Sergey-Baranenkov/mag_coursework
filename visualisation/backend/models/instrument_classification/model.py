from torch import nn

def linear_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(out_dim),
        nn.Dropout(0.1),
    )


class Model(nn.Module):
    def __init__(self, input_dim=128):
        super(Model, self).__init__()

        self.layer_1 = linear_block(input_dim, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_3 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)

    def forward(self, x):
        x1 = self.layer_1(x)

        x2 = self.layer_2(x1)

        x3 = self.layer_3(x2 + x1)

        x4 = self.layer_out(x3 + x2)

        return x4
