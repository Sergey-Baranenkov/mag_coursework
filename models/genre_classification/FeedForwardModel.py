from torch import nn


def linear_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(out_dim),
        nn.Dropout(0.1),
    )


class FeedForwardModel(nn.Module):
    def __init__(self, input_dim, num_class):
        super(FeedForwardModel, self).__init__()

        self.linear_1 = linear_block(input_dim, 256)
        self.linear_2 = linear_block(256, 256)
        self.linear_3 = linear_block(256, 256)
        self.linear_4 = linear_block(256, 256)
        self.linear_out = nn.Linear(256, num_class)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x1 = self.linear_1(x)
        x2 = self.linear_2(x1) + x1
        x3 = self.linear_3(x2) + x2
        x4 = self.linear_4(x3) + x3

        return self.linear_out(x4)
