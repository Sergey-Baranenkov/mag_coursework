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

        self.layer_1 = linear_block(input_dim, 1024)
        self.layer_2 = linear_block(1024, 512)
        self.layer_3 = linear_block(512, 512)
        self.layer_4 = linear_block(512, 128)
        self.layer_out = nn.Linear(128, num_class)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x1 = self.layer_1(x)

        x2 = self.layer_2(x1)

        x3 = self.layer_3(x2)

        x4 = self.layer_4(x3 + x2)

        x5 = self.layer_out(x4)

        return x5
