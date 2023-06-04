import math

from torch import nn


def calculate_shape(x, n_layers=4):
    for i in range(n_layers):
        x = (x - 4) / 2

    return int(x)


def linear_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(out_dim),
        nn.Dropout(0.1),
    )


def conv_block(in_ch, out_ch, kernel_size, pooling_kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size),
        nn.MaxPool2d(pooling_kernel_size),
        nn.BatchNorm2d(out_ch),
        nn.ReLU()
    )


# Свертка по времени
class Conv1Model(nn.Module):
    def __init__(self, num_class, time_size, feature_size=128, n_conv_layers=4):
        super(Conv1Model, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(conv_block(1, 16, (5, 1), (2, 1)))
        self.conv_layers.append(conv_block(16, 32, (5, 1), (2, 1)))

        for i in range(n_conv_layers - 2):
            self.conv_layers.append(conv_block(32, 32, (5, 1), (2, 1)))

        self.linear_1 = linear_block(32 * feature_size * calculate_shape(time_size, n_conv_layers), 256)
        self.linear_2 = linear_block(256, 256)
        self.linear_3 = linear_block(256, 256)
        self.linear_4 = linear_block(256, 256)

        self.linear_out = linear_block(256, num_class)

    def forward(self, x):
        x = x[:, None, :, :]

        for layer in self.conv_layers:
            x = layer(x)

        x = x.flatten(start_dim=1)

        x1 = self.linear_1(x)
        x2 = self.linear_2(x1) + x1
        x3 = self.linear_3(x2) + x2
        x4 = self.linear_4(x3) + x3

        return self.linear_out(x4)
