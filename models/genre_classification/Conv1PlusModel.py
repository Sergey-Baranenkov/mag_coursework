from torch import nn


def ConvBlock(in_dim, out_dim, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size),
        nn.BatchNorm2d(out_dim),
        nn.ReLU()
    )


def LinearBlock(in_dim, out_dim, dropout=0.1):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ELU(),
        nn.Dropout(dropout)
    )


# Свертка по времени
class Conv1PlusModel(nn.Module):
    def __init__(self, num_class, time_size, feature_size=128, num_conv_layers=4, num_linear_layers=3):
        super(Conv1PlusModel, self).__init__()

        horizontal_size = 16

        conv_channels = [1] + [2 ** (i + 3) for i in range(1, num_conv_layers + 1)]
        self.conv_block = nn.Sequential(*[
            ConvBlock(in_dim, out_dim, (horizontal_size, 1))
            for in_dim, out_dim in zip(conv_channels, conv_channels[1:])
        ])

        linear_dims = [feature_size * conv_channels[-1] * (time_size - (horizontal_size - 1) * num_conv_layers)] + [
            2 ** (8 - i) for i in
            range(0, num_linear_layers)]
        self.linear_block = nn.Sequential(*[
            LinearBlock(in_dim, out_dim)
            for in_dim, out_dim in zip(linear_dims, linear_dims[1:])
        ])

        self.linear_out = nn.Linear(linear_dims[-1], num_class)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x[:, None, :, :]

        x = self.conv_block(x)

        x = self.global_pooling(x)

        x = x.flatten(start_dim=1)

        x = self.linear_block(x)

        return self.linear_out(x)
