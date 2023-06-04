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


class CRNNModel(nn.Module):
    def __init__(self, time_size, feature_size, num_class):
        super(CRNNModel, self).__init__()

        self.conv1 = conv_block(1, 16, (5, 1), (2, 1))
        self.conv2 = conv_block(16, 32, (5, 1), (2, 1))
        self.conv3 = conv_block(32, 32, (5, 1), (2, 1))
        self.conv4 = conv_block(32, 32, (5, 1), (2, 1))

        self.lstm = nn.LSTM(
            feature_size * 32,
            256,
            bidirectional=False,
            batch_first=True,
            num_layers=3
        )
        self.tanh = nn.Tanh()

        self.linear_1 = linear_block(256, 256)
        self.linear_2 = linear_block(256, 256)
        self.linear_3 = linear_block(256, 256)
        self.linear_4 = linear_block(256, 256)

        self.linear_out = linear_block(256, num_class)



    def forward(self, x):
        x = x[:, None, :, :]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.permute(0, 2, 1, 3)
        x = x.flatten(start_dim=2)

        out, _ = self.lstm(x)
        out = out[:, -1]
        x = self.tanh(out)

        x1 = self.linear_1(x)
        x2 = self.linear_2(x1) + x1
        x3 = self.linear_3(x2) + x2
        x4 = self.linear_4(x3) + x3

        return self.linear_out(x4)
