# Свертка с пулингом
from torch import nn


def calculate_shape(x, n_layers=4):
    for i in range(n_layers):
        x = (x - 2) / 2

    return int(x)

class Conv3Model(nn.Module):
    def __init__(self, num_class, time_size, feature_size, ):
        super(Conv3Model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))

        time_shape = calculate_shape(time_size)
        feature_shape = calculate_shape(feature_size)

        self.pooling1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pooling2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pooling3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pooling4 = nn.MaxPool2d(kernel_size=(2, 2 if feature_shape > 0 else 1))

        feature_shape = max(feature_shape, 1)
        fc_input_size = time_shape * feature_shape * 128

        self.linear_1 = nn.Linear(fc_input_size, 512)
        self.linear_2 = nn.Linear(512, 256)
        self.linear_3 = nn.Linear(256, 128)
        self.linear_4 = nn.Linear(128, 64)

        self.linear_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()

        self.conv_batchnorm1 = nn.BatchNorm2d(16, affine=True)
        self.conv_batchnorm2 = nn.BatchNorm2d(32, affine=True)
        self.conv_batchnorm3 = nn.BatchNorm2d(64, affine=True)
        self.conv_batchnorm4 = nn.BatchNorm2d(128, affine=True)

        self.linear_batchnorm1 = nn.BatchNorm1d(512, affine=True)
        self.linear_batchnorm2 = nn.BatchNorm1d(256, affine=True)
        self.linear_batchnorm3 = nn.BatchNorm1d(128, affine=True)
        self.linear_batchnorm4 = nn.BatchNorm1d(64, affine=True)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout4 = nn.Dropout(0.1)

    def forward(self, x):
        x = x[:, None,:,:]

        x = self.conv1(x)
        x = self.conv_batchnorm1(x)
        x = self.relu(x)
        x = self.pooling1(x)

        x = self.conv2(x)
        x = self.conv_batchnorm2(x)
        x = self.relu(x)
        x = self.pooling2(x)

        x = self.conv3(x)
        x = self.conv_batchnorm3(x)
        x = self.relu(x)
        x = self.pooling3(x)

        x = self.conv4(x)
        x = self.conv_batchnorm4(x)
        x = self.relu(x)
        x = self.pooling4(x)

        x = x.flatten(start_dim = 1)

        x = self.linear_1(x)
        x = self.linear_batchnorm1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.linear_2(x)
        x = self.linear_batchnorm2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.linear_3(x)
        x = self.linear_batchnorm3(x)
        x = self.relu(x)
        x = self.dropout3(x)

        x = self.linear_4(x)
        x = self.linear_batchnorm4(x)
        x = self.relu(x)
        x = self.dropout4(x)

        return self.linear_out(x)
#%%
