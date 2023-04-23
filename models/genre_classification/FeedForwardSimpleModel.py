from torch import nn


class FeedForwardSimpleModel(nn.Module):
    def __init__(self, input_dim, num_class):
        super(FeedForwardSimpleModel, self).__init__()

        self.layer_1 = nn.Linear(input_dim, 1024)
        self.layer_2 = nn.Linear(1024, 512)
        self.layer_3 = nn.Linear(512, 128)
        self.layer_4 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.batchnorm4 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        #
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)

        x = self.layer_4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)

        x = self.layer_out(x)

        return x