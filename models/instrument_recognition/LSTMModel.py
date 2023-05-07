from torch import nn


def linear_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(out_dim),
        nn.Dropout(0.1),
    )


class LSTMModel(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=128):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2)

        self.layer_1 = linear_block(hidden_dim, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_3 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)

        self.tanh = nn.Tanh()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1]
        x = self.tanh(out)

        x1 = self.layer_1(x)

        x2 = self.layer_2(x1)

        x3 = self.layer_3(x2 + x1)

        x4 = self.layer_out(x3 + x2)

        return x4
