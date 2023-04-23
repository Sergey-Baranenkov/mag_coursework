from torch import nn


class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()

        # Слой эмбеддинга
        self.embedding = nn.Linear(input_dim, d_model)

        # Энкодер
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Классификатор
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Применяем слой эмбеддинга
        x = self.embedding(x)

        # Транспонируем входные данные для соответствия формату Transformer
        x = x.transpose(0, 1)

        # Применяем энкодер
        x = self.encoder(x)

        # Среднее значение по временной оси
        x = x.mean(dim=0)

        # Применяем классификатор
        x = self.classifier(x)

        return x