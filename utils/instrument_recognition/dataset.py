from torch.utils.data import Dataset


class InstrumentClassificationDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        x = x.astype('float32')

        if self.transform:
            x = self.transform(x)

        return x, y