import torch

model = torch.hub.load('harritaylor/torchvggish', 'vggish')
model.eval()


def extract_vggish_features(filename):
    return model.forward(filename).cpu().detach().numpy()
