import torch

from models.instrument_classification.model import Model
from constants.index import DEVICE

models = dict()

idx_to_label = {0: 'accordion',
                1: 'banjo',
                2: 'bass',
                3: 'cello',
                4: 'clarinet',
                5: 'cymbals',
                6: 'drums',
                7: 'flute',
                8: 'guitar',
                9: 'mallet_percussion',
                10: 'mandolin',
                11: 'organ',
                12: 'piano',
                13: 'saxophone',
                14: 'synthesizer',
                15: 'trombone',
                16: 'trumpet',
                17: 'ukulele',
                18: 'violin',
                19: 'voice'
                }

for name in idx_to_label.values():
    models[name] = Model().to(DEVICE)
    models[name].load_state_dict(torch.load(f'models/instrument_classification/weights/{name}'))
    models[name].eval()



def predict_instruments(features):
    features = torch.tensor([features]).to(DEVICE)
    instruments = []
    for key, model in models.items():
        logits = model(features)
        res = (torch.sigmoid(logits) > 0.5).flatten().item()
        if res:
            instruments.append(key)

    return instruments
