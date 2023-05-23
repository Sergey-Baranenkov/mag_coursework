from catboost import CatBoostClassifier

clf = CatBoostClassifier()
clf.load_model('models/decade_classification/decade_classification_model')

idx_to_label = {0: '1990', 1: '2000', 2: '2010', 3: 'old'}


def predict_decade(features):
    predicted_idx = clf.predict(features)[0]

    return idx_to_label[predicted_idx]
