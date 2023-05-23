from catboost import CatBoostClassifier

clf = CatBoostClassifier()
clf.load_model('models/genre_classification/genre_classification_model')

idx_to_label = {
    0: 'Blues',
    1: 'Classical',
    2: 'Electronic',
    3: 'Folk',
    4: 'Hip-Hop',
    5: 'Jazz',
    6: 'Pop',
    7: 'Rock'
}


def predict_genre(features):
    predicted_idx = clf.predict(features)[0]

    return idx_to_label[predicted_idx]
