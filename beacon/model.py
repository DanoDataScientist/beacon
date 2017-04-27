import os, pickle
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer

def get_model(path='models/model.pkl'):
    if not os.path.exists(path):
        classifier = RandomForestClassifier(n_estimators=500, class_weight='balanced')
        model = make_pipeline(DictVectorizer(sparse=False), classifier)
        save_model(model, path=path)
    else:
        model = load_model(path=path)
    return model

def save_model(model, path='models/model.pkl'):
    dir_name = os.path.dirname(os.path.abspath(path))
    if not os.path.exists(dir_name): os.mkdir(dir_name)
    with open(path, 'wb') as model_file:
        pickle.dump(model, model_file)

def load_model(path='models/model.pkl'):
    model_file = open(path, 'rb')
    model = pickle.load(model_file)
    return model