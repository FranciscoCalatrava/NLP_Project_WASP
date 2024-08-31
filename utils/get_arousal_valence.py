from essentia.standard import TensorflowPredictMusiCNN, TensorflowPredict2D
import numpy as np
import pandas as pd
import os
import h5py
import sys


def get_data(path):
    embeddings = {}

    with h5py.File(path+"/embeddings/embedding.h5", 'r') as hf:
        names_files = hf.keys()
        for a in names_files:
            grp = hf[a]
            data = grp['data'][:]
            embeddings[a] = data
    return embeddings
    
def get_models(dataset, type):

    sub_models = {'musicnn': 'msd-musicnn',
                  'vgg': 'audioset-vggish'}

    paths_embedding = f"./embedding_model/{dataset}/{sub_models[type]}/{sub_models[type]}.pb"
    paths_prediction = f"./classifier_model/{dataset}/{sub_models[type]}/{dataset}-{sub_models[type]}-2.pb"

    print(paths_embedding)
    print(paths_prediction)

    embedding_model = TensorflowPredictMusiCNN(graphFilename=paths_embedding, output="model/dense/BiasAdd")
    model = TensorflowPredict2D(graphFilename=paths_prediction, output="model/Identity")

    return embedding_model, model

def get_arousal_valence(embedding_model, model, data, dataset, type):
    labels = {}
    for a in data.keys():
        predictions = model(embedding_model(data[a]))
        results = np.mean(predictions.squeeze(), axis=0)
        results = (results - 5) / 4
        labels[a] = results

    # with h5py.File(path+f"labels/labels_{dataset}_{type}.h5", 'w') as hf:
    #     for name in labels.keys() :
    #                 grp = hf.create_group(name)
    #                 grp.create_dataset('data', data=labels[name])
    return labels




if __name__ == "__main__":

    input_parameters = {"path": sys.argv[1],
                        "dataset": sys.argv[2],
                        "type": sys.argv[3]}

    embeddings = get_data(input_parameters['path'])

    embedding, prediction = get_models(input_parameters['dataset'],input_parameters['type'])

    get_arousal_valence(embedding, prediction, embeddings, input_parameters['dataset'], input_parameters['type'])


