from essentia.standard import TensorflowPredictMusiCNN, TensorflowPredict2D
import numpy as np
import pandas as pd
import os
import h5py
import sys


def get_data(path):
    # dir_names = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    # all_files_names = []
    # for a in dir_names:
    #     files_names = [f for f in os.listdir(dataset_path+ f'/{a}') if os.path.isfile(os.path.join(dataset_path+ f'/{a}', f))]
    #     for b in files_names:
    #         all_files_names.append(b)
    

    embeddings = {}

    with h5py.File(path+"/embeddings/embedding.h5", 'r') as hf:
        names_files = hf.keys()
        for a in names_files:
            grp = hf[a]
            data = grp['data'][:]
            embeddings[a] = data
    return embeddings
    
def get_models(embedding_model, prediction_model):
    embedding_model = TensorflowPredictMusiCNN(graphFilename=embedding_model, output="model/dense/BiasAdd")
    model = TensorflowPredict2D(graphFilename=prediction_model, output="model/Identity")
    return embedding_model, model

def get_arousal_valence(embedding_model, model, data):
    labels = {}
    path = "dataset/midicap/midicaps/prepared/"
    for a in data.keys():
        predictions = model(embedding_model(data[a]))
        results = np.mean(predictions.squeeze(), axis=0)
        results = (results - 5) / 4
        labels[a] = results

    print(labels)
    with h5py.File(path+"labels/labels.h5", 'w') as hf:
        for name in labels.keys() :
                    grp = hf.create_group(name)
                    grp.create_dataset('data', data=labels[name])


    








if __name__ == "__main__":

    input_parameters = {"path": sys.argv[1],
                        "embedding_model": sys.argv[2],
                        "prediction_model": sys.argv[3]}

    embeddings = get_data(input_parameters['path'])

    embedding, prediction = get_models(input_parameters['embedding_model'], input_parameters['prediction_model'])

    get_arousal_valence(embedding, prediction, embeddings)


