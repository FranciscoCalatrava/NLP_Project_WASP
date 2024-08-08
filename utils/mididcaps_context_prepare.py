import pandas as pd
import numpy as np
import os
import sys
import random

from utils.from_midi_to_wav import create_wav_dataset, create_embeddings
from utils.get_arousal_valence import get_models, get_arousal_valence

def get_name_midi_files(path, percentage):
    dir_names = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    total_lenght_fataset = 0
    for a in dir_names:
        files_names = [f for f in os.listdir(path+ f'\\{a}') if os.path.isfile(os.path.join(path+ f'\\{a}', f))]
        total_lenght_fataset += len(files_names)

    total_number_context_samples = round(total_lenght_fataset * percentage)

    path_list = []

    for a in dir_names:
        print(a)
        files_names = [f for f in os.listdir(path+ f'\\{a}') if os.path.isfile(os.path.join(path+ f'\\{a}', f))]
        total_lenght_fataset += len(files_names)
        random_list = random.sample(files_names, total_number_context_samples)
        for b in random_list:
            path_list.append(path+ f"\\{a}" + f"\\{b}")
    return path_list

def extract_audio_embeddings(path):
    embeddings = {}
    for a in path:
        wav_path = create_wav_dataset(a)
        embedding = create_embeddings(wav_path)
        os.remove(wav_path) ## To not compromise the memory
        name = a.split('\\')[-1]
        embeddings[name] = embedding

def get_ground_truth_arousal_valence(embeddings):
    datasets = ['deam', 'emomusic', 'muse']
    model_type = ['musicnn']
    ground_truth = {}
    for a in datasets:
        for b in model_type:
            feature_extractor, classifier = get_models(a,b)
            ground_truth[a] = get_arousal_valence(feature_extractor, classifier, embeddings, a, b)
    
    final_ground_truth = {}
    
    for a in embeddings.keys():
        auxiliar = []
        for b in datasets:
           auxiliar.append(ground_truth[b][a])
        final_ground_truth[a] = np.mean(np.vstack(auxiliar), axis = 0)
    return final_ground_truth


def get_audio_features(embeddings):
    features = {}

    return features


def get_midi_features(paths):
    features = {}

    return features



def create_final_file(features_midi, features_audio, ground_truth):

    final_dataset = {}
    for a in features_midi.keys():
        aux= {}
        for b in features_midi[a].keys():
            aux[b] = features_midi[a][b]
        for b in features_audio[a].keys():
            aux[b] = features_audio[a][b]
        for b in ground_truth[a].keys():
            aux[b] = ground_truth[a][b]
        final_dataset[a] = aux
    
    return final_dataset

def save_file(path, data):


    print("saving data")






if __name__ == '__main__':
    print("Hello")

    path = 'dataset\lmd_full'
    get_name_midi_files(path, 0.0001)

    dict1 = {"hola":{'a':0, 'b':0},
             'adios':{'c':0, 'd':0}}

    dict2 = {"hola":{'e':0, 'f':0},
             'adios':{'g':0, 'h':0}}

    dict3 = {"hola":{'i':0, '¡j':0},
             'adios':{'k':0, 'l':0}}
    
    print(create_final_file(dict1, dict2, dict3))


