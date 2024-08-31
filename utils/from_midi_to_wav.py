from midi2audio import FluidSynth
from essentia.standard import MonoLoader
import sys
import random
import os
import numpy as np
import h5py


def get_paths(samples,dataset_path):
    # Get a list of all files in the directory
    dir_names = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    number_folders = len(dir_names)

    number_of_samples_per_folder = samples // number_folders

    print(number_of_samples_per_folder)

    final_list_paths = []

    for a in dir_names:
        files_names = [f for f in os.listdir(dataset_path+ f'/{a}') if os.path.isfile(os.path.join(dataset_path+ f'/{a}', f))]
        # print(files_names)
        random_list = random.sample(files_names, number_of_samples_per_folder)
        for b in random_list:
            final_list_paths.append(dataset_path+ f'/{a}/{b}')

    return final_list_paths


def create_wav_dataset(a, soundfont_path):
    fs = FluidSynth(soundfont_path)
    name = a.split("/")[-1]
    fs.midi_to_audio(a, f"{os.getcwd()}/dataset/midicap/prepared/{name[:-4]}.wav")
    return f"{os.getcwd()}/dataset/midicap/prepared/{name[:-4]}.wav"


def create_embeddings(path):
    audio = MonoLoader(filename = path, sampleRate=16000, resampleQuality=4)()
    return audio
    




if __name__ == "__main__":

    input_parameters = {
        "soundfont_path": sys.argv[1],
        "dataset_path": sys.argv[2],
        "samples":int(sys.argv[3]),
        "seed": int(sys.argv[4])
    }

    # samples = get_paths(input_parameters['samples'], input_parameters['dataset_path'])


    # create_wav_dataset(samples, input_parameters['soundfont_path'])

    create_embeddings()
    print(get_paths(input_parameters['samples'], input_parameters['dataset_path'])[0:5])