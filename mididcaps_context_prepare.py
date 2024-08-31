
import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import random
import h5py


import pretty_midi
import json

from essentia.standard import MonoLoader, TensorflowPredict2D, TensorflowPredictMusiCNN,TensorflowPredictEffnetDiscogs
from music21 import converter, corpus, instrument, midi, note, chord, pitch, stream,meter



from utils.from_midi_to_wav import create_wav_dataset, create_embeddings
from utils.get_arousal_valence import get_models, get_arousal_valence

def get_name_midi_files(path, samples):
    dir_names = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    total_lenght_fataset = 0
    for a in dir_names:
        files_names = [f for f in os.listdir(path+ f'/{a}') if os.path.isfile(os.path.join(path+ f'/{a}', f))]
        total_lenght_fataset += len(files_names)

    total_number_context_samples = samples

    path_list = []

    for a in dir_names:
        print(a)
        files_names = [f for f in os.listdir(path+ f'/{a}') if os.path.isfile(os.path.join(path+ f'/{a}', f))]
        total_lenght_fataset += len(files_names)
        for b in files_names:
            path_list.append(path+ f"/{a}" + f"/{b}")
    random_files = random.sample(path_list, total_number_context_samples)
    return random_files

def extract_audio_embeddings(path):
    embeddings = {}
    for a in path:
        wav_path = create_wav_dataset(a, 'FluidR3_GM.sf2')
        embedding = create_embeddings(wav_path)
        os.remove(wav_path) ## To not compromise the memory
        name = a.split('/')[-1]
        embeddings[name] = embedding
    return embeddings

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
        final_ground_truth[a] = {'arousal_valence_ground_truth': np.mean(np.vstack(auxiliar), axis = 0)}
    return final_ground_truth


def get_mtg_tags(embeddings,tag_model,tag_json,max_num_tags=5,tag_threshold=0.01):

    with open(tag_json, 'r') as json_file:
        metadata = json.load(json_file)
    predictions = tag_model(embeddings)
    mean_act=np.mean(predictions,0)
    ind = np.argpartition(mean_act, -max_num_tags)[-max_num_tags:]
    tags=[]
    confidence_score=[]
    for i in ind:
        print(metadata['classes'][i] + str(mean_act[i]))
        if mean_act[i]>tag_threshold:
            tags.append(metadata['classes'][i])
            confidence_score.append(mean_act[i])
    ind=np.argsort(-np.array(confidence_score))
    tags = [tags[i] for i in ind]
    confidence_score=np.round((np.array(confidence_score)[ind]).tolist(),4).tolist()

    return tags, confidence_score


def get_audio_features_one_file(embeddings):
    features = {}
    feature_extractor = TensorflowPredictEffnetDiscogs(graphFilename='genre_models/discogs-effnet-bs64-1.pb', output="PartitionedCall:1")
    genmodel = TensorflowPredict2D(graphFilename='genre_models/mtg_jamendo_genre-discogs-effnet-1.pb')
    moodmodel = TensorflowPredict2D(graphFilename='mood_models/mtg_jamendo_moodtheme-discogs-effnet-1.pb')
    final_emb = feature_extractor(embeddings)
    mood_tags, mood_cs = get_mtg_tags(final_emb,moodmodel,'mood_models/mtg_jamendo_moodtheme-discogs-effnet-1.json',max_num_tags=5,tag_threshold=0.02)
    genre_tags, genre_cs = get_mtg_tags(final_emb,genmodel,'genre_models/mtg_jamendo_genre-discogs-effnet-1.json',max_num_tags=4,tag_threshold=0.05)    
    # chord_estimator = Chordino()           
    # chords = chord_estimator.extract(audio_file)
    # chords_out = [(x.chord, x.timestamp) for x in chords[1:-1]]
    # #chord summary
    # ch_name=[]
    # ch_time=[]
    # for ch in chords_out:
    #     ch_name.append(ch[0])
    #     ch_time.append(ch[1])
    # if len(ch_name)<3:
    #     final_seq=ch_name
    #     final_count=1
    # else:
    #     final_seq, final_count = give_me_final_seq(ch_name)
    # if final_seq is not None:
    #     if len(final_seq)==4:
    #         if final_seq[0]==final_seq[2] and final_seq[1]==final_seq[3]:
    #             final_seq=final_seq[0:2]
    # chord_summary=[final_seq,final_count]
    features['mood_tags'] = mood_tags
    features['mood_confidence_score'] = mood_cs

    features['genre_tags'] = genre_tags
    features['genre_confidence_score'] = genre_cs
    return features

def get_audio_features(embeddings):
    features = {}
    for name in embeddings.keys():
        features[name] = get_audio_features_one_file(embeddings[name])
    
    return features

def get_midi_key(midi_path):
    try:
                # Load the MIDI file
        midi = converter.parse(midi_path)

        # Filter out percussion parts
        non_percussion_parts = stream.Stream()
        for part in midi.parts:
            if not any(isinstance(instr, instrument.Percussion) for instr in part.getInstruments()):
                non_percussion_parts.append(part)

        # Combine the non-percussion parts
        combined_parts = non_percussion_parts.flatten()

        # Analyze the key
        key_signature = combined_parts.analyze('key')
        return key_signature
    except Exception as e:
        print(f"An error occurred while processing {midi_path}: {e}")
        return None

def get_keys(midi):
    keys = midi.analyze('keys')
    return keys

def read_midi(path):
    mf = midi.MidiFile()
    mf.open(path)
    mf.read()
    mf.close()
    return midi.translate.midiFileToStream(mf)

def get_midi_features_one_file(file_path):
    try:
        midi_data = pretty_midi.PrettyMIDI(file_path)

        try:
            res_key = get_midi_key(file_path)
            key = res_key.tonic.name + " " + res_key.mode
        except:
            key = None
        #key postprocessing
        if key is None:
            key=key
        elif '-' in key:
            key=key.replace('-','b')
        else:
            key=key
        
        # Calculating basic features
        tempo = midi_data.estimate_tempo()
        total_notes = sum([len(instrument.notes) for instrument in midi_data.instruments if not instrument.is_drum])
        
        # Pitch features
        pitches = [note.pitch for instrument in midi_data.instruments for note in instrument.notes if not instrument.is_drum]
        average_pitch = np.mean(pitches) if pitches else 0
        pitch_range = np.max(pitches) - np.min(pitches) if pitches else 0
        std_pitch = np.std(pitches) if pitches else 0
        
        # Rhythm features
        note_density = total_notes / midi_data.get_end_time() if midi_data.get_end_time() > 0 else 0
        
        # Dynamics
        velocities = [note.velocity for instrument in midi_data.instruments for note in instrument.notes if not instrument.is_drum]
        average_velocity = np.mean(velocities) if velocities else 0
        velocity_changes = np.std(velocities) if velocities else 0

        # Melodic intervals
        melodic_intervals = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for i in range(len(instrument.notes) - 1):
                    melodic_intervals.append(instrument.notes[i+1].pitch - instrument.notes[i].pitch)
        average_melodic_interval = np.mean(melodic_intervals) if melodic_intervals else 0

        features = {}
        features['key'] = key
        features['tempo'] = tempo
        features['total_notes'] = total_notes
        features['average_pitch'] = average_pitch
        features['pitch_range'] = pitch_range
        features['std_pitch'] = std_pitch
        features['note_density'] = note_density
        features['average_velocity'] = average_velocity
        features['velocity_changes'] = velocity_changes
        features['average_melodic_interval'] = average_melodic_interval

        print(features)

        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # Return NaN values so the row isn't empty in case of error
        return [np.nan] * 9  # Update the number as per features
    



    
def get_midi_features_dataset(paths_files, number):
    midi_features = {}

    real_path_files = []
    should_break = False

    for a in paths_files:
        name = a.split('/')[-1]
        aux = get_midi_features_one_file(a)
        if aux['key'] != None:
            midi_features[name] = get_midi_features_one_file(a)
            real_path_files.append(a)
            if len(midi_features) == number:
                should_break =True
                break
        if should_break == True:
            break

    return midi_features,real_path_files




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

def save_file(path, final,samples):
    with h5py.File(path + f"midicaps_features_{samples}.h5", 'w') as hf:
        for idx, name in enumerate(final.keys()) :
            grp = hf.create_group(name)
            for feature_name in final[name].keys():
                print(f"The file is {name}")
                print(f"The feature is {feature_name}")
                print(f"The data is {final[name][feature_name]}")
                sub_grp = grp.create_group(feature_name)
                sub_grp.create_dataset('data', data=final[name][feature_name])
    print("saving data")


def read_data(file_paht):
    final_data = {}
    # Open the HDF5 file in read mode
    with h5py.File(file_paht+'vgmidi_features.h5', 'r') as hf:
        # Iterate through the top-level groups
        for group_name in hf.keys():
            aux = {}
            print(f"Top-level group: {group_name}")
            group = hf[group_name]

            # Iterate through subgroups
            for subgroup_name in group.keys():
                
                print(f"  Subgroup: {subgroup_name}")
                subgroup = group[subgroup_name]

                # Access the dataset within the subgroup
                data = subgroup['data'][()]
                print(f"    Data: {data}")
                aux[subgroup_name] = data
        final_data[group_name] = aux
    return final_data

def deep_equal(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False  # Keys must match exactly

    for key in dict1:
        val1, val2 = dict1[key], dict2[key]
        # Check if both values are dictionaries
        if isinstance(val1, dict) and isinstance(val2, dict):
            if not deep_equal(val1, val2):
                return False
        else:
            if val1 != val2:
                return False
    return True







if __name__ == '__main__':
    print("Hello")

    samples = int(sys.argv[1])

    path = 'dataset/lmd_full'
    paths_list = get_name_midi_files(path, 500)
    midi_features,paths_list = get_midi_features_dataset(paths_list, samples)
    embeddings = extract_audio_embeddings(paths_list)
    audio_features = get_audio_features(embeddings)
    ground_truth = get_ground_truth_arousal_valence(embeddings)
    final_dataset = create_final_file(midi_features,audio_features,ground_truth)
    save_file('dataset/midicap/final/',final_dataset, samples)
    data = read_data('dataset/midicap/final/')

    # print(deep_equal(final_dataset,data))