import pretty_midi
import pandas as pd
import numpy as np
import pretty_midi
import numpy as np
import sys
import os
import h5py



def get_data(path):
    data = pd.read_csv(path)
    absolute_path = f'{os.getcwd()}/dataset/vgmidi/normal/'
    data['midi'] = data['midi'].apply(lambda x: absolute_path + x)
    return data

def get_midi_features(file_path):
    try:
        midi_data = pretty_midi.PrettyMIDI(file_path)
        
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

        return [tempo, total_notes, average_pitch, pitch_range, std_pitch, note_density,
                average_velocity, velocity_changes, average_melodic_interval]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # Return NaN values so the row isn't empty in case of error
        return [np.nan] * 9  # Update the number as per features
    
def encode_labels(arousal, valence):
    if arousal == 1 and valence == 1:
        return 0
    elif arousal == 1 and valence == -1:
        return 1
    elif arousal == -1 and valence == 1:
        return 2
    elif arousal == -1 and valence == -1:
        return 3


def get_name(name):
    return name.split('/')[-1]


def save_features(path, final):
    with h5py.File(path+ f"vgmidi_features.h5", 'w') as hf:
        for idx, name in enumerate(final.keys()) :
            grp = hf.create_group(name)
            for feature_name in final[name].keys():
                # print(f"The file is {name}")
                # print(f"The feature is {feature_name}")
                # print(f"The data is {final[name][feature_name]}")
                sub_grp = grp.create_group(feature_name)
                sub_grp.create_dataset('data', data=final[name][feature_name])



if __name__ == "__main__":
    input_parameters = {'dataset_path': sys.argv[1],
                        }
    
    data = get_data(input_parameters['dataset_path'])
    features_df = pd.DataFrame([get_midi_features(file) for file in data['midi']],
                           columns=['tempo', 'total_notes', 'average_pitch', 'pitch_range', 'std_pitch',
                                    'note_density', 'average_velocity', 'velocity_changes', 'average_melodic_interval'])
    
    features_df['label'] = data.apply(lambda row: encode_labels(row['arousal'], row['valence']), axis=1)
    features_df['name'] = data.apply(lambda row: get_name(row['midi']), axis=1 )
    features_df.set_index('name', inplace = True)
    features = features_df.to_dict(orient='index')

    save_features(f'{os.getcwd()}/dataset/vgmidi/prepared/', features)

    

    


    