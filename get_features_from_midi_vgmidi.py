import pretty_midi
import pandas as pd
import numpy as np
import pretty_midi
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import sys



def get_data(path):
    data = pd.read_csv(path)
    absolute_path = '/home/calatrava/Documents/PhD/Courses/DeepLearningNLP/final_project/NLP_Project_WASP/dataset/vgmidi/'
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




if __name__ == "__main__":
    input_parameters = {'dataset_path': sys.argv[1],
                        }
    
    data = get_data(input_parameters['dataset_path'])
    features_df = pd.DataFrame([get_midi_features(file) for file in data['midi']],
                           columns=['tempo', 'total_notes', 'average_pitch', 'pitch_range', 'std_pitch',
                                    'note_density', 'average_velocity', 'velocity_changes', 'average_melodic_interval'])
    
    features_df['label'] = data.apply(lambda row: encode_labels(row['arousal'], row['valence']), axis=1)


    