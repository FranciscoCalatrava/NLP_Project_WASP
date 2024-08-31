import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import h5py
import numpy as np
import pandas as pd
import os
import time




def get_model(model_name, access_token):
  # Configure 4-bit quantization
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True,
  )

  # Load the model with the defined BitsAndBytesConfig
  llm_model = AutoModelForCausalLM.from_pretrained(
      model_name,
      quantization_config=bnb_config,
      token=access_token  # Use 'token' instead of the deprecated 'use_auth_token'
  )

  # Load the tokenizer
  tokenizer = AutoTokenizer.from_pretrained(model_name, token = access_token)
  return llm_model, tokenizer


def get_feature_from_file(path):
    final_dataset = {}
    # Open the HDF5 file in read mode
    with h5py.File(path, 'r') as hf:
        # Iterate through the top-level groups
        for group_name in hf.keys():
            sub_data = {}
            # print(f"Top-level group: {group_name}")
            group = hf[group_name]

            # Iterate through subgroups
            for subgroup_name in group.keys():
                # print(f"  Subgroup: {subgroup_name}")
                subgroup = group[subgroup_name]

                # Access the dataset within the subgroup
                data = subgroup['data'][()]
                # print(f"    Data: {data}")
                sub_data[subgroup_name] = data
            # print(sub_data)
            final_dataset[str(group_name)] = sub_data
    return final_dataset


def get_subgroup_features(group,features):
  if group == 0:
    new_sub_group = {"average velocity": ("%.2f" % features['average_velocity']),
                         "note_density": ("%.2f" % features['note_density']),
                         "pitch_range": ("%.2f" % features['pitch_range']),
                         "tempo": ("%.2f" % features['tempo']),
                         "total_notes" : ("%.2f" % features['total_notes']),
                         "velocity_changes": ("%.2f" % features['velocity_changes'])}
  elif group ==1:
    new_sub_group = {"average instesity of notes": ("%.2f" % features['average_velocity']),
                     "standard deviation instesity of notes": ("%.2f" % features['velocity_changes']),
                      "note_density": ("%.2f" % features['note_density']),
                      "tempo": ("%.2f" % features['tempo']),
                      "pitch_range": ("%.2f" % features['pitch_range']),
                      "total_notes" : ("%.2f" % features['total_notes']),
                      "key": features['key'],
                        #  "mood": list(features['mood']),
                        #  "genre": list(features['genre']),
                        #  "arousal_from_audio": features['muse'][1],
                        #  "valence_from_audio": features['muse'][0]
                     }

  # elif group == 2:


  # elif group == 3:


  # elif group == 4:
  return new_sub_group



# Step 1: Preprocess the dataset
def preprocess_dataset_for_arousal_description_confidence(dataset, feature_subgroup):
    preprocessed_data = []

    for music_name, features in dataset.items():

      sub_group = get_subgroup_features(feature_subgroup,features)

      preprocessed_data.append({
          'music_name': music_name,
          'features': sub_group,
      })

    return preprocessed_data


def test_one_prompt(prompt, model, tokenizer, cache):
  if prompt in cache:
      answer = cache[prompt]
  else:
    # Generate new response
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=7000)
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)


    answer = full_output.strip()

    # Cache the result
    cache[prompt] = answer


    # # Extract the answer by removing the prompt part
    answer = full_output.split("<Start Respose>")[-1].strip()

    # Cache the result
    cache[prompt] = answer
  # Log all cached prompts and their answers to wandb
  table = wandb.Table(columns=["prompt", "generation"])
  for prompt, answer in cache.items():
      table.add_data(prompt, answer)
  wandb.log({'All Inferences': table})
  return answer,cache


if __name__ == '__main__':
    # Define your model name and access token
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    access_token = ""
    model,tokennizer = get_model(model_name, access_token)
    path_vgmidi_features = 'vgmidi_features .h5'
    raw_features_vgmidi = get_feature_from_file(path_vgmidi_features)
    pre_processed_features = preprocess_dataset_for_arousal_description_confidence(raw_features_vgmidi,1)

    print(pre_processed_features)
    

    prompts = [
"""
We will provide you 15 examples of features paired with the values of arousal and valence from the dataset MIDICaps to improve the classification performance on another datasets called VGMIDI. These 2 datasets contains different types of music. The MidiCaps dataset is a large-scale dataset of 168,385 midi music files with descriptive text captions, and a set of extracted musical features. This dataset contains many music pieces with a wide
range of instruments (piano, Drums,String Ensemble,Acoustic Guitar,Electric Bass,Acoustic Bass,Electric Guitar, Fretless Bass, Clean Electric Guitar ,Synth Pad,Distortion Guitar, Overdriven Guitar, Synth Strings,Choir Aahs, Flute, Brass Sectio), genre (electronic, pop, classical, rock, soundtrack, jazz) and tempos (from 40 to 250 bpm aproximately). VGMIDI is a dataset of piano arrangements of video game soundtracks generally with fasts tempos.
The list of features examples paired to the arousal and valence values from MIDICaps are:
The set of musical feature: {'average instesity of notes': '76.53', 'standard deviation instesity of notes': '12.47', 'note_density': '10.94', 'tempo': '194.16', 'pitch_range': '55.00', 'total_notes': '2232.00', 'velocity_changes': '12.47', 'key': b'D minor'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '81.17', 'standard deviation instesity of notes': '20.97', 'note_density': '6.78', 'tempo': '120.59', 'pitch_range': '58.00', 'total_notes': '1200.00', 'velocity_changes': '20.97', 'key': b'Eb major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '127.00', 'standard deviation instesity of notes': '0.00', 'note_density': '18.57', 'tempo': '211.16', 'pitch_range': '37.00', 'total_notes': '3693.00', 'velocity_changes': '0.00', 'key': b'C minor'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '82.49', 'standard deviation instesity of notes': '16.08', 'note_density': '23.29', 'tempo': '243.67', 'pitch_range': '92.00', 'total_notes': '2793.00', 'velocity_changes': '16.08', 'key': b'C minor'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '100.00', 'standard deviation instesity of notes': '0.00', 'note_density': '5.97', 'tempo': '178.70', 'pitch_range': '55.00', 'total_notes': '553.00', 'velocity_changes': '0.00', 'key': b'Bb major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '101.60', 'standard deviation instesity of notes': '13.24', 'note_density': '8.59', 'tempo': '241.70', 'pitch_range': '47.00', 'total_notes': '1478.00', 'velocity_changes': '13.24', 'key': b'D minor'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '76.42', 'standard deviation instesity of notes': '17.73', 'note_density': '9.57', 'tempo': '249.61', 'pitch_range': '53.00', 'total_notes': '2599.00', 'velocity_changes': '17.73', 'key': b'D major'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '100.00', 'standard deviation instesity of notes': '0.00', 'note_density': '2.93', 'tempo': '108.33', 'pitch_range': '31.00', 'total_notes': '103.00', 'velocity_changes': '0.00', 'key': b'G minor'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '125.54', 'standard deviation instesity of notes': '2.57', 'note_density': '16.21', 'tempo': '193.89', 'pitch_range': '48.00', 'total_notes': '3352.00', 'velocity_changes': '2.57', 'key': b'E minor'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '77.70', 'standard deviation instesity of notes': '8.73', 'note_density': '5.64', 'tempo': '188.20', 'pitch_range': '40.00', 'total_notes': '1046.00', 'velocity_changes': '8.73', 'key': b'D major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '98.75', 'standard deviation instesity of notes': '9.65', 'note_density': '10.65', 'tempo': '273.99', 'pitch_range': '62.00', 'total_notes': '2464.00', 'velocity_changes': '9.65', 'key': b'F major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '75.82', 'standard deviation instesity of notes': '14.51', 'note_density': '18.91', 'tempo': '198.19', 'pitch_range': '77.00', 'total_notes': '2301.00', 'velocity_changes': '14.51', 'key': b'G# minor'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '92.41', 'standard deviation instesity of notes': '14.69', 'note_density': '19.17', 'tempo': '215.93', 'pitch_range': '55.00', 'total_notes': '1725.00', 'velocity_changes': '14.69', 'key': b'A major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '63.07', 'standard deviation instesity of notes': '3.48', 'note_density': '3.38', 'tempo': '150.77', 'pitch_range': '55.00', 'total_notes': '110.00', 'velocity_changes': '3.48', 'key': b'E major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '51.04', 'standard deviation instesity of notes': '24.36', 'note_density': '16.66', 'tempo': '146.16', 'pitch_range': '38.00', 'total_notes': '2710.00', 'velocity_changes': '24.36', 'key': b'E major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '100.00', 'standard deviation instesity of notes': '0.00', 'note_density': '7.40', 'tempo': '240.00', 'pitch_range': '36.00', 'total_notes': '698.00', 'velocity_changes': '0.00', 'key': b'A minor'} classified with the arousal value of -1 and the valence value of -1
Now, Given this set of musical feature '{{feature}}' from VGMIDI, provide predictions for the following:
1. Arousal Level (Emotional Intensity). You only have 2 possible values which are 1 for high arousal and -1 for low arousal. Arousal reflects the level of emotional energy or intensity in the music.
2. Valence Level (Emotional Pleasantness). You only have 2 possible values which are 1 for high valence and -1 for low valence. Valence represents the degree of positivity or negativity in the emotional tone of the music.
For each prediction, provide a brief explanation that connects the specific musical features (e.g., tempo, key, mood, genre) to the expected levels of arousal and valence. Ensure the reasoning is grounded in how these features typically influence emotional perception. Format your response as a JSON object like this: <Start Respose>{"Arousal": 'predicted arousal level', "Valence": 'predicted valence level', "Confidence Level": 'numerical confidence level', "Explanation": 'brief explanation of the predictions based on the features'}<End Respose>
Please ensure to replace the placeholders in the JSON structure with actual values based on the input features and your analysis.
""" ,

"""We will share 15 examples from the MIDICaps dataset, where each example includes features along with their arousal and valence values. This information is intended to enhance classification performance on the VGMIDI dataset. MIDICaps is an extensive dataset with over 168,000 MIDI files that cover a wide emotional spectrum through various musical features, instruments, and genres. The music in this dataset includes everything from serene classical pieces to high-energy electronic tracks. In contrast, VGMIDI consists mainly of fast-paced piano arrangements from video game soundtracks, which might display a different emotional range. Here are the selected examples from MIDICaps:
The set of musical feature: {'average instesity of notes': '76.53', 'standard deviation instesity of notes': '12.47', 'note_density': '10.94', 'tempo': '194.16', 'pitch_range': '55.00', 'total_notes': '2232.00', 'velocity_changes': '12.47', 'key': b'D minor'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '81.17', 'standard deviation instesity of notes': '20.97', 'note_density': '6.78', 'tempo': '120.59', 'pitch_range': '58.00', 'total_notes': '1200.00', 'velocity_changes': '20.97', 'key': b'Eb major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '127.00', 'standard deviation instesity of notes': '0.00', 'note_density': '18.57', 'tempo': '211.16', 'pitch_range': '37.00', 'total_notes': '3693.00', 'velocity_changes': '0.00', 'key': b'C minor'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '82.49', 'standard deviation instesity of notes': '16.08', 'note_density': '23.29', 'tempo': '243.67', 'pitch_range': '92.00', 'total_notes': '2793.00', 'velocity_changes': '16.08', 'key': b'C minor'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '100.00', 'standard deviation instesity of notes': '0.00', 'note_density': '5.97', 'tempo': '178.70', 'pitch_range': '55.00', 'total_notes': '553.00', 'velocity_changes': '0.00', 'key': b'Bb major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '101.60', 'standard deviation instesity of notes': '13.24', 'note_density': '8.59', 'tempo': '241.70', 'pitch_range': '47.00', 'total_notes': '1478.00', 'velocity_changes': '13.24', 'key': b'D minor'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '76.42', 'standard deviation instesity of notes': '17.73', 'note_density': '9.57', 'tempo': '249.61', 'pitch_range': '53.00', 'total_notes': '2599.00', 'velocity_changes': '17.73', 'key': b'D major'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '100.00', 'standard deviation instesity of notes': '0.00', 'note_density': '2.93', 'tempo': '108.33', 'pitch_range': '31.00', 'total_notes': '103.00', 'velocity_changes': '0.00', 'key': b'G minor'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '125.54', 'standard deviation instesity of notes': '2.57', 'note_density': '16.21', 'tempo': '193.89', 'pitch_range': '48.00', 'total_notes': '3352.00', 'velocity_changes': '2.57', 'key': b'E minor'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '77.70', 'standard deviation instesity of notes': '8.73', 'note_density': '5.64', 'tempo': '188.20', 'pitch_range': '40.00', 'total_notes': '1046.00', 'velocity_changes': '8.73', 'key': b'D major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '98.75', 'standard deviation instesity of notes': '9.65', 'note_density': '10.65', 'tempo': '273.99', 'pitch_range': '62.00', 'total_notes': '2464.00', 'velocity_changes': '9.65', 'key': b'F major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '75.82', 'standard deviation instesity of notes': '14.51', 'note_density': '18.91', 'tempo': '198.19', 'pitch_range': '77.00', 'total_notes': '2301.00', 'velocity_changes': '14.51', 'key': b'G# minor'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '92.41', 'standard deviation instesity of notes': '14.69', 'note_density': '19.17', 'tempo': '215.93', 'pitch_range': '55.00', 'total_notes': '1725.00', 'velocity_changes': '14.69', 'key': b'A major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '63.07', 'standard deviation instesity of notes': '3.48', 'note_density': '3.38', 'tempo': '150.77', 'pitch_range': '55.00', 'total_notes': '110.00', 'velocity_changes': '3.48', 'key': b'E major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '51.04', 'standard deviation instesity of notes': '24.36', 'note_density': '16.66', 'tempo': '146.16', 'pitch_range': '38.00', 'total_notes': '2710.00', 'velocity_changes': '24.36', 'key': b'E major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '100.00', 'standard deviation instesity of notes': '0.00', 'note_density': '7.40', 'tempo': '240.00', 'pitch_range': '36.00', 'total_notes': '698.00', 'velocity_changes': '0.00', 'key': b'A minor'} classified with the arousal value of -1 and the valence value of -1
Now, Given this set of musical feature '{{feature}}' from VGMIDI, provide predictions for the following:
1. Arousal Level (Emotional Intensity). You only have 2 possible values which are 1 for high arousal and -1 for low arousal. Arousal reflects the level of emotional energy or intensity in the music.
2. Valence Level (Emotional Pleasantness). You only have 2 possible values which are 1 for high valence and -1 for low valence. Valence represents the degree of positivity or negativity in the emotional tone of the music.
For each prediction, provide a brief explanation that connects the specific musical features (e.g., tempo, key, mood, genre) to the expected levels of arousal and valence. Ensure the reasoning is grounded in how these features typically influence emotional perception. Format your response as a JSON object like this: <Start Respose>{"Arousal": 'predicted arousal level', "Valence": 'predicted valence level', "Confidence Level": 'numerical confidence level', "Explanation": 'brief explanation of the predictions based on the features'}<End Respose>
Please ensure to replace the placeholders in the JSON structure with actual values based on the input features and your analysis.
""" ,

"""
We will provide you with 15 feature examples paired with arousal and valence values from the MIDICaps dataset. These examples are intended to enhance classification on a different dataset called VGMIDI. Although both datasets involve music, they differ significantly in content and style. MIDICaps is a large dataset featuring a variety of instruments, genres, and tempos, offering a broad musical landscape. On the other hand, VGMIDI is focused on piano arrangements from video game soundtracks, which tend to have quicker tempos and may present different emotional cues. By analyzing these examples from MIDICaps, you can learn to better classify the music in VGMIDI. Here are the feature examples:
The set of musical feature: {'average instesity of notes': '76.53', 'standard deviation instesity of notes': '12.47', 'note_density': '10.94', 'tempo': '194.16', 'pitch_range': '55.00', 'total_notes': '2232.00', 'velocity_changes': '12.47', 'key': b'D minor'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '81.17', 'standard deviation instesity of notes': '20.97', 'note_density': '6.78', 'tempo': '120.59', 'pitch_range': '58.00', 'total_notes': '1200.00', 'velocity_changes': '20.97', 'key': b'Eb major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '127.00', 'standard deviation instesity of notes': '0.00', 'note_density': '18.57', 'tempo': '211.16', 'pitch_range': '37.00', 'total_notes': '3693.00', 'velocity_changes': '0.00', 'key': b'C minor'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '82.49', 'standard deviation instesity of notes': '16.08', 'note_density': '23.29', 'tempo': '243.67', 'pitch_range': '92.00', 'total_notes': '2793.00', 'velocity_changes': '16.08', 'key': b'C minor'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '100.00', 'standard deviation instesity of notes': '0.00', 'note_density': '5.97', 'tempo': '178.70', 'pitch_range': '55.00', 'total_notes': '553.00', 'velocity_changes': '0.00', 'key': b'Bb major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '101.60', 'standard deviation instesity of notes': '13.24', 'note_density': '8.59', 'tempo': '241.70', 'pitch_range': '47.00', 'total_notes': '1478.00', 'velocity_changes': '13.24', 'key': b'D minor'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '76.42', 'standard deviation instesity of notes': '17.73', 'note_density': '9.57', 'tempo': '249.61', 'pitch_range': '53.00', 'total_notes': '2599.00', 'velocity_changes': '17.73', 'key': b'D major'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '100.00', 'standard deviation instesity of notes': '0.00', 'note_density': '2.93', 'tempo': '108.33', 'pitch_range': '31.00', 'total_notes': '103.00', 'velocity_changes': '0.00', 'key': b'G minor'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '125.54', 'standard deviation instesity of notes': '2.57', 'note_density': '16.21', 'tempo': '193.89', 'pitch_range': '48.00', 'total_notes': '3352.00', 'velocity_changes': '2.57', 'key': b'E minor'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '77.70', 'standard deviation instesity of notes': '8.73', 'note_density': '5.64', 'tempo': '188.20', 'pitch_range': '40.00', 'total_notes': '1046.00', 'velocity_changes': '8.73', 'key': b'D major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '98.75', 'standard deviation instesity of notes': '9.65', 'note_density': '10.65', 'tempo': '273.99', 'pitch_range': '62.00', 'total_notes': '2464.00', 'velocity_changes': '9.65', 'key': b'F major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '75.82', 'standard deviation instesity of notes': '14.51', 'note_density': '18.91', 'tempo': '198.19', 'pitch_range': '77.00', 'total_notes': '2301.00', 'velocity_changes': '14.51', 'key': b'G# minor'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '92.41', 'standard deviation instesity of notes': '14.69', 'note_density': '19.17', 'tempo': '215.93', 'pitch_range': '55.00', 'total_notes': '1725.00', 'velocity_changes': '14.69', 'key': b'A major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '63.07', 'standard deviation instesity of notes': '3.48', 'note_density': '3.38', 'tempo': '150.77', 'pitch_range': '55.00', 'total_notes': '110.00', 'velocity_changes': '3.48', 'key': b'E major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '51.04', 'standard deviation instesity of notes': '24.36', 'note_density': '16.66', 'tempo': '146.16', 'pitch_range': '38.00', 'total_notes': '2710.00', 'velocity_changes': '24.36', 'key': b'E major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '100.00', 'standard deviation instesity of notes': '0.00', 'note_density': '7.40', 'tempo': '240.00', 'pitch_range': '36.00', 'total_notes': '698.00', 'velocity_changes': '0.00', 'key': b'A minor'} classified with the arousal value of -1 and the valence value of -1
Now, Given this set of musical feature '{{feature}}' from VGMIDI, provide predictions for the following:
1. Arousal Level (Emotional Intensity). You only have 2 possible values which are 1 for high arousal and -1 for low arousal. Arousal reflects the level of emotional energy or intensity in the music.
2. Valence Level (Emotional Pleasantness). You only have 2 possible values which are 1 for high valence and -1 for low valence. Valence represents the degree of positivity or negativity in the emotional tone of the music.
For each prediction, provide a brief explanation that connects the specific musical features (e.g., tempo, key, mood, genre) to the expected levels of arousal and valence. Ensure the reasoning is grounded in how these features typically influence emotional perception. Format your response as a JSON object like this: <Start Respose>{"Arousal": 'predicted arousal level', "Valence": 'predicted valence level', "Confidence Level": 'numerical confidence level', "Explanation": 'brief explanation of the predictions based on the features'}<End Respose>
Please ensure to replace the placeholders in the JSON structure with actual values based on the input features and your analysis.
""" ,

"""
We will share 15 examples from the MIDICaps dataset, each paired with arousal and valence values, to aid in improving the classification accuracy on the VGMIDI dataset. While MIDICaps is a broad dataset with music pieces spanning multiple genres, instruments, and tempos, VGMIDI is more focused, featuring piano arrangements of video game music that often exhibit rapid tempos. Despite these differences, understanding the emotional attributes of MIDICaps can help bridge the gap and improve your ability to classify VGMIDI. Here are the selected examples from MIDICaps:
The set of musical feature: {'average instesity of notes': '76.53', 'standard deviation instesity of notes': '12.47', 'note_density': '10.94', 'tempo': '194.16', 'pitch_range': '55.00', 'total_notes': '2232.00', 'velocity_changes': '12.47', 'key': b'D minor'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '81.17', 'standard deviation instesity of notes': '20.97', 'note_density': '6.78', 'tempo': '120.59', 'pitch_range': '58.00', 'total_notes': '1200.00', 'velocity_changes': '20.97', 'key': b'Eb major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '127.00', 'standard deviation instesity of notes': '0.00', 'note_density': '18.57', 'tempo': '211.16', 'pitch_range': '37.00', 'total_notes': '3693.00', 'velocity_changes': '0.00', 'key': b'C minor'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '82.49', 'standard deviation instesity of notes': '16.08', 'note_density': '23.29', 'tempo': '243.67', 'pitch_range': '92.00', 'total_notes': '2793.00', 'velocity_changes': '16.08', 'key': b'C minor'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '100.00', 'standard deviation instesity of notes': '0.00', 'note_density': '5.97', 'tempo': '178.70', 'pitch_range': '55.00', 'total_notes': '553.00', 'velocity_changes': '0.00', 'key': b'Bb major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '101.60', 'standard deviation instesity of notes': '13.24', 'note_density': '8.59', 'tempo': '241.70', 'pitch_range': '47.00', 'total_notes': '1478.00', 'velocity_changes': '13.24', 'key': b'D minor'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '76.42', 'standard deviation instesity of notes': '17.73', 'note_density': '9.57', 'tempo': '249.61', 'pitch_range': '53.00', 'total_notes': '2599.00', 'velocity_changes': '17.73', 'key': b'D major'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '100.00', 'standard deviation instesity of notes': '0.00', 'note_density': '2.93', 'tempo': '108.33', 'pitch_range': '31.00', 'total_notes': '103.00', 'velocity_changes': '0.00', 'key': b'G minor'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '125.54', 'standard deviation instesity of notes': '2.57', 'note_density': '16.21', 'tempo': '193.89', 'pitch_range': '48.00', 'total_notes': '3352.00', 'velocity_changes': '2.57', 'key': b'E minor'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '77.70', 'standard deviation instesity of notes': '8.73', 'note_density': '5.64', 'tempo': '188.20', 'pitch_range': '40.00', 'total_notes': '1046.00', 'velocity_changes': '8.73', 'key': b'D major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '98.75', 'standard deviation instesity of notes': '9.65', 'note_density': '10.65', 'tempo': '273.99', 'pitch_range': '62.00', 'total_notes': '2464.00', 'velocity_changes': '9.65', 'key': b'F major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '75.82', 'standard deviation instesity of notes': '14.51', 'note_density': '18.91', 'tempo': '198.19', 'pitch_range': '77.00', 'total_notes': '2301.00', 'velocity_changes': '14.51', 'key': b'G# minor'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '92.41', 'standard deviation instesity of notes': '14.69', 'note_density': '19.17', 'tempo': '215.93', 'pitch_range': '55.00', 'total_notes': '1725.00', 'velocity_changes': '14.69', 'key': b'A major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '63.07', 'standard deviation instesity of notes': '3.48', 'note_density': '3.38', 'tempo': '150.77', 'pitch_range': '55.00', 'total_notes': '110.00', 'velocity_changes': '3.48', 'key': b'E major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '51.04', 'standard deviation instesity of notes': '24.36', 'note_density': '16.66', 'tempo': '146.16', 'pitch_range': '38.00', 'total_notes': '2710.00', 'velocity_changes': '24.36', 'key': b'E major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '100.00', 'standard deviation instesity of notes': '0.00', 'note_density': '7.40', 'tempo': '240.00', 'pitch_range': '36.00', 'total_notes': '698.00', 'velocity_changes': '0.00', 'key': b'A minor'} classified with the arousal value of -1 and the valence value of -1
Now, Given this set of musical feature '{{feature}}' from VGMIDI, provide predictions for the following:
1. Arousal Level (Emotional Intensity). You only have 2 possible values which are 1 for high arousal and -1 for low arousal. Arousal reflects the level of emotional energy or intensity in the music.
2. Valence Level (Emotional Pleasantness). You only have 2 possible values which are 1 for high valence and -1 for low valence. Valence represents the degree of positivity or negativity in the emotional tone of the music.
For each prediction, provide a brief explanation that connects the specific musical features (e.g., tempo, key, mood, genre) to the expected levels of arousal and valence. Ensure the reasoning is grounded in how these features typically influence emotional perception. Format your response as a JSON object like this: <Start Respose>{"Arousal": 'predicted arousal level', "Valence": 'predicted valence level', "Confidence Level": 'numerical confidence level', "Explanation": 'brief explanation of the predictions based on the features'}<End Respose>
Please ensure to replace the placeholders in the JSON structure with actual values based on the input features and your analysis.
""" 
"""
These are examples of musical features linked to arousal and valence values:
The set of musical feature: {'average instesity of notes': '76.53', 'standard deviation instesity of notes': '12.47', 'note_density': '10.94', 'tempo': '194.16', 'pitch_range': '55.00', 'total_notes': '2232.00', 'velocity_changes': '12.47', 'key': b'D minor'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '81.17', 'standard deviation instesity of notes': '20.97', 'note_density': '6.78', 'tempo': '120.59', 'pitch_range': '58.00', 'total_notes': '1200.00', 'velocity_changes': '20.97', 'key': b'Eb major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '127.00', 'standard deviation instesity of notes': '0.00', 'note_density': '18.57', 'tempo': '211.16', 'pitch_range': '37.00', 'total_notes': '3693.00', 'velocity_changes': '0.00', 'key': b'C minor'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '82.49', 'standard deviation instesity of notes': '16.08', 'note_density': '23.29', 'tempo': '243.67', 'pitch_range': '92.00', 'total_notes': '2793.00', 'velocity_changes': '16.08', 'key': b'C minor'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '100.00', 'standard deviation instesity of notes': '0.00', 'note_density': '5.97', 'tempo': '178.70', 'pitch_range': '55.00', 'total_notes': '553.00', 'velocity_changes': '0.00', 'key': b'Bb major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '101.60', 'standard deviation instesity of notes': '13.24', 'note_density': '8.59', 'tempo': '241.70', 'pitch_range': '47.00', 'total_notes': '1478.00', 'velocity_changes': '13.24', 'key': b'D minor'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '76.42', 'standard deviation instesity of notes': '17.73', 'note_density': '9.57', 'tempo': '249.61', 'pitch_range': '53.00', 'total_notes': '2599.00', 'velocity_changes': '17.73', 'key': b'D major'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '100.00', 'standard deviation instesity of notes': '0.00', 'note_density': '2.93', 'tempo': '108.33', 'pitch_range': '31.00', 'total_notes': '103.00', 'velocity_changes': '0.00', 'key': b'G minor'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '125.54', 'standard deviation instesity of notes': '2.57', 'note_density': '16.21', 'tempo': '193.89', 'pitch_range': '48.00', 'total_notes': '3352.00', 'velocity_changes': '2.57', 'key': b'E minor'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '77.70', 'standard deviation instesity of notes': '8.73', 'note_density': '5.64', 'tempo': '188.20', 'pitch_range': '40.00', 'total_notes': '1046.00', 'velocity_changes': '8.73', 'key': b'D major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '98.75', 'standard deviation instesity of notes': '9.65', 'note_density': '10.65', 'tempo': '273.99', 'pitch_range': '62.00', 'total_notes': '2464.00', 'velocity_changes': '9.65', 'key': b'F major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '75.82', 'standard deviation instesity of notes': '14.51', 'note_density': '18.91', 'tempo': '198.19', 'pitch_range': '77.00', 'total_notes': '2301.00', 'velocity_changes': '14.51', 'key': b'G# minor'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '92.41', 'standard deviation instesity of notes': '14.69', 'note_density': '19.17', 'tempo': '215.93', 'pitch_range': '55.00', 'total_notes': '1725.00', 'velocity_changes': '14.69', 'key': b'A major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '63.07', 'standard deviation instesity of notes': '3.48', 'note_density': '3.38', 'tempo': '150.77', 'pitch_range': '55.00', 'total_notes': '110.00', 'velocity_changes': '3.48', 'key': b'E major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '51.04', 'standard deviation instesity of notes': '24.36', 'note_density': '16.66', 'tempo': '146.16', 'pitch_range': '38.00', 'total_notes': '2710.00', 'velocity_changes': '24.36', 'key': b'E major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '100.00', 'standard deviation instesity of notes': '0.00', 'note_density': '7.40', 'tempo': '240.00', 'pitch_range': '36.00', 'total_notes': '698.00', 'velocity_changes': '0.00', 'key': b'A minor'} classified with the arousal value of -1 and the valence value of -1
Now, Given this set of musical feature '{{feature}}' from VGMIDI, provide predictions for the following:
1. Arousal Level (Emotional Intensity). You only have 2 possible values which are 1 for high arousal and -1 for low arousal. Arousal reflects the level of emotional energy or intensity in the music.
2. Valence Level (Emotional Pleasantness). You only have 2 possible values which are 1 for high valence and -1 for low valence. Valence represents the degree of positivity or negativity in the emotional tone of the music.
For each prediction, provide a brief explanation that connects the specific musical features (e.g., tempo, key, mood, genre) to the expected levels of arousal and valence. Ensure the reasoning is grounded in how these features typically influence emotional perception. Format your response as a JSON object like this: <Start Respose>{"Arousal": 'predicted arousal level', "Valence": 'predicted valence level', "Confidence Level": 'numerical confidence level', "Explanation": 'brief explanation of the predictions based on the features'}<End Respose>
Please ensure to replace the placeholders in the JSON structure with actual values based on the input features and your analysis.
""" 

    ]

    start_time = time.time()
    llm_cache = {}
    print(len(prompts))

    for indx, prompt in enumerate(prompts):
        answers = []
        wandb.init(project='wasp-nlp-project-final', name=f"baseline_shift_adaptation_prompt_15_examples_{indx}")
        start_time = time.time()
        for dictionary in pre_processed_features:
            feature = dictionary['features']
            print(feature)
            final_prompt = prompt.replace('{{feature}}', str(feature))
            answer, llm_cache = test_one_prompt(final_prompt, model, tokennizer, llm_cache)
            answers.append(answer)
        end_time = time.time()
        duration = end_time - start_time
        print(f"The loop took {duration} seconds to complete.")
        wandb.finish()
        # Open or create an HDF5 file
        with h5py.File(f'baseline_shift_adaptation_prompt_15_examples_{indx}.h5', 'w') as h5file:
            for key, value in llm_cache.items():
                # Save the key and value
                # Converting the value to a numpy string type before saving
                h5file.create_dataset(key, data=np.string_(value))