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
    with h5py.File(path, 'r') as hf:
        for group_name in hf.keys():
            sub_data = {}
            # print(f"Top-level group: {group_name}")
            group = hf[group_name]
            for subgroup_name in group.keys():
                # print(f"  Subgroup: {subgroup_name}")
                subgroup = group[subgroup_name]
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
    outputs = model.generate(inputs, max_length=6200)
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
"""Here are some examples that link musical features to their corresponding arousal and valence levels:
The set of musical feature: {'average instesity of notes': '103.59', 'standard deviation instesity of notes': '19.21', 'note_density': '13.63', 'tempo': '196.82', 'pitch_range': '63.00', 'total_notes': '4236.00', 'velocity_changes': '19.21', 'key': b'B major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '103.00', 'standard deviation instesity of notes': '0.00', 'note_density': '12.21', 'tempo': '172.78', 'pitch_range': '57.00', 'total_notes': '6158.00', 'velocity_changes': '0.00', 'key': b'C major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '102.00', 'standard deviation instesity of notes': '0.00', 'note_density': '5.30', 'tempo': '235.65', 'pitch_range': '23.00', 'total_notes': '360.00', 'velocity_changes': '0.00', 'key': b'D major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '53.95', 'standard deviation instesity of notes': '24.50', 'note_density': '8.31', 'tempo': '217.96', 'pitch_range': '84.00', 'total_notes': '1479.00', 'velocity_changes': '24.50', 'key': b'E major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '100.00', 'standard deviation instesity of notes': '0.00', 'note_density': '8.24', 'tempo': '160.00', 'pitch_range': '53.00', 'total_notes': '875.00', 'velocity_changes': '0.00', 'key': b'C major'} classified with the arousal value of -1 and the valence value of -1
The set of musical feature: {'average instesity of notes': '84.29', 'standard deviation instesity of notes': '14.62', 'note_density': '21.98', 'tempo': '169.77', 'pitch_range': '51.00', 'total_notes': '2498.00', 'velocity_changes': '14.62', 'key': b'D minor'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '106.39', 'standard deviation instesity of notes': '7.25', 'note_density': '13.46', 'tempo': '190.03', 'pitch_range': '65.00', 'total_notes': '1547.00', 'velocity_changes': '7.25', 'key': b'Eb major'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '63.24', 'standard deviation instesity of notes': '30.35', 'note_density': '12.05', 'tempo': '197.98', 'pitch_range': '65.00', 'total_notes': '2551.00', 'velocity_changes': '30.35', 'key': b'A minor'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '101.86', 'standard deviation instesity of notes': '12.92', 'note_density': '10.39', 'tempo': '210.00', 'pitch_range': '72.00', 'total_notes': '3686.00', 'velocity_changes': '12.92', 'key': b'B minor'} classified with the arousal value of 1 and the valence value of 1
The set of musical feature: {'average instesity of notes': '100.37', 'standard deviation instesity of notes': '18.08', 'note_density': '27.85', 'tempo': '229.21', 'pitch_range': '75.00', 'total_notes': '5986.00', 'velocity_changes': '18.08', 'key': b'C minor'} classified with the arousal value of 1 and the valence value of 1
Now, given this new set of musical feature '{{feature}}', provide predictions for the following:
1. Arousal Level (Emotional Intensity). You only have 2 possible values which are 1 for high arousal and -1 for low arousal. Arousal reflects the level of emotional energy or intensity in the music.
2. Valence Level (Emotional Pleasantness). You only have 2 possible values which are 1 for high valence and -1 for low valence. Valence represents the degree of positivity or negativity in the emotional tone of the music.
For each prediction, provide a brief explanation that connects the specific musical features (e.g., tempo, key, mood, genre) to the expected levels of arousal
and valence. Ensure the reasoning is grounded in how these features typically influence emotional perception. Format your response as a JSON object like this: <Start Respose>{"Arousal": 'predicted arousal level', "Valence": 'predicted valence level', "Confidence Level": 'numerical confidence level', "Explanation": 'brief explanation of the predictions based on the features'}<End Respose>
Please ensure to replace the placeholders in the JSON structure with actual values based on the input features and your analysis."""]

    start_time = time.time()
    llm_cache = {}

    for indx, prompt in enumerate(prompts):
        answers = []
        wandb.init(project='wasp-nlp-project', name=f"few_shot_prompting_10")
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
        with h5py.File('few_shot_prompting_10.h5', 'w') as h5file:
            for key, value in llm_cache.items():
                # Save the key and value
                # Converting the value to a numpy string type before saving
                h5file.create_dataset(key, data=np.string_(value))