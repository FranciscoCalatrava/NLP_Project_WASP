import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import h5py
import numpy as np
import pandas as pd
import os
import pickle




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
      token=access_token 
  )

  # Load the tokenizer
  tokenizer = AutoTokenizer.from_pretrained(model_name, token = access_token)
  return llm_model, tokenizer


def get_feature_from_file(path):
  final_dataset = {}
  with h5py.File(path, 'r') as hf:
      for group_name in hf.keys():
          sub_data = {}
          group = hf[group_name]
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



if __name__ == "__main__":
   # Define your model name and access token
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    access_token = "" ##If you want to run this, you need the access token
    model,tokennizer = get_model(model_name, access_token)
    path_vgmidi_features = './vgmidi_features.h5'
    raw_features_vgmidi = get_feature_from_file(path_vgmidi_features)
    pre_processed_features = preprocess_dataset_for_arousal_description_confidence(raw_features_vgmidi,1)

    import time

    # Initialize the cache dictionary
    cache_file = "llm_cache_new.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            llm_cache = pickle.load(f)
    else:
        llm_cache = {}

    prompts = [

    """Given the musical feature '{{feature}}', predict the following:
    1. Arousal level
    2. Valence level
    Format your response as a JSON object like this: <Start Respose>{"Arousal": 'predicted arousal level', "Valence": 'predicted valence level', "Confidence Level": 'numerical confidence level'}<End Respose> Please ensure to replace the placeholders in the JSON structure with actual values based on the input features and your analysis.""",

    """Given the musical feature '{{feature}}', predict the following:
    1. Arousal level
    2. Valence level
    Provide a short reasoning for these predictions. Format your response as a JSON object like this: <Start Respose>{"Arousal": 'predicted arousal level', "Valence": 'predicted valence level', "Confidence Level": 'numerical confidence level', "Explanation": 'brief explanation of the predictions based on the features'}<End Respose> Please ensure to replace the placeholders in the JSON structure with actual values based on the input features and your analysis.""",

    """Given the musical feature '{{feature}}', predict the following:
    1. Arousal level (range -1 to 1)
    2. Valence level (range -1 to 1)
    Provide a short reasoning for these predictions. Format your response as a JSON object like this: <Start Respose>{"Arousal": 'predicted arousal level', "Valence": 'predicted valence level', "Confidence Level": 'numerical confidence level', "Explanation": 'brief explanation of the predictions based on the features'}<End Respose> Please ensure to replace the placeholders in the JSON structure with actual values based on the input features and your analysis.""",


    """Given this set of musical feature '{{feature}}', provide predictions for the following:
    1. Arousal Level (Emotional Intensity). You only have 2 possible values which are 1 for high arousal and -1 for low arousal. Arousal reflects the level of emotional energy or intensity in the music.
    2. Valence Level (Emotional Pleasantness). You only have 2 possible values which are 1 for high valence and -1 for low valence. Valence represents the degree of positivity or negativity in the emotional tone of the music.
    For each prediction, provide a brief explanation that connects the specific musical features (e.g., tempo, key, mood, genre) to the expected levels of arousal and valence. Ensure the reasoning is grounded in how these features typically influence emotional perception. Format your response as a JSON object like this: <Start Respose>{"Arousal": 'predicted arousal level', "Valence": 'predicted valence level', "Confidence Level": 'numerical confidence level', "Explanation": 'brief explanation of the predictions based on the features'}<End Respose>
    Please ensure to replace the placeholders in the JSON structure with actual values based on the input features and your analysis."""
    ]

    start_time = time.time()

    for indx, prompt in enumerate(prompts):
        answers = []
        llm_cache = {}
        wandb.init(project='wasp-nlp-project-final', name=f"prompts_from_simple_to_complex_{indx}")
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
        with open(f"./zero_shot_learning/prompts_from_simple_to_complex_{indx}", "wb") as f:
            pickle.dump(llm_cache, f)
