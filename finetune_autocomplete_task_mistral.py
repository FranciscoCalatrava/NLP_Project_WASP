from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer,AutoModelForTokenClassification
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments
from sklearn.metrics import classification_report, confusion_matrix
import re
import numpy as np
import h5py
import os
import wandb
import pickle
import time
import pandas as pd
import torch






def load_article_dataset():
    dataset = load_dataset('json', data_files={'article_1': 'dataset/articles/paper_1.json',
                                               'article_2': 'dataset/articles/paper_2.json',
                                               'article_3': 'dataset/articles/paper_3.json',
                                               'article_4': 'dataset/articles/paper_5.json',
                                               'article_5': 'dataset/articles/paper_6.json',
                                               'article_6': 'dataset/articles/paper_7.json',
                                               'validation':'dataset/articles/paper_4.json',})
    return dataset

def format_instruction(sample):
    return [f"""You are an AI assistant tasked with completing academic sentences based on the research articles related to sentiment (arousal and valence) in music. Your objective is to accurately predict the end of a sentence that has been started for you, based on your understanding of music and its sentiment.
        ### Input:
        {sample["input"]}

        ### Output:
        {sample["output"]}
    """]

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


##Functions that we will need

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
    inputs = tokenizer.encode(prompt, return_tensors="pt").to('cuda')
    outputs = model.generate(inputs, max_length=1000)
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

def manual_evaluation(model, tokenizer, article_number):
  '''
  answers should be a dictionary or a list with the answers from the LLM
  '''
  prompts = ["""Given this set of musical feature '{{feature}}', provide predictions for the following:
1. Arousal Level (Emotional Intensity). You only have 2 possible values which are 1 for high arousal and -1 for low arousal. Arousal reflects the level of emotional energy or intensity in the music.
2. Valence Level (Emotional Pleasantness). You only have 2 possible values which are 1 for high valence and -1 for low valence. Valence represents the degree of positivity or negativity in the emotional tone of the music.
For each prediction, provide a brief explanation that connects the specific musical features (e.g., tempo, key, mood, genre) to the expected levels of arousal and valence. Ensure the reasoning is grounded in how these features typically influence emotional perception. Format your response as a JSON object like this: <Start Respose>{"Arousal": 'predicted arousal level', "Valence": 'predicted valence level', "Confidence Level": 'numerical confidence level', "Explanation": 'brief explanation of the predictions based on the features'}<End Respose>
Please ensure to replace the placeholders in the JSON structure with actual values based on the input features and your analysis."""]

  start_time = time.time()
  path_vgmidi_features = 'dataset/vgmidi/vgmidi_features.h5'
  raw_features_vgmidi = get_feature_from_file(path_vgmidi_features)
  pre_processed_features = preprocess_dataset_for_arousal_description_confidence(raw_features_vgmidi,1)

  for indx, prompt in enumerate(prompts):
    answers = []
    llm_cache = {}
    wandb.init(project='wasp-nlp-project-finetuning', name=f"Finetuning_with_article_number_{article_number}")
    start_time = time.time()
    for dictionary in pre_processed_features[0:50]:
      feature = dictionary['features']
      print(feature)
      final_prompt = prompt.replace('{{feature}}', str(feature))
      answer, llm_cache = test_one_prompt(final_prompt, model, tokenizer, llm_cache)
      answers.append(answer)
    end_time = time.time()
    duration = end_time - start_time
    print(f"The loop took {duration} seconds to complete.")
    wandb.finish()
    with open(f"./results/Finetuning_with_article_number_{article_number}", "wb") as f:
        pickle.dump(llm_cache, f)
  return answers

def class_label(arousal, valence):
    if arousal >=0 and valence >=0:
        return 0
    elif arousal < 0 and valence >=0:
        return 1
    elif arousal < 0 and valence < 0:
        return 2
    elif arousal >= 0 and valence < 0:
        return 3

def get_anotations(path_annotations):
    data = pd.read_csv(path_annotations)
    annotations = pd.DataFrame()

    annotations['name'] = data['midi'].apply(lambda x: x.split('/')[-1])

    annotations['label'] = data.apply(lambda row: class_label(row['arousal'], row['valence']), axis=1)

    final_annotations = {}

    for indx, element in annotations.iterrows():
        final_annotations[element['name']] = element['label']

    return final_annotations

import itertools

def get_arousal_valence(answers,pre_processed_features):
  '''
  arousal_valence_list should be the values of the arousal and valence of the answers
  '''
  data_1={}
  for indx, a in enumerate(answers):
    print()
    print(indx)
    print(a)
    print()
    text = str(a)
    # Define the regular expressions for matching the values
    arousal_pattern = re.compile(r'"Arousal":\s*["\']?(-?\d+)["\']?')
    valence_pattern = re.compile(r'"Valence":\s*["\']?(-?\d+)["\']?')

    # Find all matches in the text
    arousal_values = arousal_pattern.findall(text)
    valence_values = valence_pattern.findall(text)

    print(arousal_values)
    print(valence_values)
    print(text)
    print(indx)



    # Extract the values as integers
    data_1[str(indx)] = {"Arousal": arousal_values, "Valence": valence_values}

  names= []
  for dictionary in pre_processed_features:
    names.append(dictionary['music_name'])

  print(names)
  print(data_1.keys())
  data_list = []
  final_data_class = {}
  for name_midi,name_data in zip(names,data_1.keys()):
    print(name_midi)
    print(name_data)

    final_data_class[name_midi] = data_1[name_data]

  return final_data_class


def classification_evaluation(generations):
  '''
  This function should return the classification report and the confusion matrix
  '''
  path_annotations = './vgmidi_labelled.csv'
  annotations = get_anotations(path_annotations)
  print(annotations)
  annotations = dict(itertools.islice(annotations.items(), 50))
  print(annotations)
  print(generations)



  ground_truth_list = []
  predictions_list = []

  for a in list(annotations.keys())[0:50]:
    print(a)
    print(generations[a[:-4]])
    if generations[a[:-4]] != {}:
      print(a)
      print(generations[a[:-4]]['Arousal'])
      print(float(generations[a[:-4]]['Valence'][0]))
      prediciton = class_label(float(generations[a[:-4]]['Arousal'][0]), float(generations[a[:-4]]['Valence'][0]))
      predictions_list.append(prediciton)
      ground_truth_list.append(annotations[a])
  classification_report_1 =classification_report(y_true=np.array(ground_truth_list), y_pred=np.array(predictions_list), labels=[0,1,2,3])
  confusion_matrix_1 = confusion_matrix(y_true=np.array(ground_truth_list), y_pred=np.array(predictions_list), labels=[0,1,2,3])
  print(ground_truth_list)
  print(predictions_list)
  print(classification_report_1)
  print(confusion_matrix_1)
  return classification_report, confusion_matrix


if __name__ == '__main__':
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    access_token = ""
    model,tokenizer = get_model(model_name, access_token)
    tokenizer.pad_token = tokenizer.eos_token
    NUMBER_OF_PAPERS = 6

    dataset = load_article_dataset()

    peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    for a in range(1, NUMBER_OF_PAPERS + 1):
        model_args = TrainingArguments(
            output_dir=f"./logs/mistral-7b-style_1_{a}",
            report_to="wandb",
            logging_dir="log",
            num_train_epochs=10,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            logging_steps=1,
            save_strategy="epoch",
            learning_rate=2e-4,
            bf16=True,
            tf32=False,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            disable_tqdm=True,
        )

        # Supervised Fine-Tuning Trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset[f'article_{a}'],
            eval_dataset=dataset['validation'],
            peft_config=peft_config,
            max_seq_length=2048,
            tokenizer=tokenizer,
            packing=False,
            formatting_func=format_instruction,
            args=model_args,
        )

        # train
        trainer.train()
        answers = manual_evaluation(model, tokenizer, a)

        # path_vgmidi_features = '/content/drive/MyDrive/datasets/vgmidi/vgmidi_features.h5'
        # raw_features_vgmidi = get_feature_from_file(path_vgmidi_features)
        # pre_processed_features = preprocess_dataset_for_arousal_description_confidence(raw_features_vgmidi,1)
        # arousal_list = get_arousal_valence(answers, pre_processed_features[0:50])
        # classification_evaluation(arousal_list)



