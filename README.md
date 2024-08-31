# NLP_Project_WASP
Final NLP project for WASP course


Here is a brief explanation of the code setup:

To run the code, you need to install the necessary requirements, download the datasets, and obtain some prediction models from Essentia. I will break down the process here:

## Requirements

To install the requirements, create your own virtual environment using Python 3.10.12. Once set up, run the following command:

`pip3 install -r requirements.txt`

If you experience issues with the package `essentia-tensorflow`, such as:

"from essentia.standard import MonoLoader, TensorflowPredict2D, TensorflowPredictMusiCNN,TensorflowPredictEffnetDiscogs
ImportError: cannot import name 'TensorflowPredict2D' from 'essentia.standard' (./music_sentiment_prediction/nlp/lib/python3.10/site-packages/essentia/standard.py)"

Perform a pip uninstall of both `essentia` and `essentia-tensorflow`, and then reinstall only `essentia-tensorflow`. This may resolve potential compatibility issues.
## Models

### Classification models for Arousal Valence from audio

The models are organized in the following directories:

- classifier_model
  - deam
    - audioset-vggish
    - msd-musicnn
  - muse
    - audioset-vggish
    - msd-musicnn
  - emomusic
    - audioset-vggish
    - msd-musicnn

Models can be downloaded from the Essentia webpage:

deam: https://essentia.upf.edu/models/classification-heads/deam/
emomusic: https://essentia.upf.edu/models/classification-heads/emomusic/
muse: https://essentia.upf.edu/models/classification-heads/muse/

### Embeddings models for the classification Arousal Valenve for audio

The folders are structured as follows:

- embedding_model
  - deam
    - audioset-vggish
    - msd-musicnn
  - muse
    - audioset-vggish
    - msd-musicnn
  - emomusic
    - audioset-vggish
    - msd-music


Models can be downloaded from the Essentia webpage:

deam: https://essentia.upf.edu/models/classification-heads/deam/
emomusic: https://essentia.upf.edu/models/classification-heads/emomusic/
muse: https://essentia.upf.edu/models/classification-heads/muse/

### Genre models from essentia

The folders are structured as follows:

- genre_models
- mood_models

Models can be downloaded from the Essentia webpage:

Genre: https://essentia.upf.edu/models/classification-heads/mtg_jamendo_genre/
Mood: https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/
## Datasets

The datasets used in this work are VGMIDI and MIDICaps. Below are the URLs for these web pages:

- MIDICaps: https://huggingface.co/datasets/amaai-lab/MidiCaps/tree/main
- VGMIDI: https://github.com/lucasnfe/vgmidi/tree/master

I have organised the datasets as follows:

- dataset
    - midicap
        - normal
        - prepared
            - labels
    - vgmidi
        - normal
        - prepared


In the normal folder, I have extracted the content from the datasets.

Additionally, to run this code, you will need the file to synthesize the audio. You can download it from the following URL:

https://member.keymusician.com/Member/FluidR3_GM/index.html

You should place the file named 'FluidR3_GM.sf2' in the working directory (./).


Once you have completed all these processes, you are ready to run the files. I will now explain the purpose of each file:

    - get_features_from_midi_vgmidi.py: This file extracts features from MIDI files labeled by VGMIDI and saves them in the ./dataset/vgmidi/prepared/ directory as vgmidi_features.h5.

    - mididcaps_context_prepare.py: This file extracts features from 'X' random samples chosen from MIDICaps. You should specify the number of samples as an argument like this:

        `python3 mididcaps_context_prepare.py 15`

        The features will be saved in the ./dataset/midicap/prepared/ directory as midicaps_features_{X Samples}.h5.

    - zero_shot_learning_mistral_7_b.py: This is the code used for zero-shot learning.

    - few_shot_learning_mistral.py: This code is used for few-shot learning. You should manually integrate the features from MIDICaps in the prompt.

    - domain_adaptation_mistral.py: This code is used for domain adaptation from MIDICaps to VGMIDI.

    - finetune_autocomplete_task_mistral: This code is used for finetuning Mistral in the autocomplete task. The dataset for this autocomplete task is specified within the code.

In all cases, you will need the access token for Mistral.