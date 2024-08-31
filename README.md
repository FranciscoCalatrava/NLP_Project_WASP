# NLP_Project_WASP
Final NLP project for WASP course


Here it is a small explanation about the code:

In order to run the code you need to install the requirements, download the datasets, and download some prediction models from essentia. I am going to break it down here:

## Requirements

To install the requirements you should create your own virtual environmnet. For this work I have used Python 3.10.12. Once you have it, just run:

`pip3 install -r requirements.txt`

I have experienced some difficulties regarding to the package essentia-tensorflow. In cases get an error like:

"from essentia.standard import MonoLoader, TensorflowPredict2D, TensorflowPredictMusiCNN,TensorflowPredictEffnetDiscogs
ImportError: cannot import name 'TensorflowPredict2D' from 'essentia.standard' (./music_sentiment_prediction/nlp/lib/python3.10/site-packages/essentia/standard.py)"

Just do a pip uninstall essetia & pip uninstall essentia-tensorflow and after reinstall just essential-tensorflow. There might be some incompatibilities between these two. This is my guess.

## Models

### Classification models for Arousal Valence from audio

I have organise the folders as it follows:

- classifier_model
-- deam
--- audioset-vggish
--- msd-musicnn
-- muse
--- audioset-vggish
--- msd-musicnn 
-- emomusic
--- audioset-vggish
--- msd-musicnn 

You can get the models from the essentia webpage

deam: https://essentia.upf.edu/models/classification-heads/deam/
emomusic: https://essentia.upf.edu/models/classification-heads/emomusic/
muse: https://essentia.upf.edu/models/classification-heads/muse/

### Embeddings models for the classification Arousal Valenve for audio

I have organise the folders as it follows:

- embedding_model
-- deam
--- audioset-vggish
--- msd-musicnn
-- muse
--- audioset-vggish
--- msd-musicnn 
-- emomusic
--- audioset-vggish
--- msd-musicnn 


You can get the models from the essentia webpage

deam: https://essentia.upf.edu/models/classification-heads/deam/
emomusic: https://essentia.upf.edu/models/classification-heads/emomusic/
muse: https://essentia.upf.edu/models/classification-heads/muse/

### Genre models from essentia

I have organised the foleders as it follows:

- genre_models
- mood_models

You can download the files from:
Genre: https://essentia.upf.edu/models/classification-heads/mtg_jamendo_genre/
Mood: https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/
## Datasets

The dataset used in this work are VGMIDI and MIDICaps. Here are the URLs to the web pages:

- MIDICaps: https://huggingface.co/datasets/amaai-lab/MidiCaps/tree/main
- VGMIDI: https://github.com/lucasnfe/vgmidi/tree/master

I have organised the datasets as follows:

- dataset
-- midicap
--- normal
--- prepared
---- labels
-- vgmidi
--- normal
--- prepared


In the normal folder, I have extracted the content form the datasets.


Additionally, to run this code you will need the file to synthessize the audio. You can downloade it from the following URL:

https://member.keymusician.com/Member/FluidR3_GM/index.html

You should put the file called "FluidR3_GM.sf2" in the working directory (./)



Once you have done all this process, you are ready to run the files. I am going to explain the purpose of each file now:

- get_features_from_midi_vgmidi.py -> This file will extract the features from midi labelled midi files from VGMIDI and it will save it in the ./dataset/vgmidi/prepared/ directory as vgmidi_features.h5
- mididcaps_context_prepare.py -> This file will extract the features from 'X' random samples choosen from MIDICaps. You should give the number of samples as an argument as follows:

`python3 mididcaps_context_prepare.py 15`

The features will be saved in the directory ./dataset/midicap/prepared/ directory as midicaps_features_{X Samples}.h5
