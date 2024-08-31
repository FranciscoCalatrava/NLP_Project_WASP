import numpy as np
import pandas as pd
import mido
from music21 import converter, corpus, instrument, midi, note, chord, pitch, stream
from collections import defaultdict
import pretty_midi


def read_midi(path_midi):
    midi_file = midi

def get_instruments_pretty_midi(file_path):
    midi_data = pretty_midi.PrettyMIDI(file_path)
    ins = []
    for instrument in midi_data.instruments:
        if instrument.is_drum == True:
            ins.append(128)
        else:
            ins.append(instrument.program)
    return ins

def get_instruments_from_midi(file_path):
    midi = mido.MidiFile(file_path)
    instruments = set()
    channels = set()

    instrument_durations = defaultdict(float)
    instrument_names=[]
    instrument_channels=[]
    instrument_change_times=[]
    
    for track in midi.tracks:
        active_notes = defaultdict(float)
        last_event_time = 0
        
        for msg in track:
            # Update the time since the last event
            delta_time = msg.time
            last_event_time += delta_time
            if msg.type == 'program_change':
                print(f"The program | Channel is {msg.program} | {msg.channel}")
                instruments.add(msg.program)
                channels.add(msg.channel)
            if msg.type == 'note_on' or msg.type == 'note_off':
                # Extract the instrument (channel) and note number
                channels.add(msg.channel)
                channel = msg.channel
                note = msg.note

                print(f"{channel} | {note}")
                # Calculate the duration since the last event
                duration = last_event_time - active_notes[(channel, note)]
                active_notes[(channel, note)] = last_event_time
                
                # Accumulate the duration for this instrument
                instrument_durations[channel] += duration
    new_dict=sorted(instrument_durations.items(), key=lambda x:x[1],reverse=True)

    
    return instruments

def get_final_inst_list_1(midi_file_path):
    # Dictionary to store instrument durations
    instrument_durations = defaultdict(float)
    instrument_names=[]
    instrument_channels=[]
    instrument_change_times=[]
    # Parse MIDI file
    midi = mido.MidiFile(midi_file_path)
    # Iterate through each track
    for track in midi.tracks:
        # Dictionary to store note-on events for each instrument
        active_notes = defaultdict(float)
        last_event_time = 0
        # Iterate through each event in the track
        for msg in track:
            # Update the time since the last event
            delta_time = msg.time
            last_event_time += delta_time
            if msg.type=='program_change':
                prog=msg.program
                chan=msg.channel
                # if chan==9 and prog==0:
                if chan==9 and not 111<prog<120:
                    prog=128
                if chan in instrument_channels:
                    instrument_names[instrument_channels.index(chan)]=prog #replace the existing instrument in this channel, no ambiguity!!!
                else:
                    instrument_names.append(prog)
                    instrument_channels.append(chan)
                instrument_change_times.append(msg.time)
                # print(msg.time)
            # If it's a note-on or note-off event
            if msg.type == 'note_on' or msg.type == 'note_off':
                # Extract the instrument (channel) and note number
                channel = msg.channel
                note = msg.note
                # Calculate the duration since the last event
                duration = last_event_time - active_notes[(channel, note)]
                active_notes[(channel, note)] = last_event_time
                # Accumulate the duration for this instrument
                instrument_durations[channel] += duration
    new_dict=sorted(instrument_durations.items(), key=lambda x:x[1],reverse=True)
    if len(instrument_names)>20:
        print('too many instruments in this one!')
        print(midi_file_path)
        return [], []
    sorted_instrument_list=[]
    how_many=min(5,len(set(instrument_names)))
    if how_many==0:
        return []
    add_drums=False
    if 9 not in instrument_channels:
        for rr in new_dict:
            if 9 in rr:
                add_drums=True
                break
            else:
                add_drums=False
    if add_drums:
        instrument_names.append(128)
        instrument_channels.append(9)
    for i in range(len(new_dict)):
        try:
            sorted_instrument_list.append(instrument_names[instrument_channels.index(new_dict[i][0])])
        except Exception as e:
            print(e)
            print(midi_file_path)
            return sorted_instrument_list
    return sorted_instrument_list

def get_key(music_21_object):
    return music_21_object.analyze('keys')

def get_time_signature(music_21_object):
    return music_21_object.getTimeSignatures()[0]

def get_tempo(path_midi):
    mido_object = mido.MidiFile(path_midi)
    try:
        for msg in mido_object:
            if msg.type == 'set_tempo':
                return mido_object.tempo2bpm(msg.tempo)
    except:
        print("There was an exception while getting the tempo")
        return None

def get_mode(tempo):
    if 0<tempo<400:
        dice=np.random.randint(0,2)
        if dice==0:
            tempo_marks=np.array((40, 60, 70, 90, 110, 140, 160, 210))
            tempo_caps=['Grave', 'Largo', 'Adagio', 'Andante', 'Moderato', 'Allegro', 'Vivace', 'Presto', 'Prestissimo']
            index=np.sum(tempo>tempo_marks)
            tempo_cap=tempo_caps[index]
        else:
            tempo_marks=np.array((80, 120, 160))
            tempo_caps=['Slow', 'Moderate tempo', 'Fast', 'Very fast']
            index=np.int(np.sum(tempo>tempo_marks))
            tempo_cap=tempo_caps[index]
        
        return tempo_cap
    else:
        print(f"Invalid tempo. The variable contains {tempo}")
        return None


def get_duration(path_midi):
    mido_object = mido.MidiFile(path_midi)
    try:
        return mido_object.length
    except:
        print("There was an exception while getting the duratuib")
        return None



if __name__ == "__main__":
    sorted_list = get_instruments_from_midi("dataset/midicap/midicaps/lmd_full/0/0a0b59b984e78fccd380b44938a17ad4.mid")
    print(sorted_list)


    print(get_final_inst_list_1("dataset/midicap/midicaps/lmd_full/0/0a0b59b984e78fccd380b44938a17ad4.mid"))

    print(get_instruments_pretty_midi("dataset/midicap/midicaps/lmd_full/0/0a0b59b984e78fccd380b44938a17ad4.mid"))