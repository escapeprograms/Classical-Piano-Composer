""" This module generates notes for a midi file using the
    trained neural network """

from models import MusicLSTM

import pickle
import numpy as np
from music21 import instrument, note, stream, chord, tempo

import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessing import prepare_sequences, encode_data

from tqdm import tqdm

output_length = 200 #set how many notes to generate

#load cuda device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: {device}")
else:
    print("CUDA is not available")
    device = torch.device("cpu")



def generate():
    """ Generate a piano midi file """
    #load the notes used to train the model
    with open('data/notes', 'rb') as filepath:
        notes, durations = pickle.load(filepath)

    # Get all pitch names
    pitch_names = sorted(set(item for item in notes))
    note_lengths = sorted(set(item for item in durations))

    # Get all pitch names
    n_vocab = len(set(notes))
    d_vocab = len(set(durations))

    print("Preparing sequence data...")
    start = np.random.randint(0, len(notes)-output_length-1)
    network_input, _ = prepare_sequences(notes[start: start+output_length], durations, n_vocab, d_vocab)

    model = torch.load("models/music_model.pt")
    model.eval()

    prediction_output = generate_notes(model, network_input[0], pitch_names, note_lengths, n_vocab, d_vocab)
    create_midi(prediction_output)


def generate_notes(model, initial_sequence, pitch_names, note_lengths, n_vocab, d_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction

    int_to_note = dict((number, note) for number, note in enumerate(pitch_names))
    int_to_duration = dict((number, length) for number, length in enumerate(note_lengths))

    #reshape the randomly chosen starter sequence to (batch size = 1, sequence length = shape[0], features = shape[1])
    init_sequence = torch.reshape(initial_sequence, (1, initial_sequence.shape[0], initial_sequence.shape[1]))
    pattern = encode_data(init_sequence, n_vocab, d_vocab, device, type="input")
    prediction_output = []
    # generate some notes notes
    for note_index in tqdm(range(output_length), desc="generating song"):

        prediction = model(pattern)
        note_index = torch.multinomial(F.softmax(prediction[0, 0:n_vocab], dim=0), 1).item() #picks a note from a distribution instead of just the most probable note
        note = int_to_note[note_index]
        
        duration_index = torch.argmax(prediction[0, n_vocab:]).item()
        duration = int_to_duration[duration_index]
        prediction_output.append((note, duration))
        #reshape the new note to (batch size = 1, sequence length = 1, features = shape[1])
        new_pattern = torch.reshape(torch.tensor([note_index, duration_index]), (1, 1, initial_sequence.shape[1]))

        new_pattern = encode_data(new_pattern, n_vocab, d_vocab, device, type="input")
        pattern = torch.cat((pattern, new_pattern), dim=1)
        pattern = pattern[:, 1:, :]


    return prediction_output

def create_midi(prediction_output):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern, length in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                new_note.quarterLength = length
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            new_note.quarterLength = length
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += length

    
    midi_stream = stream.Stream(output_notes)
    midi_stream.append(tempo.MetronomeMark(number=120, beatUnit='quarter'))

    midi_stream.write('midi', fp='test_output.mid')
    print("Done! Check test_output.mid")



generate()
