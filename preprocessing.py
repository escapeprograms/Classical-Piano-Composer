import torch
import numpy as np

from torch import nn

import torch.nn.functional as F


#assemble training data in a readable format 
def prepare_sequences(notes, durations, n_vocab, d_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names/note lengths
    pitch_names = sorted(set(item for item in notes))
    note_lengths = sorted(set(item for item in durations))

     # create a dictionary to map pitches/note lengths to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitch_names))
    duration_to_int = dict((length, number) for number, length in enumerate(note_lengths))

    notes_input = []
    notes_output = []

    durations_input = []
    durations_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        #process sequence of notes
        notes_sequence_in = notes[i:i + sequence_length]
        notes_sequence_out = notes[i + sequence_length]
        notes_input.append([note_to_int[char] for char in notes_sequence_in])
        notes_output.append(note_to_int[notes_sequence_out])

        #process sequence of durations
        durations_sequence_in = durations[i:i + sequence_length]
        durations_sequence_out = durations[i + sequence_length]
        durations_input.append([duration_to_int[dur] for dur in durations_sequence_in])
        durations_output.append(duration_to_int[durations_sequence_out])

    n_patterns = len(notes_input)

    # reshape the input into a format compatible with LSTM layers
    notes_input = np.reshape(notes_input, (n_patterns, sequence_length, 1))
    # normalize input
    notes_input = notes_input / float(n_vocab)

    notes_output = F.one_hot(torch.tensor(notes_output)) #one hot encoding

    durations_input = np.reshape(durations_input, (n_patterns, sequence_length, 1))
    # normalize input
    durations_input = durations_input / float(d_vocab)

    durations_output = F.one_hot(torch.tensor(durations_output))

    network_input = np.concatenate((notes_input, durations_input), axis=2) #TODO: stack notes_input and durations_input
    network_output = torch.cat((notes_output, durations_output), dim=1)

    return (network_input, network_output)