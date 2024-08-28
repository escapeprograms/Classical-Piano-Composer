import torch
import numpy as np

from torch import nn

import torch.nn.functional as F


#assemble training data in a space-efficient categorical format
def prepare_sequences(notes, durations, n_vocab, d_vocab, device="cpu"):
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

    #number of examples
    n_patterns = len(notes_input)

    # reshape the input into a format compatible with LSTM layers
    notes_input = torch.tensor(notes_input)
    notes_output = torch.tensor(notes_output)

    durations_input = torch.tensor(durations_input)
    durations_output = torch.tensor(durations_output)

    network_input = torch.stack((notes_input, durations_input), dim=2) #TODO: stack notes_input and durations_input
    network_output = torch.stack((notes_output, durations_output), dim= 1)

    return (network_input.long(), network_output.long())




#apply one-hot encoding to the categorical data generated above
def encode_data(network_data, n_vocab, d_vocab, device="cpu", type="input"):
    if type == "input":
        notes_input = F.one_hot(network_data[:, :, 0], num_classes=n_vocab)
        durations_input = F.one_hot(network_data[:, :, 1], num_classes=d_vocab)
        network_data = torch.cat((notes_input, durations_input), dim=2).to(device, dtype=torch.float32)
    else: #type == output
        notes_output = F.one_hot(network_data[:, 0], num_classes=n_vocab)
        durations_output = F.one_hot(network_data[:, 1], num_classes=d_vocab)
        network_data = torch.cat((notes_output, durations_output), dim=1).to(device, dtype=torch.float32)
    return network_data
