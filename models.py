import torch
import numpy as np

from torch import nn


class MusicLSTM(nn.Module):
    def __init__(self, input_len, hidden_size, num_note_classes, num_duration_classes, num_layers):
        super(MusicLSTM, self).__init__()
        self.input_len = input_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_note_classes = num_note_classes
        self.num_duration_classes = num_duration_classes

        #input len = 1 (1 note, 1 duration)
        self.note_lstm = nn.LSTM(1, hidden_size, num_layers, dropout= 0.2, batch_first=True)
        self.duration_lstm = nn.LSTM(1, hidden_size, num_layers, dropout= 0.2, batch_first=True)
        self.lin = nn.Linear(hidden_size*2, self.num_note_classes + self.num_duration_classes)
    
    def forward(self, X):
        note_hidden_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        note_cell_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        duration_hidden_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        duration_cell_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size)


        input_notes = X[:,:,:1]
        input_durations = X[:,:,1:] #TODO: figure out if the dimensions are correct

        out_notes, _ = self.note_lstm(input_notes, (note_hidden_states, note_cell_states))
        out_durations, _ = self.duration_lstm(input_durations, (duration_hidden_states, duration_cell_states))

        #extract final hidden state for both note pitch and duration; selects [all batches, last time step, all features], then concat
        out = torch.cat((out_notes[:, -1, :], out_durations[:, -1, :]), dim=1) 
        out = self.lin(out)
        return out

class MusicLSTM2(nn.Module):
    def __init__(self, input_len, hidden_size, num_note_classes, num_duration_classes, num_layers):
        super(MusicLSTM2, self).__init__()
        self.input_len = input_len #currently unused, but the "input_len" is self.num_note_classes + self.num_duration_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_note_classes = num_note_classes
        self.num_duration_classes = num_duration_classes

        self.lstm = nn.LSTM(self.num_note_classes + self.num_duration_classes, hidden_size, num_layers, dropout= 0.3, batch_first=True)
        self.lin = nn.Linear(hidden_size, self.num_note_classes + self.num_duration_classes)
    
    def forward(self, X):
        _, (hidden, cell) = self.lstm(X)
        out = hidden[-1] #just take the final LSTM layer
        out = self.lin(out)
        return out


#First Model, only accounts for pitch
# class MusicLSTM(nn.Module):
#     def __init__(self, input_len, hidden_size, num_classes, num_layers):
#         super(MusicLSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.num_classes = num_classes
#         self.lstm = nn.LSTM(input_len, hidden_size, num_layers, dropout= 0.2, batch_first=True)
#         self.lin = nn.Linear(hidden_size, num_classes)
    
#     def forward(self, X):
#         hidden_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
#         cell_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
#         out, _ = self.lstm(X, (hidden_states, cell_states))

#         out = out[:, -1, :] #extract final hidden state; selects [all batches, last time step, all features]
#         out = self.lin(out)
#         return out
    


class GeminiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GeminiLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.3, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=0.3, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        #x = self.softmax(x)
        return x