
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


MAX_LENGTH = 10
USE_CUDA = False


class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init()

        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = MAX_LENGTH

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn("concat", hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

        def forward(self, word_input, last_hidden, encoder_outputs):

            # Get the embedding of the current input word (last output word)
            word_embedded = self.embedding(word_input).view(1, 1, -1)
            word_embedded = self.dropout(word_embedded)

            # Calculate attention weights and apply to encoder outputs
            attn_weights = self.attn(last_hidden[-1], encoder_outputs)
            context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

            # Combine embedded input word and attended context, run through RNN
            rnn_input = torch.cat((word_embedded, context), 2)
            output, hidden = self.gru(rnn_input, last_hidden)

            # Final output layer
            output = output.squeeze(0)  # B x N
            output = F.log_softmax(self.out(torch.cat((output, context), 1)), dim=0)

            # Return final output, hidden state, and attention weights (for visualization)
            return output, hidden, attn_weights


class Attn(nn.Module):
    def __init__(self, method, hidden_size, max_length=MAX_LENGTH):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, self.hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(seq_len))  # B x 1 x S
        if USE_CUDA:
            attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies, dim=0).unsqueeze(0).unsqueeze(0)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            # energy = hidden.dot(energy)
            # energy = torch.squeeze(hidden).dot(torch.squeeze(energy))
            energy = torch.dot(hidden.view(-1), energy.view(-1))
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy


class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()

        # Keep parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # Define layers
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size * 2, self.hidden_size, self.n_layers, dropout=self.dropout_p)
        self.out = nn.Linear(self.hidden_size * 2, self.output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(self.attn_model, self.hidden_size)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):

        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1)

        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)), dim=0)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights


