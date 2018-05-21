import unicodedata
import io
import string
import re
import random
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from decoder import AttnDecoderRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_CUDA = False

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 10

good_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

teacher_forcing_ratio = 0.5
clip = 5.0


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: 'SOS', 1: 'EOS'}
        self.n_words = 2

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_langs(lang1, lang2, reverse=False):
    print "Reading lines..."

    lines = io.open('../data/%s-%s.txt' % (lang1, lang2), encoding="utf-8").read().strip().split('\n')

    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filter_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and \
           p[1].startswith(good_prefixes)


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang1_name, lang2_name, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1_name, lang2_name, reverse)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))

    print("Indexing words...")
    for pair in pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])

    return input_lang, output_lang, pairs


# Return a list of indexes, one for each word in the sentence
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
#     print('var =', var)
    if USE_CUDA: var = var.cuda()
    return var


def variables_from_pair(pair):
    input_variable = variable_from_sentence(input_lang, pair[0])
    target_variable = variable_from_sentence(output_lang, pair[1])
    return (input_variable, target_variable)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, word_inputs, hidden):
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if USE_CUDA: hidden = hidden.cuda()
        return hidden


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0

    # Get size of input and target sentences
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    # Run words through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_hidden = encoder_hidden
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:

        # Teacher forcing: Use the ground-truth target as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            # print decoder_output
            # print target_variable[di]
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]

    else:
        # Without teacher forcing: use network's own prediction as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_context, decoder_hidden, encoder_outputs
            )
            # print decoder_output.nonzero()
            # print target_variable[di].size()
            loss += criterion(decoder_output, target_variable[di])

            # Get most likely word index (highest value) from output
            topv, topi = decoder_output.data.topk(1)
            print topv
            print topi
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))  # Chosen word is next input
            if USE_CUDA:
                decoder_input = decoder_input.cuda()

            # Stop at end of sentence (not necessary when using known targets)
            if ni == EOS_token:
                break

    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data.item() / target_length


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es -s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


if __name__ == '__main__':

    input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)
    pair = random.choice(pairs)
    print variable_from_sentence(input_lang, pair[0])
    print variable_from_sentence(output_lang, pair[1])

    encoder_test = EncoderRNN(10, 10, 2)
    decoder_test = AttnDecoderRNN('general', 10, 10, 2)

    encoder_hidden = encoder_test.init_hidden()
    word_input = Variable(torch.LongTensor([1, 2, 3]))
    if USE_CUDA:
        encoder_test.cuda()
        word_input = word_input.cuda()
    encoder_outputs, encoder_hidden = encoder_test(word_input, encoder_hidden)

    word_inputs = Variable(torch.LongTensor([1, 2, 3]))
    decoder_attns = torch.zeros(1, 3, 3)
    decoder_hidden = encoder_hidden
    decoder_context = Variable(torch.zeros(1, decoder_test.hidden_size))

    if USE_CUDA:
        decoder_test.cuda()
        word_inputs = word_inputs.cuda()
        decoder_context = decoder_context.cuda()

    for i in range(3):
        decoder_output, decoder_context, decoder_hidden, decoder_attn = decoder_test(word_inputs[i], decoder_context, decoder_hidden, encoder_outputs)
        # print decoder_output
        # print decoder_hidden
        print decoder_attn
        decoder_attns[0, i] = decoder_attn.squeeze(0).cpu().data

    # Training
    # attn_model = 'general'
    # hidden_size = 500
    # n_layers = 2
    # dropout_p = 0.05
    #
    # # Initialize models
    # encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers)
    # decoder = AttnDecoderRNN(attn_model, hidden_size, output_lang.n_words, n_layers, dropout_p=dropout_p)
    #
    # # Move models to GPU
    # if USE_CUDA:
    #     encoder.cuda()
    #     decoder.cuda()
    #
    # # Initialize optimizers and criterion
    # learning_rate = 0.0001
    # encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    # decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    # criterion = nn.NLLLoss()
    #
    # # Configuring training
    # n_epochs = 200
    # plot_every = 10
    # print_every = 100
    #
    # # Keep track of time elapsed and running averages
    # start = time.time()
    # plot_losses = []
    # print_loss_total = 0
    # plot_loss_total = 0
    #
    # # Begin
    # for epoch in range(1, n_epochs + 1):
    #
    #     # Get training data for this cycle
    #     training_pair = variables_from_pair(random.choice(pairs))
    #     input_variable = training_pair[0]
    #     target_variable = training_pair[1]
    #     # print input_variable
    #     # print target_variable
    #
    #     # # Run the train function
    #     loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
    #     # print loss
    #
    #     # Keep track of loss
    #     print_loss_total += loss
    #     plot_loss_total += loss
    #
    #     if epoch == 0:
    #         continue
    #
    #     if epoch % print_every == 0:
    #         print_loss_avg = print_loss_total / print_every
    #         print_loss_total = 0
    #         print_summary = '%s (%d %d%%) %.4f' % (time_since(start, float(epoch) / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
    #         print(print_summary)
    #
    #     if epoch % plot_every == 0:
    #         plot_loss_avg = plot_loss_total / plot_every
    #         plot_losses.append(plot_loss_avg)
    #         plot_loss_total = 0





