
import tensorflow as tf
from sklearn.model_selection import train_test_split

tf.enable_eager_execution()


def create_dataset(path):
    sent_pairs = []
    for line in open(path, 'r', encoding='utf-8'):
        sent_pair = line.strip().split('\t')
        sent_pair = ['<start> ' + sent + ' <end>' for sent in sent_pair]
        sent_pairs.append(sent_pair)

    return zip(*sent_pairs)


def tokenize(lang):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(lang)
    tensor = tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post', value=0)
    return tensor, tokenizer


def load_dataset(path):

    inp_lang, tar_lang = create_dataset(path)

    inp_tensor, inp_tokenizer = tokenize(inp_lang)
    tar_tensor, tar_tokenizer = tokenize(tar_lang)

    return inp_tensor, inp_tokenizer, tar_tensor, tar_tokenizer


class Encoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.enc_units = enc_units
        self.batch_size = batch_size
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V1 = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V1(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.dec_units = dec_units
        self.batch_size = batch_size
        self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.keras.layers.Flatten()(output)
        x = self.fc(output)

        return x, state, attention_weights


def loss_function(real, pred):
    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=None)

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = tf.keras.losses.sparse_categorical_crossentropy(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def train_step(inp, tar, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([tar_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

        for t in range(1, tar.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(tar[:, t], predictions)
            # teacher forcing
            dec_input = tf.expand_dims(tar[:, t], axis=1)

    batch_loss = (loss / int(tar.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss


if __name__ == '__main__':
    p = '../data/eng-fra.txt'
    inp_tensor, inp_tokenizer, tar_tensor, tar_tokenizer = load_dataset(p)
    inp_train, inp_val, tar_train, tar_val = train_test_split(inp_tensor, tar_tensor, test_size=0.2)

    vocab_inp_size = len(inp_tokenizer.word_index) + 1
    vocab_tar_size = len(tar_tokenizer.word_index) + 1

    BUFFER_SIZE = len(inp_train)
    BATCH_SIZE = 64
    STEPS_PER_EPOCH = len(inp_train) // BATCH_SIZE
    embedding_dim = 256
    units = 128
    EPOCHS = 1

    dataset = tf.data.Dataset.from_tensor_slices((inp_train, tar_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
    optimizer = tf.keras.optimizers.Adam()

    for epoch in range(EPOCHS):
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (nb, (inp, tar)) in enumerate(dataset.take(STEPS_PER_EPOCH)):
            batch_loss = train_step(inp, tar, enc_hidden)
            total_loss += batch_loss

            if nb % 10 == 0:
                print('Epoch {} Num_batch {} Loss {:.4f}'.format(epoch, nb, batch_loss.numpy()))

        print('Epoch {} Loss{:.4f}'.format(epoch, total_loss / STEPS_PER_EPOCH))








