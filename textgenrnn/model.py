from keras.optimizers import RMSprop
from keras.layers import Input, Embedding, Dense, LSTM, CuDNNLSTM, concatenate
from keras.models import Model
from .AttentionWeightedAverage import AttentionWeightedAverage


def textgenrnn_model(weights_path, num_classes, cfg,
                     optimizer=RMSprop(lr=4e-3, rho=0.99)):
    '''
    Builds the model architecture for textgenrnn and
    loads the specified weights for the model.
    '''

    input = Input(shape=(cfg['max_length'],), name='input')
    embedded = Embedding(num_classes, cfg['dim_embeddings'],
                         input_length=cfg['max_length'],
                         name='embedding')(input)

    rnn_layer_list = []
    for i in range(cfg['rnn_layers']):
        prev_layer = embedded if i is 0 else rnn_layer_list[-1]
        rnn_layer_list.append(new_rnn(cfg, i+1)(prev_layer))

    seq_concat = concatenate([embedded] + rnn_layer_list, name='rnn_concat')
    attention = AttentionWeightedAverage(name='attention')(seq_concat)

    output = Dense(num_classes, name='output', activation='softmax')(attention)

    model = Model(inputs=[input], outputs=[output])
    model.load_weights(weights_path, by_name=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

'''
Create a new LSTM layer per parameters. Unfortunately,
each combination of parameters must be hardcoded.

The normal LSTMs use sigmoid recurrent activations
for parity with CuDNNLSTM:
https://github.com/keras-team/keras/issues/8860
'''


def new_rnn(cfg, layer_num):

    if cfg['rnn_bidirectional']:
        return Bidirectional(LSTM(cfg['rnn_size'],
                                  return_sequences=True,
                                  recurrent_activation='sigmoid',
                                  name='rnn_{}'.format(layer_num)))

    return LSTM(cfg['rnn_size'],
                return_sequences=True,
                recurrent_activation='sigmoid',
                name='rnn_{}'.format(layer_num))