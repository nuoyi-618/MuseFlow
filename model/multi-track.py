import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
import pypianoroll
import numpy as np
import data.LPD_data as LPDdata
import data.metrics as metrics
import data.midi_io as midi_io
import data.image_io as image_io
import data_processing.converter as converter
from data.config import CONFIG
import os.path
from keras.models import load_model

# from flow_layers import *
from sklearn.model_selection import train_test_split

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_path', None,
    'input file path ')

# Load training data
filename = FLAGS.input_path
x_train = LPDdata.load_data('npy', filename)
print(x_train.shape)
num_bar = 1
num_timestep = x_train.shape[1] * num_bar
num_pitch = x_train.shape[2]
num_track = x_train.shape[3]
num_test = 1000

# Extract test samples
x_train = x_train[:num_bar * (x_train.shape[0] // num_bar), :, :, :]
x_train = np.reshape(x_train, [-1, num_timestep, num_pitch, num_track])

x_train[(x_train > 0) & (x_train < 128)] = 1
x_train[(x_train > 128)] = 1


# Remove blank section
def rm_empty(data):
    res = []
    print(data.shape[0])
    for bar in range(len(data)):
        if (data[bar, :, :, 1].sum() < 1):
            res.append(bar)
    data = np.delete(data, res, axis=0)
    print(data.shape[0])
    return data

x_train= rm_empty(x_train)
x_test = x_train[:num_test * num_bar]
x_sample = np.reshape(x_test, [-1, num_bar, num_timestep, num_pitch, num_track])
x_train = x_train[num_test * num_bar:]
# Scrambled data set
rand_idx = np.random.permutation(x_train.shape[0])
x_train = x_train[rand_idx]

# test
file = '../music_data/song_of_joy.mid'
# Test data sliding window input
def SlidingWindow(file, resol, num_timestep, num_pitch, num_consecutive_bar=8, down_sample=1):
    multitrack = pypianoroll.parse(file)
    multitrack = converter.first_note_code(multitrack) 
    downbeat = multitrack.downbeat
    num_bar = len(downbeat) // resol
    hop_iter = 0
    song_ok_segments = []
    track = multitrack.tracks[0]
    for bidx in range(num_bar - num_consecutive_bar // 2):
        if hop_iter > 0:
            hop_iter -= 1
            continue
        st = bidx * resol
        ed = st + num_consecutive_bar * resol
        tmp_pianoroll = track[st:ed:down_sample]
        song_ok_segments.append(tmp_pianoroll.pianoroll[np.newaxis, :, :])
        hop_iter = num_consecutive_bar / 2 - 1
    pianoroll_compiled = np.concatenate(song_ok_segments, axis=0)[:, :, 28:88]
    x_pre = np.reshape(pianoroll_compiled, [-1, num_timestep, num_pitch, 1])
    x_pre[(x_pre > 0) & (x_pre < 128)] = 1
    x_pre[(x_pre > 128)] = 1
    return x_pre


x_pre = SlidingWindow(file, 24, num_timestep, num_pitch)

# from transformer
class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=initializers.Ones(),
                                     trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=initializers.Zeros(),
                                    trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        std = K.std(inputs, axis=-1, keepdims=True)
        return self.gamma * (inputs - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape

class SplitVector(Layer):
   
    def __init__(self, pattern=None, **kwargs):
        super(SplitVector, self).__init__()
        self.pattern = pattern

    def call(self, inputs):
        if self.pattern is None:
            in_dim = K.int_shape(inputs)[-1]
            self.pattern = [_ for _ in range(in_dim)]
        partion = [_ for _ in range(in_dim)]
        return [inputs[..., i] for i in partion]

    def compute_output_shape(self, input_shape):
        return [input_shape[:-1] for d in self.pattern]

    def inverse(self):
        layer = ConcatVector()
        return layer

    def inverse(self):  
        layer = ConcatVector()
        return layer



class ConcatVector(Layer):

    def __init__(self, **kwargs):
        super(ConcatVector, self).__init__(**kwargs)
    def call(self, inputs):
        inputs = [K.expand_dims(i, -1) for i in inputs]
        return K.concatenate(inputs, -1)
    def compute_output_shape(self, input_shape):
        return input_shape[0] +(2,)
    def inverse(self):
        layer = SplitVector()
        return layer


class AddCouple(Layer):


    def __init__(self, isinverse=False, **kwargs):
        self.isinverse = isinverse
        super(AddCouple, self).__init__(**kwargs)

    def call(self, inputs):
        part0, part1, part2, part3, part4, mpart0, mpart2, mpart3, mpart4 = inputs
        if self.isinverse:
            return [part0 + mpart0, part1, part2 + mpart2, part3 + mpart3, part4 + mpart4] 
        else:
            return [part0 - mpart0, part1, part2 - mpart2, part3 - mpart3, part4 - mpart4]  

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_shape[4]]

    def inverse(self):
        layer = AddCouple(True)
        return layer



class Scale(Layer):


    def __init__(self, **kwargs):
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1, input_shape[1], input_shape[2]),
                                      initializer='random_normal',
                                      trainable=True)

    def call(self, inputs):
        self.add_loss(-K.sum(self.kernel))  
        return K.exp(self.kernel) * inputs

    def inverse(self):
        scale = K.exp(-self.kernel)
        return Lambda(lambda x: scale * x)



def build_basic_model_CNN(num_timestep, num_pitch):

    _in = Input(shape=(num_timestep, num_pitch))
    _ = Reshape((num_timestep, num_pitch,1))(_in)
    _ = Conv2D(32, (5, 5), padding='same', activation='relu')(_)
    _ = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(_)
    _ = Flatten()(_)
    _ = Dense(64, activation='relu')(_)
    _ = Reshape((num_timestep, num_pitch))(_)

    return Model(_in, _)



def build_basic_model_DNN(num_timestep, num_pitch):

    _in = Input(shape=(num_timestep, num_pitch,))
    _ = _in
    # _ = BatchNormalization(momentum=0.8)(_)
    _ = Reshape((-1, num_timestep * num_pitch))(_)
    for i in range(3):
        _ = Dense(1000, activation='relu')(_)
        _ = BatchNormalization(momentum=0.8)(_)
        _ = Dropout(0.02)(_)
    _ = Dense(num_timestep * num_pitch)(_)
    _ = Reshape((num_timestep, num_pitch))(_)
    return Model(_in, _)


def build_basic_model_GRU(num_timestep,num_pitch):

    _in = Input(shape=(num_timestep,num_pitch))
    _ = _in
    #_ = BatchNormalization(momentum=0.8)(_)
    _ = LayerNormalization()(_)
    #_ = CuDNNGRU(1000,  return_sequences=True)(_)
    _ = Bidirectional(CuDNNGRU(128, return_sequences=True))(_)
    _ = CuDNNGRU(60, return_sequences=True)(_)

    return Model(_in, _)


def save_samples(config, filename, samples, save_midi=False, shape=None, postfix=None):
    """Save samples to an image file (and a MIDI file)."""
    path = './result/'
    if shape is None:
        shape = config["sample_grid"]
    if len(samples) > config['num_sample']:
        samples = samples[:config['num_sample']]
    if postfix is None:
        imagepath = os.path.join(path, '{}.png'.format(filename))
    else:
        imagepath = os.path.join(path, '{}_{}.png'.format(filename, postfix))
    image_io.save_image(imagepath, samples, shape)
    if save_midi:
        binarized = (samples > 0)
        midipath = os.path.join(path, '{}.mid'.format(filename))
        midi_io.save_midi(midipath, binarized, config)



def sample(n=10, std=1, epoch_n='0'):

    nice_sample = []
    for i in range(n):
        bar_id = i
        if bar_id == 0:
            z1 = np.random.randn(1, num_timestep // 2, num_pitch, (num_track - 1))
        z2 = np.random.randn(1, num_timestep // 2, num_pitch, (num_track - 1))
        # here
        z_sample = np.concatenate([z1, z2], axis=1) * std
        z_samples = np.insert(z_sample, 1, values=x_test[bar_id, :, :, 1], axis=-1)
        x_decoded = decoder.predict(z_samples)
        digit = x_decoded[0]
        # Slide window generation, keep the next section
        melodies = digit.reshape(-1, num_timestep // 2, num_pitch, num_track)
        if bar_id == 0:
            nice_sample.append(melodies[0, :, :, :])
            nice_sample.append(melodies[1, :, :, :])
        else:
            nice_sample.append(melodies[1, :, :, :])
        z1 = z2
    melodies_onehot = np.round(nice_sample).astype('int8')
    melodies_onehot = np.clip(melodies_onehot, 0, 1)
    melodies_onehot = np.array(melodies_onehot).astype(bool)
    nice_sample_bool = np.array(melodies_onehot)[:melodies_onehot.shape[0] // 2 * 2, :, :, :]
    nice_sample_bool = nice_sample_bool.reshape(-1, 2, num_timestep // 2, num_pitch, num_track)
    return nice_sample_bool


def SlidingWindow_pre_sample(n, std=1, name='test'):
    """Sample to see the generated result
    """
    nice_sample = []

    for i in range(n):
        bar_id = i
        if bar_id == 0:
            z1 = np.random.randn(1, num_timestep // 2, num_pitch, (num_track - 1))
        z2 = np.random.randn(1, num_timestep // 2, num_pitch, (num_track - 1))
        # here
        z_sample = np.concatenate([z1, z2], axis=1) * std
        z_samples = np.insert(z_sample, 1, values=x_pre[bar_id, :, :, 0], axis=-1)
        x_decoded = decoder.predict(z_samples)
        digit = x_decoded[0]
        #melodies = np.array(digit).astype('int8')
        melodies = np.round(digit).astype('int8')
        melodies = np.clip(melodies, 0, 1)

        melodies = melodies.reshape(-1, num_timestep // 2, num_pitch, num_track)
        if bar_id == 0:
            nice_sample.append(melodies[0, :, :, :])
            nice_sample.append(melodies[1, :, :, :])
        else:
            nice_sample.append(melodies[1, :, :, :])
        z1 = z2
    melodies_onehot = np.array(nice_sample).astype(bool)
    nice_sample = np.array(melodies_onehot)
    nice_sample = nice_sample.reshape(-1, 2, num_timestep // 2, num_pitch, num_track)
    return nice_sample

save_samples(CONFIG['model'], 'x_test_sample', x_sample, save_midi=True)

split = SplitVector()  # Designated guide track
couple = AddCouple()
concat = ConcatVector()
scale1 = Scale()
scale2 = Scale()
basic_model_11 = build_basic_model_DNN(num_timestep, num_pitch)
basic_model_12 = build_basic_model_GRU(num_timestep, num_pitch)
basic_model_13 = build_basic_model_GRU(num_timestep, num_pitch)
basic_model_14 = build_basic_model_GRU(num_timestep, num_pitch)

basic_model_21 = build_basic_model_DNN(num_timestep, num_pitch)
basic_model_22 = build_basic_model_GRU(num_timestep, num_pitch)
basic_model_23 = build_basic_model_GRU(num_timestep, num_pitch)
basic_model_24 = build_basic_model_GRU(num_timestep, num_pitch)

basic_model_31 = build_basic_model_DNN(num_timestep, num_pitch)
basic_model_32 = build_basic_model_GRU(num_timestep, num_pitch)
basic_model_33 = build_basic_model_GRU(num_timestep, num_pitch)
basic_model_34 = build_basic_model_GRU(num_timestep, num_pitch)

basic_model_41 = build_basic_model_DNN(num_timestep, num_pitch)
basic_model_42 = build_basic_model_GRU(num_timestep, num_pitch)
basic_model_43 = build_basic_model_GRU(num_timestep, num_pitch)
basic_model_44 = build_basic_model_GRU(num_timestep, num_pitch)

x_in = Input(shape=(num_timestep, num_pitch, num_track,))
x = x_in


x0, x1, x2, x3, x4 = split(x)
mx0 = basic_model_11(x1)
mx2 = basic_model_12(x1)
mx3 = basic_model_13(x1)
mx4 = basic_model_14(x1)
x0, x1, x2, x3, x4 = couple([x0, x1, x2, x3, x4, mx0, mx2, mx3, mx4])

mx0 = basic_model_21(x1)
mx2 = basic_model_22(x1)
mx3 = basic_model_23(x1)
mx4 = basic_model_24(x1)
x0, x1, x2, x3, x4 = couple([x0, x1, x2, x3, x4, mx0, mx2, mx3, mx4])

mx0 = basic_model_31(x1)
mx2 = basic_model_32(x1)
mx3 = basic_model_33(x1)
mx4 = basic_model_34(x1)
x0, x1, x2, x3, x4 = couple([x0, x1, x2, x3, x4, mx0, mx2, mx3, mx4])

mx0 = basic_model_41(x1)
mx2 = basic_model_42(x1)
mx3 = basic_model_43(x1)
mx4 = basic_model_44(x1)
x0, x1, x2, x3, x4 = couple([x0, x1, x2, x3, x4, mx0, mx2, mx3, mx4])

x0 = scale1(x0)
x2 = scale2(x2)
x3 = scale2(x3)
x4 = scale2(x4)
x = concat([x0, x1, x2, x3, x4])

# x = scale(x)


encoder = Model(x_in, x)
encoder.summary()


def mean_pred(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, num_timestep * num_pitch, 5))
    loss = K.sum(0.5 * y_pred ** 2, 1)
    return loss


encoder.compile(loss=mean_pred, optimizer='adam')

#encoder.load_weights('./best_flow_nice_dnn.weights')

# Build the inverse model (generate model) and perform all operations backwards

x = x_in
# x = scale.inverse()(x)

x0, x1, x2, x3, x4 = concat.inverse()(x)
x0 = scale1.inverse()(x0)
x2 = scale2.inverse()(x2)
x3 = scale2.inverse()(x3)
x4 = scale2.inverse()(x4)

mx0 = basic_model_41(x1)
mx2 = basic_model_42(x1)
mx3 = basic_model_43(x1)
mx4 = basic_model_44(x1)
x0, x1, x2, x3, x4 = couple.inverse()([x0, x1, x2, x3, x4, mx0, mx2, mx3, mx4])

mx0 = basic_model_31(x1)
mx2 = basic_model_32(x1)
mx3 = basic_model_33(x1)
mx4 = basic_model_34(x1)
x0, x1, x2, x3, x4 = couple.inverse()([x0, x1, x2, x3, x4, mx0, mx2, mx3, mx4])

mx0 = basic_model_21(x1)
mx2 = basic_model_22(x1)
mx3 = basic_model_23(x1)
mx4 = basic_model_24(x1)
x0, x1, x2, x3, x4 = couple.inverse()([x0, x1, x2, x3, x4, mx0, mx2, mx3, mx4])

mx0 = basic_model_11(x1)
mx2 = basic_model_12(x1)
mx3 = basic_model_13(x1)
mx4 = basic_model_14(x1)
x0, x1, x2, x3, x4 = couple.inverse()([x0, x1, x2, x3, x4, mx0, mx2, mx3, mx4])
x = split.inverse()([x0, x1, x2, x3, x4])

decoder = Model(x_in, x)


class myCallback(Callback):

    def __init__(self):
        self.lowest = float('inf')
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):

        ex = 0
        if epoch % 50 == 0:
            nice_sample = sample(num_test, 1, str(ex + epoch))
            metrics.eval_samples(nice_sample, CONFIG['model'])
            save_samples(CONFIG['model'], 'nice_sample_' + str(ex + epoch), nice_sample[:100], save_midi=True)
            sample_pre = SlidingWindow_pre_sample(x_pre.shape[0], std=1, name='test')
            save_samples(CONFIG['model'], 'test' + str(ex + epoch), sample_pre, save_midi=True)
        elif epoch % 10 == 0:
            nice_sample = sample(num_test, 1, str(ex + epoch))
            save_samples(CONFIG['model'], 'nice_sample_' + str(ex + epoch), nice_sample[:100], save_midi=True)
            sample_pre = SlidingWindow_pre_sample(int(x_pre.shape[0] / CONFIG['model']['num_bar']), std=1, name='test')
            save_samples(CONFIG['model'], 'test' + str(ex + epoch), sample_pre, save_midi=True)

        self.losses.append((epoch, logs['loss']))

        if logs['loss'] < self.lowest:
            self.lowest = logs['loss']
            encoder.save_weights('./best_museflow.weights')
            #encoder.save("./model/flow_encoder.h5")
            #decoder.save("./model/flow_decoder.h5")
        elif logs['loss'] >= self.lowest:
            lr = K.get_value(encoder.optimizer.lr)
            encoder.load_weights('./best_museflow.weights')
            K.set_value(encoder.optimizer.lr, lr * 0.1)



checkpoint = myCallback()

encoder.fit(x_train, x_train, batch_size=128, epochs=501, callbacks=[checkpoint])

encoder.save("./model/museflow_encoder.h5")
decoder.save("./model/museflow_decoder.h5")

