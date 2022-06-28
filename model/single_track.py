#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# @author: 丁凡彧
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from tensorflow.compat.v1.keras.layers import CuDNNGRU as CUGRU
from tensorflow.compat.v1.keras.layers import *
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.layers import Layer, InputSpec

from tensorflow.compat.v1.keras import initializers, regularizers, constraints
from tensorflow.compat.v1.keras import backend as K
import gc
from tensorflow.compat.v1.keras.callbacks import Callback
#from keras.callbacks import ModelCheckpoint
import imageio
import numpy as np
import data.LPD_data as LPDdata
import data.metrics as metrics
import data.midi_io as midi_io
import data.image_io as image_io
from data.config import CONFIG
import os.path
from pathlib import Path
#from keras.models import load_model
import pypianoroll

import data_processing.converter as converter
#import model.flow_layers as flow
#from keras.backend.tensorflow_backend import set_session
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_path', None,
    'input file path ')
flags.DEFINE_string(
    'test_path', None,
    'input file path ')

# load data
dir = FLAGS.input_path
testset = FLAGS.test_path


x_test = np.load(testset)['arr_0'].astype('float16')
#x_test = x_train[:1000,:,:,:]
print(x_test.shape)
num_bar = 1
num_timestep = x_test.shape[1] * num_bar
num_pitch = x_test.shape[2]
num_track = x_test.shape[3]
num_test = 100
resol = 24

# Convert to 0/1 binary
#x_train[(x_train > 0)] = 1
x_test[(x_test >0)] = 1
x_test = np.reshape(x_test, [-1, num_bar*num_timestep, num_pitch, num_track])
x_sample = np.reshape(x_test, [-1, num_bar, num_timestep, num_pitch, num_track])

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
    # x_pre = np.reshape(pianoroll_compiled, (-1, num_timestep, num_pitch, 1))
    x_pre = np.reshape(pianoroll_compiled, [-1, num_timestep, num_pitch, 1])
    x_pre[(x_pre > 0) & (x_pre < 128)] = 1
    x_pre[(x_pre > 128)] = 1
    x_pre = x_pre.astype('float32')
    return x_pre
# preprocessing
x_pre = SlidingWindow(file, 24, num_timestep, num_pitch, num_consecutive_bar=4 * 8)

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

    def __init__(self, **kwargs):
        super(SplitVector, self).__init__(**kwargs)

    def call(self, inputs):
        v_dim = K.int_shape(inputs)[-1]
        return [inputs[..., 0], inputs[..., 1]]

    def compute_output_shape(self, input_shape):
        v_dim = input_shape[-1]
        return [(None, input_shape[-2]), (None, input_shape[-2])]

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
        return input_shape[0] + (2,)

    def inverse(self):
        layer = SplitVector()
        return layer


class AddCouple(Layer):

    def __init__(self, isinverse=False, **kwargs):
        self.isinverse = isinverse
        super(AddCouple, self).__init__(**kwargs)

    def call(self, inputs):
        part1, part2, mpart1 = inputs
        if self.isinverse:
            return [part1, part2 + mpart1]  
        else:
            return [part1, part2 - mpart1]  

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1]]

    def inverse(self):
        layer = AddCouple(True)
        return layer


class Flatten(Layer):

    def __init__(self, shape=None, **kwargs):
        super(Flatten, self).__init__(**kwargs)
        self.shape = shape

    def call(self, inputs):
        self.in_shape = [i or -1 for i in K.int_shape(inputs)]
        if self.shape is None:
            self.shape = [-1, np.prod(self.in_shape[1:])]
        return K.reshape(inputs, self.shape)

    def compute_output_shape(self, input_shape):
        return tuple([i if i != -1 else None for i in self.shape])

    def inverse(self):
        return Flatten(self.in_shape)

class Scale(Layer):

    def __init__(self, **kwargs):
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1, input_shape[1], input_shape[2]),
                                      initializer='glorot_normal',
                                      trainable=True)

    def call(self, inputs):
        self.add_loss(-K.sum(self.kernel))  
        return K.exp(self.kernel) * inputs

    def inverse(self):
        scale = K.exp(-self.kernel)
        return Lambda(lambda x: scale * x)


def build_basic_model_DNN(num_timestep, num_pitch):
    _in = Input(shape=(num_timestep, num_pitch,))
    _ = _in
    #_ = BatchNormalization(momentum=0.8)(_)
    _ = LayerNormalization()(_)
    _ = Reshape((-1, num_timestep * num_pitch))(_)
    for i in range(3):
        _ = Dense(2000, activation='relu')(_)
        _ = BatchNormalization(momentum=0.8)(_)
        _ = Dropout(0.02)(_)
    _ = Dense(num_timestep * num_pitch)(_)
    _ = Reshape((num_timestep, num_pitch))(_)
    return Model(_in, _)

def build_basic_model_GRU(num_timestep,num_pitch):
    """The basic model is m in the additive coupling layer
    """
    _in = Input(shape=(num_timestep,num_pitch))
    _ = _in
    _ = LayerNormalization()(_)
    _ = Bidirectional(CUGRU(256, return_sequences=True))(_)
    _ = CUGRU(60, return_sequences=True)(_)

    return Model(_in, _)

split = SplitVector()
couple = AddCouple()
concat = ConcatVector()
scale = Scale()
final_reshape = Flatten()

basic_model_11 = build_basic_model_GRU(num_timestep, num_pitch)
basic_model_21 = build_basic_model_GRU(num_timestep, num_pitch)
basic_model_31 = build_basic_model_GRU(num_timestep, num_pitch)
basic_model_41 = build_basic_model_DNN(num_timestep, num_pitch)

x_in = Input(shape=(num_timestep, num_pitch, num_track,))
x = x_in


x1, x2 = split(x)
x2 = scale(x2)

mx1 = basic_model_11(x1)
x1, x2 = couple([x1, x2, mx1])

mx2 = basic_model_21(x1)
x1, x2 = couple([x1, x2, mx2])

mx3 = basic_model_31(x1)
x1, x2 = couple([x1, x2, mx3])

mx4 = basic_model_41(x1)
x1, x2 = couple([x1, x2, mx4])

x2 = final_reshape(x2)

encoder = Model(inputs=x_in, outputs=x2)
encoder.summary()


def my_loss(y_true, y_pred):
    y_r = 0.5 * K.sum(y_pred**2, 1)
    return y_r


encoder.compile(loss=my_loss,
                optimizer='adam')

encoder.load_weights('./single_bass.weights')


# Build the inverse model (generate model) and perform all operations backwards
#x_in = Input(shape=(num_timestep* num_pitch, num_track,))
x_in = Input(shape=(num_timestep, num_pitch, num_track,))
x = x_in

x1, x2 = concat.inverse()(x)
x2 = scale.inverse()(x2)

mx4 = basic_model_41(x1)
x1, x2 = couple.inverse()([x1, x2, mx4])

mx3 = basic_model_31(x1)
x1, x2 = couple.inverse()([x1, x2, mx3])

mx2 = basic_model_21(x1)
x1, x2 = couple.inverse()([x1, x2, mx2])

mx1 = basic_model_11(x1)
x1, x2 = couple.inverse()([x1, x2, mx1])

x = split.inverse()([x1, x2])

decoder = Model(x_in, x)

config = {
    # Metrics
    'metric_map': np.array([
        # indices of tracks for the metrics to compute
        [True] + [True],  # empty bar rate
        [True] + [True],  # number of pitch used
        [False] + [True],  # qualified note rate
        [False] + [True],  # polyphonicity
        [False] + [True],  # in scale rate
        [True] + [False],  # in drum pattern rate
        [False] + [True]  # number of chroma used
    ], dtype=bool),
    'tonal_distance_pairs': [(1, 0)],  # pairs to compute the tonal distance
    'scale_mask': list(map(bool, [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])),
    'drum_filter': np.tile([1., .1, 0., 0., 0., .1], 128),
    'tonal_matrix_coefficient': (1., 1., .5),

    # Data
    'num_bar': 8,
    'num_beat': 4,
    'num_pitch': 60,
    'num_track': 2,
    'num_timestep': num_timestep,
    'beat_resolution': 24,
    'lowest_pitch': 28,  # MIDI note number of the lowest pitch in data tensors

    # Tracks
    'track_names': (
        'Melody', 'bass'
    ),
    'programs': (0, 0),
    'is_drums': (False, False),

    # Playback
    'pause_between_samples': 96,
    'tempo': 90.,

    # Samples
    'num_sample': 8,
    'sample_grid': (2, 4),

}

# Sample to see the generated result
def sample(n=10, std=1, epoch_n='0'):
    """Sample View the generated effect Sample view the generated effect
    """
    nice_sample = []

    for i in range(n):
        melodies_onehot = []

        for j in range(num_bar):
            bar_id = j + i * num_bar
            z_sample = np.array(np.random.randn(1, num_timestep, num_pitch, 1)) 
            z_samples = np.insert(z_sample, 0, values=x_test[bar_id, :,:, 0], axis=-1)
            # z_samples = z_samples.reshape(1, num_timestep * num_pitch, 2)
            x_decoded = decoder.predict(z_samples)
            digit = x_decoded[0]
            melodies = np.array(digit)
            #digit = np.round(digit)
            #digit = np.array(digit).astype('int8')
            melodies[(melodies < 1)] = 0
            melodies = np.clip(digit, 0, 1)
            melodies = melodies.reshape(num_timestep, num_pitch, 2)
            melodies_onehot.append(melodies)

        melodies_onehot = np.array(melodies_onehot).astype(bool)
        nice_sample.append(melodies_onehot)
    nice_sample = np.array(nice_sample)
    nice_sample = nice_sample.reshape(-1, 2, num_timestep // 2, num_pitch, 2)
    return nice_sample


def pre_sample(n, std=1, name='test'):
    """Sample to see the generated result
    """
    nice_sample = []

    for i in range(n):
        bar_id = i
        if bar_id == 0:
            z1 = np.array(np.random.randn(1, num_timestep // 2, num_pitch, 1)) * std
        z2 = np.array(np.random.randn(1, num_timestep // 2, num_pitch, 1)) * std
        z_sample = np.concatenate([z1, z2], axis=1)
        z_samples = np.insert(z_sample, 0, values=x_pre[bar_id,:, :, 0], axis=-1)
        # z_samples = z_samples.reshape(1, num_timestep*num_pitch, 2)
        x_decoded = decoder.predict(z_samples)
        digit = x_decoded[0]
        melodies = np.array(digit)
        #digit = np.round(digit).astype('int8')
        melodies[(melodies < 1)] = 0
        melodies = np.clip(digit, 0, 1)

        melodies = melodies.reshape(-1, num_timestep // 2, num_pitch, 2)
        if bar_id == 0:
            nice_sample.append(melodies[0, :, :, :])
            nice_sample.append(melodies[1, :, :, :])
        else:
            nice_sample.append(melodies[1, :, :, :])
        z1 = z2
    melodies_onehot = np.array(nice_sample).astype(bool)
    nice_sample = np.array(melodies_onehot)
    nice_sample = nice_sample.reshape(-1, 2, num_timestep // 2, num_pitch, 2)
    return nice_sample


def save_samples(config, filename, samples, save_midi=False, shape=None, postfix=None):
    """Save samples to an image file (and a MIDI file)."""
    path = './nice_res/'
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


x_sample = x_sample.reshape(-1, 2, num_timestep // 2, num_pitch, 2)
save_samples(config, 'x_test_sample', x_sample, save_midi=True)


class myCallback(Callback):

    def __init__(self):
        self.lowest = float('inf')
        #self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        ex = 0
        if epoch % 2 == 0:
            nice_sample = sample(num_test, 1, str(ex + epoch))
            metrics.eval_samples(nice_sample, config)
            save_samples(config, 'nice_sample_b_' + str(ex + epoch), nice_sample[:num_test * num_bar], save_midi=True)
            sample_pre = pre_sample(x_pre.shape[0], std=1)
            save_samples(config, 'test_b_' + str(ex + epoch), sample_pre, save_midi=True,shape=(sample_pre.shape[0],1))
            #np.save("./nice_res/bass" + str(ex + epoch) +".npy", sample_pre)
        elif epoch % 5 == 0:
            nice_sample = sample(num_test, 1, str(ex + epoch))
            save_samples(config, 'nice_sample_b_' + str(ex + epoch), nice_sample[:num_test * num_bar], save_midi=True)
            sample_pre = pre_sample(x_pre.shape[0], std=1)
            save_samples(config, 'test_b_' + str(ex + epoch), sample_pre, save_midi=False,shape=(sample_pre.shape[0],1))

        #self.losses.append((epoch, logs['loss']))
        if logs['loss'] < self.lowest:
            self.lowest = logs['loss']
            encoder.save_weights('./single_bass.weights')
            #encoder.save("./model/single_drum.h5")
            # decoder.save("./model/flow_decoder.h5")
        elif logs['loss'] >= self.lowest:
            lr = K.get_value(encoder.optimizer.lr)
            encoder.load_weights('./single_bass.weights')
            K.set_value(encoder.optimizer.lr, lr * 0.1)
        gc.collect()


checkpoint = myCallback()
# encoder = load_model('./model/flow_encoder.h5', custom_objects={"SplitVector": SplitVector(),"AddCouple":AddCouple(),
#                                                    "ConcatVector":ConcatVector(),"Scale":Scale(),"my_loss":my_loss})
# encoder._make_predict_function()

def process_line(path):
    x_train = np.load(path)['arr_0'].astype('float16')
    x_train[(x_train > 0)] = 1
    #print(path.name)
    #print(x_train.shape)
    return x_train

def generate_arrays_from_file(dir,batch_size=128):
    X = []
    while 1:
        for file in Path(dir).rglob('*.npz'):
            # create Numpy arrays of input data
            # and labels, from each line in the file
            x_train = process_line(file)
            for f in x_train:
                f = f.astype('float16')
                X.append(f)
                if len(X) == batch_size:
                    X = np.array(X)
                    yield X, X
                    X = []

encoder.fit_generator(generate_arrays_from_file(dir),steps_per_epoch=100,epochs=101,callbacks=[checkpoint])


nice_sample = sample(100, 1, str(100))
# metrics.eval_samples(nice_sample, CONFIG['model'])
save_samples(config, 'nice_sample_' + str(100), nice_sample[:100], save_midi=True)

from tensorflow.python.util import compat

def export_savedmodel(model,model_path):

    model_version = 0  
    model_signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'input': model.input}, outputs={'output': model.output})
    export_path = os.path.join(compat.as_bytes(model_path), compat.as_bytes(str(model_version))) 
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)  
    builder.add_meta_graph_and_variables(  
        sess=K.get_session(),  
        tags=[tf.saved_model.tag_constants.SERVING], 
        clear_devices=True,  
        signature_def_map={  
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:  
                model_signature 
        })
    builder.save()  
    print("save model pb success ...")

export_savedmodel(decoder,model_path="./model/single/bass/decoder/") 
export_savedmodel(encoder,model_path="./model/single/bass/encoder/")  
