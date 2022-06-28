#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# @author: 丁凡彧


import os.path
import tensorflow.compat.v1 as tf
import pypianoroll
from tensorflow.compat.v1.keras.layers import *
from tensorflow.compat.v1.keras.layers import Layer, InputSpec
from tensorflow.compat.v1.keras import initializers, regularizers, constraints
from tensorflow.compat.v1.keras import backend as K
import numpy as np

import sys
sys.path.append("..")
import data_processing.converter as converter
import data.metrics as metrics
import data.midi_io as midi_io
import data.image_io as image_io
#verify GPU
print("is_gpu: ", tf.test.is_gpu_available())



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

class Shuffle(Layer):
    """Shuffling layers provide two ways to shuffle input dimensions
    One is direct inversion, one is random shuffling, and the default is direct inversion of dimensions
    """

    def __init__(self, idxs=None, mode='reverse', **kwargs):
        super(Shuffle, self).__init__(**kwargs)
        self.idxs = idxs
        self.mode = mode

    def call(self, inputs):
        v_dim = K.int_shape(inputs)[-1]
        if self.idxs == None:
            self.idxs = list(range(v_dim))
            if self.mode == 'reverse':
                self.idxs = self.idxs[::-1]  # flashback
            elif self.mode == 'random':
                np.random.shuffle(self.idxs)  # Then reverse
        inputs = K.transpose(inputs)
        outputs = K.gather(inputs, self.idxs)  # Retrieval, out of order
        outputs = K.transpose(outputs)
        return outputs

    def inverse(self):
        v_dim = len(self.idxs)
        _ = sorted(zip(range(v_dim), self.idxs), key=lambda s: s[1])
        reverse_idxs = [i[0] for i in _]
        return Shuffle(reverse_idxs)

class SplitVector(Layer):
    # Partition the input into two parts, staggered partition

    def __init__(self, **kwargs):
        super(SplitVector, self).__init__(**kwargs)

    def call(self, inputs):
        v_dim = K.int_shape(inputs)[-1]
        return [inputs[..., 0], inputs[..., 1]]

    def compute_output_shape(self, input_shape):
        v_dim = input_shape[-1]
        return [(None, input_shape[-3], input_shape[-2]), (None, input_shape[-3], input_shape[-2])]

    def inverse(self):
        layer = ConcatVector()
        return layer

class ConcatVector(Layer):
    # Remerge the two parts of the partition
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
    """Additive coupling layer
    """

    def __init__(self, isinverse=False, **kwargs):
        self.isinverse = isinverse
        super(AddCouple, self).__init__(**kwargs)

    def call(self, inputs):
        part1, part2, mpart1 = inputs
        if self.isinverse:
            return [part1, part2 + mpart1]  # 逆为加
        else:
            return [part1, part2 - mpart1]  # 正为减

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1]]

    def inverse(self):
        layer = AddCouple(True)
        return layer

class Flatten(Layer):
    """Redefines  default Flatten add the inverse methods
    combination of keras's Reshape and Flatten. And add inverse().
    """

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
    """Scaling layer dimensionality reduction
    """

    def __init__(self, **kwargs):
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1, input_shape[1], input_shape[2]),
                                      initializer='random_normal',
                                      trainable=True)

    def call(self, inputs):
        self.add_loss(-K.sum(self.kernel))  # Logarithmic determinant
        return K.exp(self.kernel) * inputs

    def inverse(self):
        scale = K.exp(-self.kernel)
        return Lambda(lambda x: scale * x)

def where_paino(multitrack):
    def is_piano(program, is_drum): return not is_drum and ((program >= 0 and program <= 7)
                                                            or (program >= 16 and program <= 23))

    for track in multitrack.tracks:
        if is_piano(track.program, track.is_drum):
            return track



def SlidingWindow(file, resol, num_timestep, num_pitch, num_consecutive_bar=8, down_sample=1):
    """Test data sliding window input
    """
    multitrack = pypianoroll.parse(file)
    track = where_paino(multitrack)
    multitrack = converter.first_note_code(multitrack)  # 标记起始音
    downbeat = multitrack.downbeat
    num_bar = len(downbeat) // resol
    hop_iter = 0
    song_ok_segments = []

    for bidx in range(num_bar - num_consecutive_bar//2):
        if hop_iter > 0:
            hop_iter -= 1
            continue
        st = bidx * resol
        ed = st + num_consecutive_bar * resol
        tmp_pianoroll = track[st:ed:down_sample]
        song_ok_segments.append(tmp_pianoroll.pianoroll[np.newaxis, :, :])
        hop_iter = num_consecutive_bar / 2 - 1
        #hop_iter = num_consecutive_bar

    if song_ok_segments[-1].shape[1]<num_timestep:
        song_ok_segments[-1] = np.pad(song_ok_segments[-1], ((0, 0), (0, num_timestep - song_ok_segments[-1].shape[1]), (0, 0)),
                                      'constant')

    pianoroll_compiled = np.concatenate(song_ok_segments, axis=0)[:, :, 28:88]
    # x_pre = np.reshape(pianoroll_compiled, (-1, num_timestep, num_pitch, 1))

    x_pre = np.reshape(pianoroll_compiled, [-1, num_timestep, num_pitch, 1])
    x_pre[(x_pre > 0) & (x_pre < 128)] = 1
    x_pre[(x_pre > 128)] = 1
    x_pre = x_pre.astype('float32')
    return x_pre

def rm_empty(data):
    """
    Removing null data
    """
    res = []
    print(data.shape[0])
    for bar in range(len(data)):
        if (data[bar, :, :, 0].sum() < 1):
            res.append(bar)
    data = np.delete(data, res, axis=0)
    print(data.shape[0])
    return data

def save_samples(config, filename, samples, path, save_midi=False, shape=None, postfix=None):
    """Save samples to an image file (and a MIDI file)."""
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

