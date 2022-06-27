#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# @author: 丁凡彧
import os.path
import numpy as np
import pretty_midi
from pathlib import Path
import sys
from pypianoroll import Multitrack,  Track
import generate
import data.metrics

CONFIG = {}

CONFIG['model'] = {
    'joint_training': None,  # For BinaryMuseGAN only

    # Parameters
    'batch_size': 32,  # Note: tf.layers.conv3d_transpose requires a fixed batch
    # size in TensorFlow < 1.6
    'gan': {
        'type': 'wgan-gp',  # 'gan', 'wgan', 'wgan-gp'
        'clip_value': .01,
        'gp_coefficient': 10.
    },
    'optimizer': {
        # Parameters for the Adam optimizers
        'lr': .002,
        'beta1': .5,
        'beta2': .9,
        'epsilon': 1e-8
    },

    # Data
    'num_bar': 1,
    'num_beat': 4,
    'num_pitch': 128,
    'num_track': 5,
    'num_timestep': 96,
    'beat_resolution': 24,  # 24
    'lowest_pitch': 0,  # MIDI note number of the lowest pitch in data tensors

    # Tracks
    'track_names': (
        'Drums', 'Piano', 'Guitar', 'Bass', 'Strings'
    ),
    'programs': (0, 0, 24, 32, 48),
    'is_drums': (True, False, False, False, False),

    # Network architectures (define them here if not using the presets)
    'net_g': None,
    'net_d': None,
    'net_r': None,  # For BinaryMuseGAN only

    # Playback
    'pause_between_samples': 0,
    'tempo': 90.,

    # Samples
    'num_sample': 8,
    'sample_grid': (2, 4),

    # Metrics
    'metric_map': np.array([
        # indices of tracks for the metrics to compute
        [True] * 8,  # empty bar rate
        [True] * 8,  # number of pitch used
        [False] + [True] * 7,  # qualified note rate
        [False] + [True] * 7,  # polyphonicity
        [False] + [True] * 7,  # in scale rate
        [True] + [False] * 7,  # in drum pattern rate
        [True] * 8,  # total_pitch_class_histogram
        [False] + [True] * 7,  # note_length_hist
    ], dtype=bool),
    'metric_list': np.array([
        # indices of tracks for the metrics to compute
        [True] * 8,  # total_pitch_class_histogram
        [True] * 8,  # note_length_hist
    ], dtype=bool),
    'tonal_distance_pairs': [(1, 0), (1, 2), (1, 3), (1, 4), (3, 2), (3, 4)],  # pairs to compute the tonal distance
    'scale_mask': list(map(bool, [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])),
    'drum_filter': np.tile([1., .1, 0., 0., 0., .1], 16),
    'tonal_matrix_coefficient': (1., 1., .5),

    # Directories
    'checkpoint_dir': None,
    'sample_dir': None,
    'eval_dir': None,
    'log_dir': None,
    'src_dir': None,
}

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'file_path', None,
    'evaluate file path (npy)')

file = FLAGS.file_path
x_pre = np.load(file)

x_pre[(x_pre > 0) & (x_pre < 128)] = 1
x_pre[(x_pre > 128)] = 2

x_pre = generate.combine(x_pre, filt=True)
samples = np.reshape(x_pre, [-1, 96, 60, 5])
pad_width = ((0, 0),
             (0, 0),
             (28,128 - 28 - 60),
             (0, 0))
samples = np.pad(samples, pad_width, 'constant')
print(samples.shape)
score_matrix, score_list_matrix = metrics.eval_samples(samples, CONFIG['model'])
print(score_matrix)
print(score_list_matrix)
