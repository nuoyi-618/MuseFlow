#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# @author: 丁凡彧

import data.midi_io as midi_io
import data.image_io as image_io
from data.config import CONFIG
import os.path
import data.LPD_data as LPDdata
import numpy as np
import pypianoroll
import data_processing.converter as converter
# Load training data
"""npy 2 painoroll"""
#filename = "../../Music/data/lpd_cleaned_s.npy"
#filename = "../music_data/lpd_cleaned_s.npy"
filename = "D:\\0.Project\music\music_data\LMD\lmd_coverter\Blues_Contemporary.npy"
x_train = LPDdata.load_data('npy', filename)


def save_samples(config, filename, samples, save_midi=False, shape=None, postfix=None):
    """Save samples to an image file (and a MIDI file)."""
    if shape is None:
        shape = config["sample_grid"]
    if len(samples) > config['num_sample']:
        samples = samples[:config['num_sample']]
    if postfix is None:
        imagepath = os.path.join('./', '{}.png'.format(filename))
    else:
        imagepath = os.path.join('./', '{}_{}.png'.format(filename, postfix))
    image_io.save_image(imagepath, samples, shape)
    if save_midi:
        binarized = (samples > 0)
        midipath = os.path.join('./', '{}.mid'.format(filename))
        midi_io.save_midi(midipath, binarized, config)


x_sample = x_train[np.random.choice(len(x_train), CONFIG['model']['batch_size'], False)]
save_samples(CONFIG['model'], 'x_sample', x_train, save_midi=True)
"""painoroll 2 npy"""
# 欢乐颂
file = 'D:/0.Project\music\MAGtest\data_processing\dataset\song_of_joy.mid'

# 测试数据滑窗输入
def SlidingWindow(file, resol, num_timestep, num_pitch, num_consecutive_bar=8, down_sample=1):
    multitrack = pypianoroll.parse(file)
    multitrack = converter.first_note_code(multitrack)  # 标记起始音
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


x_pre = SlidingWindow(file, 24, 96, 60)
np.save('D:/0.Project\music\MAGtest\data_processing\dataset\song_of_joy_1.npy', x_pre)