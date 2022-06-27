#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: 丁凡彧
import numpy as np
from pypianoroll import Multitrack,  Track
from pathlib import Path
from glob import glob
import os
import argparse
import pretty_midi
import pypianoroll
import errno
import pandas as pd
CONFIG = {
    'multicore': 40,  # the number of cores to use (1 to disable multiprocessing)
    'beat_resolution': 24,  # temporal resolution (in time step per beat)
    'time_signatures': ['4/4']  # '3/4', '2/4'
}

family_name = [
    "drum",
    "piano",
    "guitar",
    "bass",
    "string",
]

family_thres = [
    (1, 1),  # drum
    (1, 1),  # piano
    (1, 1),  # guitar
    (1, 1),  # bass
    (1, 1),  # string
]


def check_which_family(track):
    def is_piano(program, is_drum): return not is_drum and ((program >= 0 and program <= 7)
                                                            or (program >= 16 and program <= 23))

    def is_guitar(program): return program >= 24 and program <= 31

    def is_bass(program): return program >= 32 and program <= 39

    def is_string(program): return program >= 40 and program <= 51

    # drum, bass, guitar, string, piano
    def is_instr_act(program, is_drum): return np.array([is_drum, is_piano(program, is_drum), is_guitar(program),is_bass(program)
                                                         ,is_string(program)])

    instr_act = is_instr_act(track.program, track.is_drum)
    return instr_act

def segment_quality(pianoroll, thres_pitch, thres_beats):
    pitch_sum = sum(np.sum(pianoroll, axis=0) > 0)
    beat_sum = sum(np.sum(pianoroll, axis=1) > 0)
    score = pitch_sum + beat_sum
    return (pitch_sum >= thres_pitch) and (beat_sum >= thres_beats), (pitch_sum, beat_sum)


def segment_quality_paino(pianoroll, thres_pitch, thres_beats):
    #复调
    number_poly_note = np.count_nonzero(np.count_nonzero(pianoroll, 1) > 1)

    pitch_sum = sum(np.sum(pianoroll, axis=0) > 0)
    beat_sum = sum(np.sum(pianoroll, axis=1) > 0)
    score = pitch_sum + beat_sum
    return (number_poly_note < thres_pitch) and (beat_sum >= thres_beats), (pitch_sum, beat_sum)


def get_midi_info(pm):
    """Return useful information from a MIDI object."""
    if pm.time_signature_changes:
        pm.time_signature_changes.sort(key=lambda x: x.time)
        first_beat_time = pm.time_signature_changes[0].time
    else:
        first_beat_time = pm.estimate_beat_start()

    tc_times, tempi = pm.get_tempo_changes()

    if len(pm.time_signature_changes) == 1:
        time_sign = '{}/{}'.format(pm.time_signature_changes[0].numerator,
                                   pm.time_signature_changes[0].denominator)
    else:
        time_sign = None

    midi_info = {
        'first_beat_time': first_beat_time,
        'num_time_signature_change': len(pm.time_signature_changes),
        'constant_time_signature': time_sign,
        'constant_tempo': tempi[0] if len(tc_times) == 1 else None
    }

    return midi_info


def change_prefix(path, src, dst):
    """Return the path with its prefix changed from `src` to `dst`."""
    return os.path.join(dst, os.path.relpath(path, src))

def converter(filepath, src, dst):
    """Convert a MIDI file to a multi-track piano-roll and save the
    resulting multi-track piano-roll to the destination directory. Return a
    tuple of `midi_md5` and useful information extracted from the MIDI file.
    """
    try:
        midi_md5 = os.path.splitext(os.path.basename(filepath))[0]
        multitrack = Multitrack(beat_resolution=CONFIG['beat_resolution'],
                                name=midi_md5)

        pm = pretty_midi.PrettyMIDI(filepath)
        multitrack.parse_pretty_midi(pm)
        #midi_info = get_midi_info(pm)

        result_dir = change_prefix(os.path.dirname(filepath), src, dst)
        make_sure_path_exists(result_dir)

        multitrack.save(os.path.join(result_dir, midi_md5 + '.npz'))

        return midi_md5

    except:
        return None
def make_sure_path_exists(path):
    """Create intermidate directories if the path does not exist."""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
def findall_endswith(postfix, root):
    """Traverse `root` recursively and yield all files ending with `postfix`."""
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(postfix):
                yield os.path.join(dirpath, filename)

def first_note_code(multitrack):
    "Mark the first pronunciation position as -"
    for track in multitrack.tracks:
        if track.is_drum==False:
            pianoroll = track.pianoroll
            # The position of the first value of the tone
            # If the difference from the previous time point is not 0 and is less than 128(Uint8,0-127, greater than 127 is a negative value,[-127->129,-1->255]), it indicates that there is an up tone
            ax_0 = np.unique(np.where((np.r_[np.zeros((1, 128)), np.diff(pianoroll, axis=0)] > 0) &
                                      (np.r_[np.zeros((1, 128)), np.diff(pianoroll, axis=0)] < 128))[0])
            # The note starts at -1
            for i in ax_0:
                for j in np.nonzero(pianoroll[i])[0]:
                    if pianoroll[i - 1][j] == 0:
                        track.pianoroll[i][j] = -1*track.pianoroll[i][j]
            if np.nonzero(pianoroll[0]):
                for j in np.nonzero(pianoroll[0])[0]:
                        track.pianoroll[0][j] = -1*track.pianoroll[0][j]
    return multitrack

#Check if it's 4/4
def midi_filter(midi_info):
    """Return True for qualified MIDI files and False for unwanted ones."""
    if midi_info['constant_time_signature'] not in CONFIG['time_signatures']:
        return False
    return True

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_path', None,
    'input file path ')

flags.DEFINE_string(
    'out_path', None,
    'Save file path')

if __name__ == "__main__":
    '''
    num_consecutive_bar = 4
    resol = 24
    down_sample = 1
    cnt_total_segments = 0
    cnt_augmented = 0
    ok_segment_list = []
    hop_size = (num_consecutive_bar / 4)
    '''
    recursive = True

    dir = FLAGS.input_path
    if not os.path.isdir(dir):
        raise argparse.ArgumentTypeError("dir must be a directory")
    outfile = FLAGS.out_path

    for file in (Path(dir).rglob("*.mid") if recursive else glob(f"{dir}/*.mid")):
        print(f"Processing {file}")

        #1-Select 4/4 beat
        pm = pretty_midi.PrettyMIDI(str(file))
        midi_info = get_midi_info(pm)
        if not midi_filter(midi_info):
            #print('not 4/4 ,skip')
            continue

        multitrack = pypianoroll.parse(str(file))

        #2-The main melody is not a piano jump
        main_melody = np.where(check_which_family(multitrack.tracks[0]))[0]
        # 0-drum, 1-piano, 2-guitar, 3-bass, 4-string
        if not len(main_melody) or main_melody[0]!=1:
            #print('main melody not paino ,skip')
            continue

        #3-Whether the theme (piano) is monophonic
        '''
        number_poly_note = np.count_nonzero(np.count_nonzero(multitrack.tracks[0].pianoroll, 1) > 1)
        if number_poly_note > 0:
            continue
        '''
        '''
        sourcePath = os.path.join(root, f)
        filter_list += 1
        os.system("cp -rf " + sourcePath + " " + targetPath)
        '''
        parent = str(file.parent).replace('lmd_style', 'lmd_filter')
        if not os.path.exists(parent):
            os.makedirs(parent)

        targetPath = str(file).replace('lmd_style', 'lmd_filter')
        os.system("cp -rf " + str(file) + " " + targetPath)


