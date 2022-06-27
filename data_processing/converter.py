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
    'multicore': 40, # the number of cores to use (1 to disable multiprocessing)
    'beat_resolution': 24, # temporal resolution (in time step per beat)
    'time_signatures': ['4/4'] # '3/4', '2/4'
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
    (1, 96),  # piano
    (1, 96),  # guitar
    (1, 172),  # bass
    (1, 172),  # string
]


def parse():
    parser = argparse.ArgumentParser(
        description="Convert midi files into training set")
    parser.add_argument("dir",
                        help="directory containing .mid files")
    parser.add_argument("-r", "--recursive", dest="recursive", action="store_true",
                        help="search directory recursively")
    parser.add_argument("--outfile", dest="out",
                        default="data_processing/train.npy",
                        help="output file")

    return parser.parse_args()

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


def compress_pianoroll(multitrack):
    "Compress and mark tracks, 16 ticks per bar, first note marked -"
    "np.repeat(small_pianoroll,6,axis=0)"
    for track in multitrack.tracks:
        pianoroll = track.pianoroll
        if track.is_drum == False:
            for i in range(pianoroll.shape[1]):
                pitch = track.pianoroll[:, i]
                # The position in which the note changes
                ax_0 = np.where(np.diff(np.r_[np.zeros(1), pitch], axis=0))[0]
                for k in range(len(ax_0) // 2):
                    if ax_0[k * 2 + 1] - ax_0[k * 2] < 3:
                        track.pianoroll[ax_0[k * 2]:ax_0[k * 2 + 1], i] = 0
                    elif k % 2 == 0:
                        track.pianoroll[ax_0[k * 2]:ax_0[k * 2 + 1], i] = -1 * track.pianoroll[ax_0[k * 2]][i]
        tmp_pianoroll = track.pianoroll[::6]
        track.pianoroll = tmp_pianoroll
        for i in range(tmp_pianoroll.shape[1]):
            tmp_pitch = track.pianoroll[:, i]
            ax_1 = np.where(np.diff(np.r_[np.zeros(1), tmp_pitch, np.zeros(1)], axis=0))[0]

            for k in range(len(ax_1)-1):
                #If the current tone is less than 128, the first value is inverted. If the current tone is greater than 128, all other pitches in the range of duration other than the starting point are inverted
                if tmp_pitch[ax_1[k]]<128 and tmp_pitch[ax_1[k]] > 0:
                    track.pianoroll[ax_1[k]][i] = -1 * track.pianoroll[ax_1[k]][i]
                if tmp_pitch[ax_1[k]]>128 and ax_1[k+1]-ax_1[k]>1:
                    track.pianoroll[ax_1[k] + 1: ax_1[k + 1], i] = -1 * track.pianoroll[ax_1[k]][i]
    return multitrack


#Check if it's 4/4
def midi_filter(midi_info):
    """Return True for qualified MIDI files and False for unwanted ones."""
    if midi_info['constant_time_signature'] not in CONFIG['time_signatures']:
        return False
    return True

if __name__ == "__main__":

    num_consecutive_bar = 4*2
    resol = 24
    down_sample = 1
    cnt_total_segments = 0
    cnt_augmented = 0
    ok_segment_list = []
    recursive = True
    print("8beat,have_sw,q=1,single")
    ags=parse()
    dir = ags.dir
    if not os.path.isdir(dir):
        raise argparse.ArgumentTypeError("dir must be a directory")
    outfile = ags.outfile
    if not outfile.endswith(".npy") and not outfile.endswith(".npz"):
        outfile += ".npy"
    try:
        outdir = os.path.split(outfile)[0]
        if outdir != "" and not os.path.exists(outdir):
            os.makedirs(outdir)
        open(outfile, "w").close()
    except Exception as e:
        raise argparse.ArgumentTypeError("outfile is not valid")
    for file in (Path(dir).rglob("*.mid") if recursive else glob(f"{dir}/*.mid")):
        print(f"Processing {file}")
        #midi_md5 = os.path.splitext(os.path.basename(file))[0]

        #1-Select 4/4 beat
        pm = pretty_midi.PrettyMIDI(str(file))
        midi_info = get_midi_info(pm)
        if not midi_filter(midi_info):
            #print('not 4/4 ,skip')
            continue

        #multitrack = pypianoroll.parse(str(file))
        multitrack = Multitrack(str(file), beat_resolution=24)

        #2-The main melody is not a piano jump
        main_melody = np.where(check_which_family(multitrack.tracks[0]))[0]
        # 0-drum, 1-piano, 2-guitar, 3-bass, 4-string
        if not len(main_melody) or main_melody[0]!=1:
            #print('main melody not paino ,skip')
            continue

        downbeat = multitrack.downbeat

        num_bar = len(downbeat) // resol 
        hop_iter = 0

        song_ok_segments = []
        for bidx in range(num_bar-num_consecutive_bar):
            if hop_iter > 0:
                hop_iter -= 1
                continue

            st = bidx * resol
            ed = st + num_consecutive_bar * resol

            best_instr = [Track(pianoroll=np.zeros(
                (num_consecutive_bar*resol, 128),dtype=np.uint8))] * 5
            best_score = [-1] * 5



            for tidx, track in enumerate(multitrack.tracks):
                # 0-drum, 1-piano, 2-guitar,3-bass, 4-string
                tmp_map = check_which_family(track)
                in_family = np.where(tmp_map)[0]

                if not len(in_family):
                    continue
                family = in_family[0]

                tmp_pianoroll = track[st:ed:down_sample]
                #3-The piano is frustration segment_quality_paino
                if family==1:
                    is_ok, score = segment_quality(
                        tmp_pianoroll.pianoroll, family_thres[family][0], family_thres[family][1])
                    if not is_ok:
                        break
                else:
                    is_ok, score = segment_quality(
                    tmp_pianoroll.pianoroll, family_thres[family][0], family_thres[family][1])

                if is_ok and sum(score) > best_score[family]:
                    track.name = family_name[family]
                    best_instr[family] = track[st:ed:down_sample]
                    best_score[family] = sum(score)

            hop_iter = num_consecutive_bar//2-1
            #4-At least 3 tracks sound
            if sum(np.array(best_score) >0) > 2:
                sign_multitrack = Multitrack(tracks=best_instr, beat_resolution=24)
                # Mark the tone position
                #sign_multitrack = compress_pianoroll(sign_multitrack)
                sign_multitrack = first_note_code(sign_multitrack)
                song_ok_segments.append(sign_multitrack)

        cnt_ok_segment = len(song_ok_segments)
        '''
        if cnt_ok_segment > 6:
            seed = (6, cnt_ok_segment//2)
            if cnt_ok_segment > 11:
                seed = (11, cnt_ok_segment//3)
            if cnt_ok_segment > 15:
                seed = (15, cnt_ok_segment//4)

            rand_idx = np.random.permutation(cnt_ok_segment)[:max(seed)]
            song_ok_segments = [song_ok_segments[ridx] for ridx in rand_idx]
            ok_segment_list.extend(song_ok_segments)
            cnt_ok_segment = len(rand_idx)
        else:
            ok_segment_list.extend(song_ok_segments)
        '''
        ok_segment_list.extend(song_ok_segments)

        cnt_total_segments += len(song_ok_segments)
        print(f"current: {cnt_ok_segment} | cumulative: {cnt_total_segments}")
    print("-"*30)
    print(cnt_total_segments)
    num_item = len(ok_segment_list)
    compiled_list = []
    for lidx in range(num_item):
        multi_track = ok_segment_list[lidx]
        pianorolls = []

        for tracks in multi_track.tracks:
            pianorolls.append(tracks.pianoroll[:, :, np.newaxis])
        print(lidx)
        #pianoroll_compiled = np.reshape(np.concatenate(pianorolls, axis=2)[:, 28:88, :], (num_consecutive_bar, resol, 60, 5))
        pianoroll_compiled = np.concatenate(pianorolls, axis=2)[:, 28:88, :]
        pianoroll_compiled = pianoroll_compiled[np.newaxis, :]
        #pianoroll_compiled = pianoroll_compiled[np.newaxis, :] > 0
        #compiled_list.append(pianoroll_compiled.astype(bool))
        compiled_list.append(pianoroll_compiled)
    result = np.concatenate(compiled_list, axis=0)
    result = result.astype(np.uint8)
    print(f"output shape: {result.shape}")
    if outfile.endswith(".npz"):
        np.savez_compressed(
            outfile, nonzero=np.array(result.nonzero()),
            shape=result.shape)
    else:
        np.save(outfile, result)
    print(f"saved to {outfile}")

