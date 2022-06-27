"""Class and utilities for metrics
"""
import os
import warnings
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import midi
import pretty_midi
import sys

sys.path.append("..")
import data.midi_io as midi_io
from config import CONFIG
#warnings.simplefilter("error")

def extract_feature(_file):
    """
    This function extracts two midi feature:

    Returns:
        dict(pretty_midi: pretty_midi object,
             midi_pattern: midi pattern contains a list of tracks)
    """
    feature = {'pretty_midi': pretty_midi.PrettyMIDI(_file),
               'midi_pattern': midi.read_midifile(_file)}
    return feature


def total_used_note(feature, track_num=1):
    """
    total_used_note (Note count): The number of used notes.
    As opposed to the pitch count, the note count does not contain pitch information but is a rhythm-related feature.

    Args:
    'track_num' : specify the track number in the midi pattern, default is 1 (the second track).

    Returns:
    'used_notes': a scalar for each sample.
    """
    pattern = feature['midi_pattern']
    used_notes = 0
    for i in range(0, len(pattern[track_num])):
        if type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] != 0:
            used_notes += 1
    return used_notes


def avg_pitch_shift(feature, track_num=1):
    """
    avg_pitch_shift (Average pitch interval):
    Average value of the interval between two consecutive pitches in semitones.

    Args:
    'track_num' : specify the track number in the midi pattern, default is 1 (the second track).

    Returns:
    'pitch_shift': a scalar for each sample.
    """
    pattern = feature['midi_pattern']
    pattern.make_ticks_abs()
    resolution = pattern.resolution
    _used_note = total_used_note(feature, track_num=track_num)
    if _used_note < 2:
        return 0
    d_note = np.zeros((max(_used_note - 1, 0)))
    current_note = 0
    counter = 0
    for i in range(0, len(pattern[track_num])):
        if type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] != 0:
            if counter != 0:
                d_note[counter - 1] = current_note - pattern[track_num][i].data[0]
                current_note = pattern[track_num][i].data[0]
                counter += 1
            else:
                current_note = pattern[track_num][i].data[0]
                counter += 1
    pitch_shift = np.mean(abs(d_note))
    return pitch_shift


def avg_IOI(pianoroll):
    """
    avg_IOI (Average inter-onset-interval):
    To calculate the inter-onset-interval in the symbolic music domain, we find the time between two consecutive notes.

    Returns:
    'avg_ioi': a scalar for each sample.
    """
    padded = np.pad(pianoroll.astype(int), ((1, 1), (0, 0)), 'constant')
    diff = np.diff(padded, axis=0)
    flattened = diff.T.reshape(-1, )
    onsets = (flattened > 0).nonzero()[0]
    ioi = np.diff(onsets)
    avg_ioi = np.mean(ioi)
    return avg_ioi


def total_pitch_class_histogram(feature):
    """
    total_pitch_class_histogram (Pitch class histogram):
    The pitch class histogram is an octave-independent representation of the pitch content with a dimensionality of 12 for a chromatic scale.
    In our case, it represents to the octave-independent chromatic quantization of the frequency continuum.

    Returns:
    'histogram': histrogram of 12 pitch, with weighted duration shape 12
    """
    piano_roll = feature
    histogram = np.zeros(12)
    for i in range(0, 128):
        pitch_class = i % 12
        histogram[pitch_class] += np.sum(piano_roll, axis=0)[i]
    histogram = histogram / sum(histogram)
    return histogram


def note_length_hist(feature, track_num=1, normalize=True, pause_event=False):
    """
    note_length_hist (Note length histogram):
    To extract the note length histogram, we first define a set of allowable beat length classes:
    [full, half, quarter, 8th, 16th, dot half, dot quarter, dot 8th, dot 16th, half note triplet, quarter note triplet, 8th note triplet].
    The pause_event option, when activated, will double the vector size to represent the same lengths for rests.
    The classification of each event is performed by dividing the basic unit into the length of (barlength)/96, and each note length is quantized to the closest length category.

    Args:
    'track_num' : specify the track number in the midi pattern, default is 1 (the second track).
    'normalize' : If true, normalize by vector sum.
    'pause_event' : when activated, will double the vector size to represent the same lengths for rests.

    Returns:
    'note_length_hist': The output vector has a length of either 12 (or 24 when pause_event is True).
    """

    pattern = feature['midi_pattern']
    if pause_event is False:
        note_length_hist = np.zeros((12))
        pattern.make_ticks_abs()
        resolution = pattern.resolution
        # basic unit: bar_length/96
        for i in range(0, len(pattern[track_num])):
            if type(pattern[track_num][i]) == midi.events.TimeSignatureEvent:
                time_sig = pattern[track_num][i].data
                bar_length = time_sig[track_num] * resolution * 4 / 2 ** (time_sig[1])
            elif type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] != 0:
                if 'time_sig' not in locals():  # set default bar length as 4 beat
                    bar_length = 4 * resolution
                    time_sig = [4, 2, 24, 8]
                unit = bar_length / 96.
                hist_list = [unit * 96, unit * 48, unit * 24, unit * 12, unit * 6, unit * 72, unit * 36, unit * 18,
                             unit * 9, unit * 32, unit * 16, unit * 8]
                current_tick = pattern[track_num][i].tick
                current_note = pattern[track_num][i].data[0]
                # find next note off
                for j in range(i, len(pattern[track_num])):
                    if type(pattern[track_num][j]) == midi.events.NoteOffEvent or (
                            type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[
                        1] == 0):
                        if pattern[track_num][j].data[0] == current_note:
                            note_length = pattern[track_num][j].tick - current_tick
                            distance = np.abs(np.array(hist_list) - note_length)
                            idx = distance.argmin()
                            note_length_hist[idx] += 1
                            break
    else:
        note_length_hist = np.zeros((24))
        pattern.make_ticks_abs()
        resolution = pattern.resolution
        # basic unit: bar_length/96
        for i in range(0, len(pattern[track_num])):
            if type(pattern[track_num][i]) == midi.events.TimeSignatureEvent:
                time_sig = pattern[track_num][i].data
                bar_length = time_sig[track_num] * resolution * 4 / 2 ** (time_sig[1])
            elif type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] != 0:
                check_previous_off = True
                if 'time_sig' not in locals():  # set default bar length as 4 beat
                    bar_length = 4 * resolution
                    time_sig = [4, 2, 24, 8]
                unit = bar_length / 96.
                tol = 3. * unit
                hist_list = [unit * 96, unit * 48, unit * 24, unit * 12, unit * 6, unit * 72, unit * 36, unit * 18,
                             unit * 9, unit * 32, unit * 16, unit * 8]
                current_tick = pattern[track_num][i].tick
                current_note = pattern[track_num][i].data[0]
                # find next note off
                for j in range(i, len(pattern[track_num])):
                    # find next note off
                    if type(pattern[track_num][j]) == midi.events.NoteOffEvent or (
                            type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[
                        1] == 0):
                        if pattern[track_num][j].data[0] == current_note:

                            note_length = pattern[track_num][j].tick - current_tick
                            distance = np.abs(np.array(hist_list) - note_length)
                            idx = distance.argmin()
                            note_length_hist[idx] += 1
                            break
                        else:
                            if pattern[track_num][j].tick == current_tick:
                                check_previous_off = False

                # find previous note off/on
                if check_previous_off is True:
                    for j in range(i - 1, 0, -1):
                        if type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[
                            1] != 0:
                            break

                        elif type(pattern[track_num][j]) == midi.events.NoteOffEvent or (
                                type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[
                            1] == 0):

                            note_length = current_tick - pattern[track_num][j].tick
                            distance = np.abs(np.array(hist_list) - note_length)
                            idx = distance.argmin()
                            if distance[idx] < tol:
                                note_length_hist[idx + 12] += 1
                            break

    if normalize is False:
        return note_length_hist

    elif normalize is True:

        return note_length_hist / np.sum(note_length_hist)


def get_tonal_matrix(r1=1.0, r2=1.0, r3=0.5):
    """Compute and return a tonal matrix for computing the tonal distance [1].
    Default argument values are set as suggested by the paper.

    [1] Christopher Harte, Mark Sandler, and Martin Gasser. Detecting harmonic
    change in musical audio. In Proc. ACM MM Workshop on Audio and Music
    Computing Multimedia, 2006.
    """
    tonal_matrix = np.empty((6, 12))
    tonal_matrix[0] = r1 * np.sin(np.arange(12) * (7. / 6.) * np.pi)
    tonal_matrix[1] = r1 * np.cos(np.arange(12) * (7. / 6.) * np.pi)
    tonal_matrix[2] = r2 * np.sin(np.arange(12) * (3. / 2.) * np.pi)
    tonal_matrix[3] = r2 * np.cos(np.arange(12) * (3. / 2.) * np.pi)
    tonal_matrix[4] = r3 * np.sin(np.arange(12) * (2. / 3.) * np.pi)
    tonal_matrix[5] = r3 * np.cos(np.arange(12) * (2. / 3.) * np.pi)
    return tonal_matrix


def get_num_pitch_used(pianoroll):
    """Return the number of unique pitches used in a piano-roll."""
    return np.sum(np.sum(pianoroll, 0) > 0)


def get_qualified_note_rate(pianoroll, threshold=2):
    """Return the ratio of the number of the qualified notes (notes longer than
    `threshold` (in time step)) to the total number of notes in a piano-roll."""
    padded = np.pad(pianoroll.astype(int), ((1, 1), (0, 0)), 'constant')
    diff = np.diff(padded, axis=0)
    flattened = diff.T.reshape(-1, )
    onsets = (flattened > 0).nonzero()[0]
    offsets = (flattened < 0).nonzero()[0]
    num_qualified_note = (offsets - onsets >= threshold).sum()
    return num_qualified_note / len(onsets)


def get_polyphonic_ratio(pianoroll, threshold=2):
    """Return the ratio of the number of time steps where the number of pitches
    being played is larger than `threshold` to the total number of time steps"""
    return np.sum(np.sum(pianoroll, 1) >= threshold) / pianoroll.shape[0]


def get_in_scale(chroma, scale_mask=None):
    """Return the ratio of chroma."""
    measure_chroma = np.sum(chroma, axis=0)
    in_scale = np.sum(np.multiply(measure_chroma, scale_mask, dtype=float))
    return in_scale / np.sum(chroma)


def get_drum_pattern(measure, drum_filter):
    """Return the drum_pattern metric value."""
    padded = np.pad(measure, ((1, 0), (0, 0)), 'constant')
    measure = np.diff(padded, axis=0)
    measure[measure < 0] = 0

    max_score = 0
    for i in range(6):
        cdf = np.roll(drum_filter, i)
        score = np.sum(np.multiply(cdf, np.sum(measure, 1)))
        if score > max_score:
            max_score = score

    return max_score / np.sum(measure)


def get_harmonicity(bar_chroma1, bar_chroma2, resolution, tonal_matrix=None):
    """Return the harmonicity metric value"""
    if tonal_matrix is None:
        tonal_matrix = get_tonal_matrix()
        warnings.warn("`tonal matrix` not specified. Use default tonal matrix",
                      RuntimeWarning)
    score_list = []
    for r in range(bar_chroma1.shape[0] // resolution):
        start = r * resolution
        end = (r + 1) * resolution
        beat_chroma1 = np.sum(bar_chroma1[start:end], 0)
        beat_chroma2 = np.sum(bar_chroma2[start:end], 0)
        score_list.append(tonal_dist(beat_chroma1, beat_chroma2, tonal_matrix))
    return np.nanmean(score_list) if not np.isnan(score_list).all() else 0


def to_chroma(pianoroll):
    """Return the chroma features (not normalized)."""
    padded = np.pad(pianoroll, ((0, 0), (0, 12 - pianoroll.shape[1] % 12)),
                    'constant')
    return np.sum(np.reshape(padded, (pianoroll.shape[0], 12, -1)), 2)


def tonal_dist(chroma1, chroma2, tonal_matrix=None):
    """Return the tonal distance between two chroma features."""
    if tonal_matrix is None:
        tonal_matrix = get_tonal_matrix()
        warnings.warn("`tonal matrix` not specified. Use default tonal matrix",
                      RuntimeWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        chroma1 = chroma1 / np.sum(chroma1)
        chroma2 = chroma2 / np.sum(chroma2)
    result1 = np.matmul(tonal_matrix, chroma1)
    result2 = np.matmul(tonal_matrix, chroma2)
    return np.linalg.norm(result1 - result2)


def plot_histogram(hist, fig_dir=None, title=None, max_hist_num=None):
    """Plot the histograms of the statistics"""
    import matplotlib.pyplot as plt
    hist = hist[~np.isnan(hist)]
    u_value = np.unique(hist)

    hist_num = len(u_value)
    if max_hist_num is not None:
        if len(u_value) > max_hist_num:
            hist_num = max_hist_num

    fig = plt.figure()
    plt.hist(hist, hist_num)
    if title is not None:
        plt.title(title)
    if fig_dir is not None and title is not None:
        fig.savefig(os.path.join(fig_dir, title))
    plt.close(fig)


class Metrics(object):
    """Class for metrics.
    """

    def __init__(self, config):
        self.metric_map = config['metric_map']
        self.metric_list = config['metric_list']
        self.tonal_distance_pairs = config['tonal_distance_pairs']
        self.track_names = config['track_names']
        self.beat_resolution = config['beat_resolution']
        self.drum_filter = config['drum_filter']
        self.scale_mask = config['scale_mask']
        self.tonal_matrix = get_tonal_matrix(
            config['tonal_matrix_coefficient'][0],
            config['tonal_matrix_coefficient'][1],
            config['tonal_matrix_coefficient'][2]
        )

        self.metric_names = [
            'avg_pitch_shift',
            'pitch_used',
            'avg_IOI',
            'polyphonicity',
            'in_scale',
            'drum_pattern',
            #'total_pitch_class_histogram',
            # 'note_length_hist',

        ]
        self.metric_list_names = [
            'total_pitch_class_histogram',
            'note_length_hist',

        ]

    def print_metrics_mat(self, metrics_mat):
        """Print the intratrack metrics as a nice formatting table"""
        print(' ' * 12, '\t'.join(['{:^14}'.format(metric_name)
                                  for metric_name in self.metric_names]))

        for t, track_name in enumerate(self.track_names):
            value_str = []
            for m in range(len(self.metric_names)):
                if np.isnan(metrics_mat[m, t]):
                    value_str.append('{:14}'.format(''))
                else:
                    value_str.append('{:^14}'.format('{:6.4f}'.format(
                        metrics_mat[m, t])))
                value_str.append('\t')

            print('{:12}'.format(track_name), '\t', ' '.join(value_str))

    def print_metrics_pair(self, pair_matrix):
        """Print the intertrack metrics as a nice formatting table"""
        for idx, pair in enumerate(self.tonal_distance_pairs):
            print("{:12} \t {:12} \t {:12.5f}".format(
                self.track_names[pair[0]], self.track_names[pair[1]],
                pair_matrix[idx]))
    def print_metrics_list(self, list_matrix):
        """Print the intertrack metrics as a nice formatting table"""
        list_matrix = np.around(list_matrix, 3)
        for m in range(len(self.metric_list_names)):
            value_str = []
            print(self.metric_list_names[m])
            for t, track_name in enumerate(self.track_names):
                if np.isnan(list_matrix[m, t,:]).all():
                    value_str.append('{:14}'.format(''))
                else:
                    print(track_name,"\t".join(str(lm) for lm in list_matrix[m, t,:]))



    def eval(self, bars, verbose=False, mat_path=None, fig_dir=None, save=False):
        """Evaluate the input bars with the metrics"""
        score_matrix = np.empty((len(self.metric_names), len(self.track_names),
                                 bars.shape[0]))
        score_matrix.fill(np.nan)
        score_pair_matrix = np.zeros((len(self.tonal_distance_pairs),
                                      bars.shape[0]))
        score_pair_matrix.fill(np.nan)
        score_list_matrix = np.zeros((2, len(self.track_names),
                                      bars.shape[0], 12))
        score_list_matrix.fill(np.nan)

        for b in range(bars.shape[0]):
            for t in range(len(self.track_names)):
                is_empty_bar = ~np.any(bars[b, ..., t])
                if is_empty_bar:
                    continue
                binarized = (bars[b:b + 1, ...] > 0)
                binarized = binarized.reshape(-1, 2, binarized.shape[1] // 2, binarized.shape[2], binarized.shape[3])
                config = CONFIG['model']
                midi_io.save_midi('./tmp.mid', binarized, config)
                feature = extract_feature('./tmp.mid')
                if self.metric_map[0, t]:
                    # score_matrix[0, t, b] = is_empty_bar
                    score_matrix[0, t, b] = avg_pitch_shift(feature, track_num=t + 2)

                if self.metric_map[1, t]:
                    score_matrix[1, t, b] = get_num_pitch_used(bars[b, ..., t])
                if self.metric_map[2, t]:
                    # score_matrix[2, t, b] = get_qualified_note_rate(bars[b, ..., t])
                    score_matrix[2, t, b] = avg_IOI(bars[b, ..., t])
                if self.metric_map[3, t]:
                    score_matrix[3, t, b] = get_polyphonic_ratio(
                        bars[b, ..., t])
                if self.metric_map[4, t]:
                    score_matrix[4, t, b] = get_in_scale(
                        to_chroma(bars[b, ..., t]), self.scale_mask)
                if self.metric_map[5, t]:
                    score_matrix[5, t, b] = get_drum_pattern(bars[b, ..., t],
                                                             self.drum_filter)
                if self.metric_map[6, t]:
                    # score_matrix[6, t, b] = get_num_pitch_used(to_chroma(bars[b, ..., t]))
                    score_list_matrix[0, t, b] = total_pitch_class_histogram(bars[b, ..., t])
                if self.metric_map[7, t]:
                    # score_matrix[6, t, b] = get_num_pitch_used(to_chroma(bars[b, ..., t]))
                    score_list_matrix[1, t, b] = note_length_hist(feature, track_num=t + 2)

            for p, pair in enumerate(self.tonal_distance_pairs):
                score_pair_matrix[p, b] = get_harmonicity(
                    to_chroma(bars[b, ..., pair[0]]),
                    to_chroma(bars[b, ..., pair[1]]), self.beat_resolution,
                    self.tonal_matrix)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            score_matrix_mean = np.nanmean(score_matrix, axis=2)
            score_pair_matrix_mean = np.nanmean(score_pair_matrix, axis=1)
            score_list_matrix_mean = np.nanmean(score_list_matrix, axis=2)

        if verbose:
            print("{:=^120}".format(' Evaluation '))
            print('Data Size:', bars.shape)
            print("{:-^120}".format('Intratrack Evaluation'))
            self.print_metrics_mat(score_matrix_mean)
            print("{:-^120}".format('Intertrack Evaluation'))
            self.print_metrics_list(score_list_matrix_mean)
            print("{:-^120}".format('Intertrack Evaluation'))
            self.print_metrics_pair(score_pair_matrix_mean)
        if save:
            if fig_dir is not None:
                if not os.path.exists(fig_dir):
                    os.makedirs(fig_dir)
                if verbose:
                    print('[*] Plotting...')
                for m, metric_name in enumerate(self.metric_names):
                    for t, track_name in enumerate(self.track_names):
                        if self.metric_map[m, t]:
                            temp = '-'.join(track_name.replace('.', ' ').split())
                            title = '_'.join([metric_name, temp])
                            plot_histogram(score_matrix[m, t], fig_dir=fig_dir,
                                           title=title, max_hist_num=20)
                if verbose:
                    print("Successfully saved to", fig_dir)

            if mat_path is not None:
                if not mat_path.endswith(".npy"):
                    mat_path = mat_path + '.npy'
                info_dict = {
                    'score_matrix_mean': score_matrix_mean,
                    'score_pair_matrix_mean': score_pair_matrix_mean,
                    'score_list_matrix_mean': score_list_matrix_mean}
                if verbose:
                    print('[*] Saving score matrices...')
                np.save(mat_path, info_dict)
                if verbose:
                    print("Successfully saved to", mat_path)

        return score_matrix, score_list_matrix

def eval_dataset(filepath, result_dir, location, config):
    """Run evaluation on a dataset stored in either shared array (if `location`
    is 'sa') or in hard disk (if `location` is 'hd') and save the results to the
    given directory.

    """

    def load_data(filepath, location):
        """Load and return the training data."""
        print('[*] Loading data...')

        # Load data from SharedArray
        if location == 'sa':
            import SharedArray as sa
            data = sa.attach(filepath)

        # Load data from hard disk
        elif location == 'hd':
            if os.path.isabs(filepath):
                data = np.load(filepath)
            else:
                root = os.path.dirname(os.path.dirname(
                    os.path.realpath(__file__)))
                data = np.load(os.path.abspath(os.path.join(
                    root, 'training_data', filepath)))

        else:
            raise ValueError("Unrecognized value for `location`")

        # Reshape data
        data = data.reshape(-1, config['num_timestep'], config['num_pitch'],
                            config['num_track'])

        return data

    print('[*] Loading data...')
    data = load_data(filepath, location)

    print('[*] Running evaluation')
    metrics = Metrics(config)
    _ = metrics.eval(data, verbose=True,
                     mat_path=os.path.join(result_dir, 'score_matrices.npy'),
                     fig_dir=result_dir, save=True)


def eval_samples(data, config):
    data = data.reshape(-1, config['num_timestep'], config['num_pitch'], config['num_track'])

    print('[*] Running evaluation')
    metrics = Metrics(config)
    score_matrix,score_list_matrix = metrics.eval(data, verbose=True)
    return score_matrix,score_list_matrix


def print_mat_file(mat_path, config):
    """Print the score matrices stored in a file."""
    metrics = Metrics(config)
    with np.load(mat_path) as loaded:
        metrics.print_metrics_mat(loaded['score_matrix_mean'])
        metrics.print_metrics_list(loaded['score_list_matrix_mean'])
        metrics.print_metrics_pair(loaded['score_pair_matrix_mean'])
