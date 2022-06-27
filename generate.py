import argparse
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from load_model import Model
import numpy as np
import model.flow_layers as flow
from data.config import CONFIG
num_timestep = 768
num_pitch = 60

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'file_path', './music_data/song_of_joy.mid',
    'Main melody file path ')

flags.DEFINE_string(
    'save_dir', './music_data/result/',
    'Save file path')

def combine(paino,sw=True,filt=False):
    path_drum = './model/single/drum/decoder/0/'
    path_bass = './model/single/bass/0/decoder/0/'
    path_string = './model/single/string/decoder/0/'
    path_guitar = './model/single/guitar/decoder/0/'
    #graph = tf.get_default_graph()
    #sess = tf.Session(graph=graph)
    drums = Model(path_drum)
    drums_output = drums.generate(paino,paino.shape[0], sw=sw, threshold=0.5)

    strings = Model(path_string)
    strings_output = strings.generate(paino, paino.shape[0], sw=sw, threshold=0.3)



    guitar = Model(path_guitar)
    guitar_output = guitar.generate(paino,paino.shape[0], sw=sw, threshold=0.2)


    bass = Model(path_bass)
    bass_output = bass.generate(paino,paino.shape[0], sw=sw, threshold=0.6)

    if filt:
        strings_output = strings.medfilt(strings_output, 17)
        guitar_output = guitar.medfilt(guitar_output, 11)
        bass_output = bass.medfilt(bass_output, 17)
        strings_output = strings.medfilt(strings_output, 17)
        guitar_output = guitar.medfilt(guitar_output, 11)
        bass_output = bass.medfilt(bass_output, 17)
    compiled_list=[drums_output[...,-1:], guitar_output, bass_output[...,-1:], strings_output[...,-1:]]
    result = np.concatenate(compiled_list, axis=-1)
    result = result.reshape(-1, 2, num_timestep // 2, num_pitch, 5)

    return result
def main(unused_argv):
    file = FLAGS.file_path
    x_pre = flow.SlidingWindow(file, 24, num_timestep, num_pitch, num_consecutive_bar=8*4)
    result = combine(x_pre,filt=True)
    np.save(FLAGS.save_dir+'result.npy', result)
    flow.save_samples(CONFIG['model'], 'result',result,path=FLAGS.save_dir, save_midi=True,shape=(1,result.shape[0]))


if __name__ == '__main__':
    tf.app.run(main)
