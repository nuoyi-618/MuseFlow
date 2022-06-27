#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# @author: 丁凡彧
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import numpy as np
import scipy.signal as signal
from tensorflow.compat.v1.graph_util import convert_variables_to_constants
import pypianoroll
import os
import sys

import data_processing.converter as converter
import data.metrics as metrics
import data.midi_io as midi_io
import data.image_io as image_io
import model.flow_layers as flow

class Model(object):

    def __init__(self, save_model_dir, num_timestep = 768 ,num_pitch=60):
        self.save_model_dir = save_model_dir
        self.num_timestep = num_timestep
        self.num_pitch = num_pitch

        self.graph = tf.get_default_graph()
        self.sess = tf.Session(graph=self.graph)
        meta_graph_def =tf.saved_model.loader.load(self.sess, ["serve"], save_model_dir)

        signature = meta_graph_def.signature_def
        signature_key = 'serving_default'
        input_key = 'input'
        output_key = 'output'
        x_tensor_name = signature[signature_key].inputs[input_key].name
        y_tensor_name = signature[signature_key].outputs[output_key].name


        self.z_op = self.sess.graph.get_tensor_by_name(x_tensor_name)
        self.generate_op = self.sess.graph.get_tensor_by_name(y_tensor_name)

    def generate(self,x_pre , n,sw=True,threshold=0.5, std=1,get_z=False):
        """ Build operation (note: batch_size is assumed to be 1)
            :param num: specifies the number of samples to be generated
            :return: shape(num, 256)
        """
        result = []
        z=[]
        for i in range(n):
            bar_id = i
            if sw:
                if bar_id == 0:
                    z1 = np.array(np.random.randn(1, self.num_timestep // 2, self.num_pitch, 1)) * std
                z2 = np.array(np.random.randn(1, self.num_timestep // 2, self.num_pitch, 1)) * std
                z_sample = np.concatenate([z1, z2], axis=1)
            else:
                z_sample = np.array(np.random.randn(1, self.num_timestep, self.num_pitch, 1)) * std
            z.append(z_sample)
            z_samples = np.insert(z_sample, 0, values=x_pre[bar_id, :, :, 0], axis=-1).astype('float32')
            x_decoded = self.sess.run(self.generate_op, feed_dict={self.z_op: z_samples})
            digit = x_decoded[0]
            digit = np.array(digit)
            digit[(digit<threshold)]=0
            melodies = digit #np.clip(digit, 0, 1)
            if sw:
                melodies = melodies.reshape(-1, self.num_timestep // 2, self.num_pitch, 2)
                if bar_id == 0:
                    result.append(melodies[0, :, :, :])
                    result.append(melodies[1, :, :, :])
                else:
                    result.append(melodies[1, :, :, :])
                z1 = z2
            else:
                melodies = melodies.reshape(self.num_timestep, self.num_pitch, 2)
                result.append(melodies)

        result_sample = np.array(result).astype(bool)
        #result_sample = result_sample.reshape(-1, 2, self.num_timestep // 2, self.num_pitch, 2)

        tf.reset_default_graph()
        if get_z:
            return result_sample,z
        return result_sample#[...,-1]

    def medfilt(self, sampel,sig=21):

        for i in range(sampel.shape[0]):
            l = sampel[i,:,:,1]
            sampel[i,:,:,1] = signal.medfilt(l, (sig, 1))
        return sampel
    def control_generate(self,x_pre,z,cz,ma,n,sw=True,threshold=0.5, std=1):
        """ Build operation (note: batch_size is assumed to be 1)
            :param num: Specifies the number of samples to be generated,z original random variable, Cz control variable, ma= parameter
            :return: shape(num, 256)
        """
        result = []

        for i in range(n):
            bar_id = i

            z_sample = std*z[bar_id]+ma*cz
            z_samples = np.insert(z_sample, 0, values=x_pre[bar_id, :, :, 0], axis=-1).astype('float32')
            x_decoded = self.sess.run(self.generate_op, feed_dict={self.z_op: z_samples})
            digit = x_decoded[0]
            digit = np.array(digit)
            digit[(digit<threshold)]=0
            melodies = digit #np.clip(digit, 0, 1)
            if sw:
                melodies = melodies.reshape(-1, self.num_timestep // 2, self.num_pitch, 2)
                if bar_id == 0:
                    result.append(melodies[0, :, :, :])
                    result.append(melodies[1, :, :, :])
                else:
                    result.append(melodies[1, :, :, :])
            else:
                melodies = melodies.reshape(self.num_timestep, self.num_pitch, 2)
                result.append(melodies)
        result_sample = np.array(result).astype(bool)
        #result_sample = result_sample.reshape(-1, 2, self.num_timestep // 2, self.num_pitch, 2)
        tf.reset_default_graph()
        return result_sample#[...,-1]