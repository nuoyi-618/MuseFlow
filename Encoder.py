#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: 丁凡彧
import os

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from load_model import Model
import model.flow_layers as flow
class EncoderModel(object):

    def __init__(self, save_model_dir, num_timestep = 768 ,num_pitch=60):
        self.save_model_dir = save_model_dir
        self.num_timestep = num_timestep
        self.num_pitch = num_pitch
        self.metric_names = ['avg_IOI', 'num_pitch_used']

        self.graph = tf.get_default_graph()
        self.sess = tf.Session(graph=self.graph)
        meta_graph_def =tf.saved_model.loader.load(self.sess, ["serve"], save_model_dir)

        signature = meta_graph_def.signature_def
        # Find the tensor name for the input and output from signature
        signature_key = 'serving_default'
        input_key = 'input'
        output_key = 'output'
        x_tensor_name = signature[signature_key].inputs[input_key].name
        y_tensor_name = signature[signature_key].outputs[output_key].name
        # Take tensor and inference

        self.z_op = self.sess.graph.get_tensor_by_name(x_tensor_name)
        self.generate_op = self.sess.graph.get_tensor_by_name(y_tensor_name)
    def encoder(self,data):
        x_encoder = self.sess.run(self.generate_op, feed_dict={self.z_op: data})
        return x_encoder

    def avg_IOI(self, pianoroll):
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
        if len(ioi) == 0:
            return None
        avg_ioi = np.mean(ioi)
        return avg_ioi

    def get_num_pitch_used(self,pianoroll):
        """Return the number of unique pitches used in a piano-roll."""
        return np.sum(np.sum(pianoroll, 0) > 0)

    def eval(self, bars):

        score_matrix = np.empty((len(self.metric_names), bars.shape[0]))
        score_matrix.fill(np.nan)
        for b in range(bars.shape[0]):
            score_matrix[0, b] = self.avg_IOI(bars[b, ...])
            score_matrix[1, b] = self.get_num_pitch_used(bars[b, ...])
        return score_matrix
    def mean_v(self, data):
        score = self.eval(data[..., 1])
        dZ=[]
        for i in range(2):
            ioi = score[i, ...]
            ioi = ioi[~np.isnan(ioi)]
            # The top ten percent and the bottom ten percent are divided into two categories, from small to large
            print('top1% '+self.metric_names[i]+' is:', np.percentile(ioi, 1))
            print('tail1% '+self.metric_names[i]+' is:', np.percentile(ioi, 99))
            min_poi = np.percentile(ioi, 1)
            max_poi = np.percentile(ioi, 99)
            # To obtain the subscript
            poi = np.where(score[i, ...] <= min_poi)
            top = data[poi, ...][0]

            poi = np.where(score[i, ...] >= max_poi)
            tail = data[poi, ...][0]

            Z1 = self.encoder(top[:60,...])
            mean_Z1 = np.mean(Z1, axis=0)

            Z2 = self.encoder(tail[:60,...])
            mean_Z2 = np.mean(Z2, axis=0)

            ioi_dZ = mean_Z1-mean_Z2
            dZ.append(ioi_dZ)
        return dZ
    def close(self):
        self.sess.close()