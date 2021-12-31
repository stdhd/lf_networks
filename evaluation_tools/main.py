# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example code that computes FVD for some empty frames.

The FVD for this setup should be around 131.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from natsort import natsorted
import tensorflow.compat.v1 as tf
import frechet_video_distance as fvd
import glob
import numpy as np
# Number of videos must be divisible by 16.
from PIL import Image
import os

BATCH_SIZE = 16


def traverse_video_dir(root_dir, video_length, size=64):
    vlist = natsorted(os.listdir(root_dir))
    print('length of video list is ', len(vlist))
    n_videos = (len(vlist) // BATCH_SIZE) * BATCH_SIZE
    print('number of considered videos is  ', len(vlist))
    vlist = vlist[:n_videos]
    result_array = np.empty(shape=(len(vlist), video_length, size, size, 3))

    for vi, video_dir in enumerate(vlist):
        filelist = natsorted(glob.glob(os.path.join(root_dir, video_dir) + '/*.png'))
        try:
            result_array[vi] = np.array([np.array(Image.open(fname)) for fname in filelist[:video_length]])
        except:
            print(filelist)
            exit()

    return result_array, len(vlist) // BATCH_SIZE


def traverse_video_dir_constant(root_dir, video_length, size=64):
    vlist = natsorted(os.listdir(root_dir))#[:160]
    print('length of video list is ', len(vlist))
    n_videos = (len(vlist) // BATCH_SIZE) * BATCH_SIZE
    print('number of considered videos is  ', len(vlist))
    vlist = vlist[:n_videos]
    result_array = np.empty(shape=(len(vlist), video_length, size, size, 3))

    for vi, video_dir in enumerate(vlist):
        filelist = natsorted(glob.glob(os.path.join(root_dir, video_dir) + '/*.png'))
        fname = filelist[0]
        try:
            result_array[vi] = np.array([np.array(Image.open(fname)) for abc in filelist[:video_length]])
        except:
            print(filelist)
            exit()

    return result_array, len(vlist) // BATCH_SIZE

def traverse_video_dir_same_image_baseline(root_dir, video_length, size=64):
    vlist = os.listdir(root_dir)
    n_videos = (len(vlist) // BATCH_SIZE) * BATCH_SIZE
    result_array = np.empty(shape=(len(vlist), video_length, size, size, 3))
    for vi, video_dir in enumerate(vlist):
        filelist = glob.glob(os.path.join(root_dir, video_dir) + '/*.png')
        result_array[vi] = np.array([np.array(Image.open(fname)) for fname in filelist[:video_length]])

    return result_array, len(vlist) // BATCH_SIZE


def main(argv):
    first, n_batches = traverse_video_dir('references', video_length=11)
    second, _ =        traverse_video_dir('predictions', video_length=11)

    result_fvd = np.zeros((n_batches))
    del argv
    with tf.device('/GPU:0'):
        with tf.Graph().as_default():
            for i in range(n_batches):
                first_set_of_videos = tf.convert_to_tensor(
                    first[
                    i * BATCH_SIZE: (i + 1) * BATCH_SIZE])
                second_set_of_videos = tf.convert_to_tensor(second[i * BATCH_SIZE: (i + 1) * BATCH_SIZE])

                result = fvd.calculate_fvd(
                    fvd.create_id3_embedding(fvd.preprocess(first_set_of_videos,
                                                            (224, 224))),
                    fvd.create_id3_embedding(fvd.preprocess(second_set_of_videos,
                                                            (224, 224))))


                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.tables_initializer())
                    fvd_val = sess.run(result)
                result_fvd[i] = fvd_val
                print(i, fvd_val)

    print(result_fvd)
    print(result_fvd.mean(), result_fvd.std())


if __name__ == "__main__":
  tf.app.run(main)
