#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import time
import numpy as np

from google.protobuf import text_format
import tensorflow as tf
from tensorflow.python.client import timeline

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.compat.v1.GraphDef()

  import os
  file_ext = os.path.splitext(model_file)[1]

  with open(model_file, "rb") as f:
    if file_ext == '.pbtxt':
      text_format.Merge(f.read(), graph_def)
    else:
      graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def, name='')

  return graph

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_graph", default=None,
                      help="graph/model to be executed")
  parser.add_argument("--input_height", default=224,
                      type=int, help="input height")
  parser.add_argument("--input_width", default=224,
                      type=int, help="input width")
  parser.add_argument("--batch_size", default=32,
                      type=int, help="batch size")
  parser.add_argument("--input_layer", default="input",
                      help="name of input layer")
  parser.add_argument("--output_layer", default="MobilenetV1/Predictions/Reshape_1",
                      help="name of output layer")
  parser.add_argument(
      '--num_inter_threads',
      help='number threads across operators',
      type=int, default=1)
  parser.add_argument(
      '--num_intra_threads',
      help='number threads for an operator',
      type=int, default=1)
  parser.add_argument("--warmup_steps", type=int, default=10,
                      help="number of warmup steps")
  parser.add_argument("--steps", type=int, default=50, help="number of steps")
  args = parser.parse_args()

  if args.input_graph:
    model_file = args.input_graph
  else:
    sys.exit("Please provide a graph file.")
  input_height = args.input_height
  input_width = args.input_width
  batch_size = args.batch_size
  input_layer = args.input_layer
  output_layer = args.output_layer
  warmup_steps = args.warmup_steps
  steps = args.steps
  assert steps > 10, "Benchmark steps should be at least 10."
  num_inter_threads = args.num_inter_threads
  num_intra_threads = args.num_intra_threads
  profile_mode = os.environ['PROFILE_MODE'] if 'PROFILE_MODE' in os.environ.keys() else None
  timeline_path = os.environ['TIMELINE_PATH'] if 'TIMELINE_PATH' in os.environ.keys() else None

  graph = load_graph(model_file)

  input_tensor = graph.get_tensor_by_name(input_layer + ":0");
  output_tensor = graph.get_tensor_by_name(output_layer + ":0");

  config = tf.compat.v1.ConfigProto()
  config.inter_op_parallelism_threads = num_inter_threads
  config.intra_op_parallelism_threads = num_intra_threads

  with tf.compat.v1.Session(graph=graph, config=config) as sess:
    input_shape = [batch_size, input_height, input_width, 3]
    images = tf.random.truncated_normal(
          input_shape,
          dtype=tf.float32,
          stddev=10,
          name='synthetic_images')
    image_data = sess.run(images)

    sys.stdout.flush()
    print("[Running warmup steps...]")
    for t in range(warmup_steps):
      start_time = time.time()
      sess.run(output_tensor, {input_tensor: image_data})
      elapsed_time = time.time() - start_time
      if((t+1) % 10 == 0):
        print("steps = {0}, {1} images/sec"
              "".format(t+1, batch_size/elapsed_time))

    print("[Running benchmark steps...]")
    total_time = 0
    options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()
    for t in range(steps):
      start_time = time.time()
      if (t + 1) % 10 == 0 and (profile_mode == 'profile' or profile_mode == 'timeline'):
        results = sess.run(output_tensor, {input_tensor: image_data},
                           options=options, run_metadata=run_metadata)
      else:
        results = sess.run(output_tensor, {input_tensor: image_data})
      elapsed_time = time.time() - start_time

      if (t + 1) % 10 == 0:
        print("steps = {0}, {1} sec, {2} images/sec"
              .format(t+1, str(elapsed_time), str(batch_size/elapsed_time)))
        if profile_mode == 'profile':
          profiler = tf.compat.v1.profiler.Profiler(sess.graph)
          step = -1
          profiler.add_step(step, run_metadata)
          option_builder = tf.compat.v1.profiler.ProfileOptionBuilder
          opts = (option_builder(option_builder.time_and_memory()).
                  select(['micros','bytes','occurrence']).order_by('micros')
                  .with_max_depth(30).build())
          profiler.profile_operations(options=opts)
        elif profile_mode == 'timeline':
          fetched_timeline = timeline.Timeline(run_metadata.step_stats)
          chrome_trace = fetched_timeline.generate_chrome_trace_format()
          with tf.compat.v1.gfile.GFile(timeline_path, 'w') as f:
            f.write(chrome_trace)
      
      if t + 1 <= steps * 0.9:
        total_time += elapsed_time

    eval_interations = int(steps * 0.9)
    time_average = total_time / eval_interations
    print('Batchsize: {0}'.format(str(batch_size)))
    print('Latency: {0:.4f} ms'.format(time_average * 1000))
    print('Throughput: {0:.4f} samples/s'.format(batch_size / time_average))
