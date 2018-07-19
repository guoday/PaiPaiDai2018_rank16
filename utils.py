# Copyright 2017 Google Inc. All Rights Reserved.
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

"""Generally useful utility functions."""
from __future__ import print_function

import codecs
import collections
import json
import math
import os
import sys
import time
import numpy as np

def print_out(s, f=None, new_line=True):
  """Similar to print but with support to flush and output to a file."""
  if isinstance(s, bytes):
    s = s.decode("utf-8")

  if f:
    f.write(s.encode("utf-8"))
    if new_line:
      f.write(b"\n")

  # stdout
  out_s = s.encode("utf-8")
  if not isinstance(out_s, str):
    out_s = out_s.decode("utf-8")
  print(out_s, end="", file=sys.stdout)

  if new_line:
    sys.stdout.write("\n")
  sys.stdout.flush()


def print_hparams(hparams, skip_patterns=None, header=None):
  """Print hparams, can skip keys based on pattern."""
  if header: print_out("%s" % header)
  values = hparams.values()
  for key in sorted(values.keys()):
    if not skip_patterns or all(
        [skip_pattern not in key for skip_pattern in skip_patterns]):
      print_out("  %s=%s" % (key, str(values[key])))


def print_step_info(prefix, global_step, info):
  """Print all info at the current global step."""
  print_out(
      "%sstep %d lr %g step-time %.2fs loss %.5f gN %.2f, %s" %
      (prefix, global_step, info["learning_rate"], info["avg_step_time"],
       info["train_ppl"], info["avg_grad_norm"], time.ctime()))   
def mrr(dist_list, gold):
  """
  dist_list: list of list of label probability for all labels.
  gold: list of gold indexes.

  Get mean reciprocal rank. (this is slow, as have to sort for 10K vocab)
  """
  mrr_per_example = []
  dist_arrays = np.array(dist_list)
  dist_sorted = np.argsort(-dist_arrays, axis=1)
  for ind, gold_i in enumerate(gold):
    rr_per_array = []
    sorted_index = dist_sorted[ind, :]
    for k in range(len(sorted_index)):
        if sorted_index[k] in gold_i :
          rr_per_array.append(1.0 / (k + 1))
    mrr_per_example.append(np.mean(rr_per_array))
  return sum(mrr_per_example) * 1.0 / len(mrr_per_example)    