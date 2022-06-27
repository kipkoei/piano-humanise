# Copyright 2022 The Magenta Authors.
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

"""MusicVAE data library additions"""
import note_seq
from note_seq import sequences_lib, constants
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

PIANO_MIN_MIDI_PITCH = 21
PIANO_MAX_MIDI_PITCH = 108
MIN_MIDI_PITCH = 0
MAX_MIDI_PITCH = 127
MIDI_PITCHES = 128

MAX_INSTRUMENT_NUMBER = 127


class NoteSequenceAugmenter(object):
  """Class for augmenting NoteSequences.

  Attributes:
    transpose_range: A tuple containing the inclusive, integer range of
        transpose amounts to sample from. If None, no transposition is applied.
    stretch_range: A tuple containing the inclusive, float range of stretch
        amounts to sample from.
  Returns:
    The augmented NoteSequence.
  """

  def __init__(self, transpose_range=None, stretch_range=None, min_pitch=constants.MIN_MIDI_PITCH, max_pitch=constants.MAX_MIDI_PITCH):
    self._transpose_range = transpose_range
    self._stretch_range = stretch_range
    self._min_pitch = min_pitch
    self._max_pitch = max_pitch

  def augment(self, note_sequence):
    """Python implementation that augments the NoteSequence.

    Args:
      note_sequence: A NoteSequence proto to be augmented.

    Returns:
      The randomly augmented NoteSequence.
    """
    transpose_min, transpose_max = (
        self._transpose_range if self._transpose_range else (0, 0))
    stretch_min, stretch_max = (
        self._stretch_range if self._stretch_range else (1.0, 1.0))

    return sequences_lib.augment_note_sequence(
        note_sequence,
        stretch_min,
        stretch_max,
        transpose_min,
        transpose_max,
        self._min_pitch,
        self._max_pitch,
        delete_out_of_range_notes=True)

  def tf_augment(self, note_sequence_scalar):
    """TF op that augments the NoteSequence."""
    def _augment_str(note_sequence_str):
      note_sequence = note_seq.NoteSequence.FromString(
          note_sequence_str.numpy())
      augmented_ns = self.augment(note_sequence)
      return [augmented_ns.SerializeToString()]

    augmented_note_sequence_scalar = tf.py_function(
        _augment_str,
        inp=[note_sequence_scalar],
        Tout=tf.string,
        name='augment')
    augmented_note_sequence_scalar.set_shape(())
    return augmented_note_sequence_scalar

