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

"""Additional configuration for MusicVAE models."""
from magenta.common import merge_hparams
from magenta.contrib import training as contrib_training
from magenta.models.music_vae import data
from magenta.models.music_vae import lstm_models
from magenta.models.music_vae.base_model import MusicVAE
from magenta.models.music_vae.configs import Config
from magenta.models.music_vae.piano_lstm_decoder import PianoLstmDecoder
from magenta.models.music_vae.piano_humanize_converter import PianoHumanizeConverter

HParams = contrib_training.HParams

CONFIG_MAP = {}
PIANO_PITCH_CLASSES = [[p] for p in range(data.PIANO_MIN_MIDI_PITCH, data.PIANO_MAX_MIDI_PITCH + 1)]
PIANO_PITCH_CLASSES_AUG = [[p] for p in range(data.PIANO_MIN_MIDI_PITCH - 3, data.PIANO_MAX_MIDI_PITCH + 4)]

# LBau
CONFIG_MAP['lbau_2bar'] = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(),
                   PianoLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            learning_rate=0.0005,
            batch_size=512,
            max_seq_len=16 * 2,  # 2 bars w/ 16 steps per bar
            z_size=512,
            enc_rnn_size=[2048, 2048],
            dec_rnn_size=[1024, 1024],
            free_bits=256,
            max_beta=0.2,
            dropout_keep_prob=0.3,
        )),
    note_sequence_augmenter=data.NoteSequenceAugmenter(
        transpose_range=(-3, 3), min_pitch=data.PIANO_MIN_MIDI_PITCH, max_pitch=data.PIANO_MAX_MIDI_PITCH),
    data_converter=PianoHumanizeConverter(
        split_bars=2, steps_per_quarter=4, quarters_per_bar=4,
        max_tensors_per_notesequence=200, humanize=True,
        pitch_classes=PIANO_PITCH_CLASSES, hop_size=7,
        inference_pitch_classes=PIANO_PITCH_CLASSES, hits_as_controls=True),
    train_examples_path='data/processed/train.tfrecord'
)
# The eval config is an exact copy of the normal config, except that it doesn't have note sequence augmentation and no overlapping samples
CONFIG_MAP['lbau_2bar_eval'] = Config(
    model=CONFIG_MAP['lbau_2bar'].model,
    hparams=CONFIG_MAP['lbau_2bar'].hparams,
    data_converter=CONFIG_MAP['lbau_2bar'].data_converter,
    eval_examples_path='data/processed/validation.tfrecord'
)
CONFIG_MAP['lbau_2bar_eval'].data_converter._hop_size = None
