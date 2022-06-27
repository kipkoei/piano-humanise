import os
import shutil
from magenta.models.music_vae.piano_humanize_converter import PianoHumanizeConverter
import numpy as np

from magenta.models.music_vae import configs
import note_seq
from note_seq import sequences_lib
from humanize import change_tempo, start_notes_at_0

# The purpose of this script is solely to count the number of (training) samples which will be produced using a certain config and dataset.

model_name = 'lbau_2bar_eval'
data_path = 'data/validation'

config = configs.CONFIG_MAP[model_name]
dc: PianoHumanizeConverter = config.data_converter

def normalize(ns):
    s = start_notes_at_0(ns)
    s = change_tempo(s, s.tempos[0].qpm)
    for ts in s.time_signatures[1:]:
        s.time_signatures.remove(ts)
    for t in s.tempos[1:]:
        s.tempos.remove(t)
    return s


def count(s):    
    def _extract_windows(tensor, window_size, hop_size):
      """Slide a window across the first dimension of a 2D tensor."""
      return [tensor[i:i+window_size, :] for i in range(0, len(tensor) - window_size + 1, hop_size)]
  
    quantized_sequence = sequences_lib.quantize_note_sequence(s, dc._steps_per_quarter)
    max_start_step = np.max([note.quantized_start_step for note in quantized_sequence.notes])

    # Round up so we pad to the end of the bar.
    total_bars = int(np.ceil((max_start_step + 1) / dc._steps_per_bar))
    max_step = dc._steps_per_bar * total_bars # 16 * 5
    fake_tensor = np.zeros((max_step, dc._num_keys))
    window_size = dc._steps_per_bar * dc._split_bars
    hop_size = dc._hop_size or window_size

    windows = _extract_windows(fake_tensor, window_size, hop_size)
    return len(windows)

files = [file for file in os.listdir(data_path) if file.endswith('.mid')]
total_count = 0

for file in files:
    print('Processing', file)
    s = note_seq.midi_file_to_note_sequence(os.path.join(data_path, file))
    s.filename = file
    s = normalize(s)
    seq_count = count(s) 
    total_count += seq_count
    print(seq_count, 'samples')
    
    
print('Total count:', total_count)