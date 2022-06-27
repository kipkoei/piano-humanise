import os
import random
import shutil
import copy
from magenta.models.music_vae.piano_humanize_converter import PianoHumanizeConverter
import numpy as np
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
import note_seq
from note_seq import sequences_lib
from tqdm import tqdm

# This is the main humanisation script. If you have a trained model and some MIDI recordings available, use this script to perform quantisation and subsequent humanisation.

model_name = 'lbau_2bar'
example_path = 'examples'

config = configs.CONFIG_MAP[model_name]
model = TrainedModel(config, 1, checkpoint_dir_or_path=f'models/{model_name}/train/')
dc: PianoHumanizeConverter = config.data_converter

quantize_path = os.path.join(example_path, 'quantized')
humanize_path = os.path.join(example_path, 'humanized')
recording_path = os.path.join(example_path, 'recording')

hop_size = int((dc._steps_per_bar * dc._split_bars) / 2)
dc._hop_size = hop_size

# This function comes fromt the GrooVAE notebook: https://colab.research.google.com/github/tensorflow/magenta-demos/blob/master/colab-notebooks/GrooVAE.ipynb
def change_tempo(note_sequence, new_tempo):
    new_sequence = copy.deepcopy(note_sequence)
    ratio = note_sequence.tempos[0].qpm / new_tempo
    for note in new_sequence.notes:
        note.start_time = note.start_time * ratio
        note.end_time = note.end_time * ratio
    for control in new_sequence.control_changes:
        control.time = control.time * ratio
    new_sequence.tempos[0].qpm = new_tempo
    return new_sequence

# This function comes fromt the GrooVAE notebook: https://colab.research.google.com/github/tensorflow/magenta-demos/blob/master/colab-notebooks/GrooVAE.ipynb
def start_notes_at_0(s):
    for n in s.notes:
        if n.start_time < 0:
            n.end_time -= n.start_time
            n.start_time = 0
    return s

def get_quantized(s, velocity: int, dc: PianoHumanizeConverter):
    print("..encoding")
    tensors, quantized = dc.to_tensors(s, True)
    print("..decoding")
    
    all_quantized =  np.concatenate([i[:hop_size, :] for i in quantized] + [quantized[-1][hop_size:,:]], axis=0)
    all_controls = np.concatenate([i[:hop_size, :] for i in tensors.controls] + [tensors.controls[-1][hop_size:,:]], axis=0)

    new_s = dc.from_tensors([all_quantized], [all_controls])[0]
    new_s.total_time = s.total_time
    new_s = change_tempo(new_s, s.tempos[0].qpm)

    if velocity != 0:
        for n in new_s.notes:
            n.velocity = velocity
            
    return new_s, tensors

def humanize(q, tensors, model: TrainedModel):
    encodings, mu, sigma = model.encode_tensors(tensors.inputs, tensors.lengths, tensors.controls)

    prev_decoded = None
    all_controls = np.empty((0, dc.control_depth))
    all_decoded = np.empty((0, dc.output_depth))

    for e, c in tqdm(list(zip(encodings, tensors.controls)), desc='decoding'):
        decoded = model.decode_to_tensors([e], length=2*hop_size, c_input=c)[0]
        all_controls = np.concatenate([all_controls, c[:hop_size, :]], axis=0)

        if prev_decoded is not None:
            for i in range(hop_size):
                step = (decoded[i, :] * (hop_size - i) + prev_decoded[i, :] * i) / float(hop_size)
                all_decoded = np.concatenate([all_decoded, np.reshape(step, (1, all_decoded.shape[1]))], axis=0)
        else:
            all_decoded = np.concatenate([all_decoded, decoded[:hop_size, :]], axis=0)

        prev_decoded = decoded

    all_controls = np.concatenate([all_controls, tensors.controls[-1][hop_size:, :]], axis=0)
    all_decoded = np.concatenate([all_decoded, prev_decoded[hop_size:, :]], axis=0)

    ns = dc.from_tensors([all_decoded], [all_controls])[0]
    ns.total_time = q.total_time
    return change_tempo(ns, q.tempos[0].qpm)


def normalize(ns):
    s = start_notes_at_0(ns)
    s = change_tempo(s, s.tempos[0].qpm)
    for ts in s.time_signatures[1:]:
        s.time_signatures.remove(ts)
    for t in s.tempos[1:]:
        s.tempos.remove(t)
    return s

def save(seq, folder, filename):
    # Only save the inner 10 seconds, the outer seconds are then used for padding
    new = sequences_lib.extract_subsequence(seq, 5., 15.)
    note_seq.note_sequence_to_midi_file(new, os.path.join(folder, filename))


if os.path.exists(quantize_path):
    shutil.rmtree(quantize_path)
if os.path.exists(humanize_path):
    shutil.rmtree(humanize_path)
if os.path.exists(recording_path):
    shutil.rmtree(recording_path)

files = [file for file in os.listdir(example_path) if file.endswith('.mid')]

os.mkdir(quantize_path)
os.mkdir(humanize_path)
os.mkdir(recording_path)

for file in files:
    print('Processing', file)
    s = note_seq.midi_file_to_note_sequence(os.path.join(example_path, file))
    
    # Take a random 20 second section from the piece, we later cut off the outer 10 seconds to ensure homogeneïty.

    start = random.uniform(0., s.total_time - 20.)    
    s = sequences_lib.extract_subsequence(s, start, start + 40.)
    s.filename = file
    s = normalize(s)    
    save(s, recording_path, file)
    
    print('Quantizing...')
    q, tensors = get_quantized(s, velocity=63, dc=dc)
    q.filename = file
    save(q, quantize_path, file)

    print('Humanizing...')
    h = humanize(q, tensors, model)
    save(h, humanize_path, file)
