import os
import soundfile as sf
from note_seq import midi_synth, midi_file_to_note_sequence
from pydub import AudioSegment

# This script produces audio samples from MIDI files. See the fluidsynth install guide for prerequisites.

example_path = 'examples'

quantize_path = os.path.join(example_path, 'quantized')
humanize_path = os.path.join(example_path, 'humanized')
recording_path = os.path.join(example_path, 'recording')

for file in os.listdir(example_path):
    if os.path.isdir(os.path.join(example_path, file)):
        continue

    print(file)
    for folder in [quantize_path, humanize_path, recording_path]:
        filepath = os.path.join(folder, file)
        wavpath = filepath + '.wav'
        mp3path = filepath + '.mp3'
        note_sequence = midi_file_to_note_sequence(filepath)

        audio_seq = midi_synth.fluidsynth(note_sequence, sample_rate=44100, sf2_path="4U-Yamaha C5 Grand-v1.6.SF2")
        sf.write(wavpath, audio_seq, 44100)
        segment = AudioSegment.from_wav(wavpath)
        segment.export(mp3path, format="mp3")
        
        
