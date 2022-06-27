from visual_midi import Plotter, Coloring, Preset
import note_seq
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# This is a supporting script to generate visualisations. It requires numpy arrays of the model inputs/outputs saved as .csv to be present. 

sample_path = "examples/piano/m_balakirev_rec.mid"

h = np.flip(np.flip(np.rot90(pd.read_csv("hits.csv").values[:,1:]), axis=0),axis=0)
d = np.flip(np.flip(np.rot90(pd.read_csv("durations.csv").values[:,1:]), axis=0),axis=0)
o = np.flip(np.flip(np.rot90(pd.read_csv("offsets.csv").values[:,1:]), axis=0),axis=0)
v = np.flip(np.flip(np.rot90(pd.read_csv("velocities.csv").values[:,1:]), axis=0),axis=0)
p = np.rot90(pd.read_csv("pedals.csv").values[1:])


sns.set(rc = {'figure.figsize':(2.02,3.8)})
sns.set(font_scale=1.2)


sns.heatmap(h, annot=True, fmt=".2", vmin=0, vmax=1, linewidths=.5, xticklabels=1, yticklabels=False)
plt.savefig('graphics/vectors/h.png', format='png')
plt.show()

sns.heatmap(d, annot=True, fmt=".2", vmin=0.18, vmax=0.42, linewidths=.5, xticklabels=1, yticklabels=False)
plt.savefig('graphics/vectors/d.png', format='png')
plt.show()
d[h == 0] = 0
sns.heatmap(d, annot=True, fmt=".2", vmin=0.18, vmax=0.42, linewidths=.5, xticklabels=1, yticklabels=False)
plt.savefig('graphics/vectors/d-masked.png', format='png')
plt.show()

sns.heatmap(v, annot=True, fmt=".2", vmin=0.4, vmax=0.8, linewidths=.5, xticklabels=1, yticklabels=False)
plt.savefig('graphics/vectors/v.png', format='png')
plt.show()
v[h == 0] = 0
sns.heatmap(v, annot=True, fmt=".2", vmin=0.4, vmax=0.8, linewidths=.5, xticklabels=1, yticklabels=False)
plt.savefig('graphics/vectors/v-masked.png', format='png')
plt.show()

sns.heatmap(o, annot=True, fmt=".2", center=0, vmin=-1, vmax=1, linewidths=.5, xticklabels=1, yticklabels=False)
plt.savefig('graphics/vectors/o.png', format='png')
plt.show()
o[h == 0] = 0
sns.heatmap(o, annot=True, fmt=".2", center=0, vmin=-1, vmax=1, linewidths=.5, xticklabels=1, yticklabels=False)
plt.savefig('graphics/vectors/o-masked.png', format='png')
plt.show()

def offset_notes(s, offset):
    for n in s.notes:
        n.end_time += offset
        n.start_time += offset
    return s

def stretch_notes(s, factor):
    for n in s.notes:
        n.end_time *= factor
        n.start_time *= factor
    return s

def trim_notes(s, lower_bound):
    for n in s.notes:
        if n.pitch < lower_bound:
            s.notes.remove(n)
    return s


s = note_seq.midi_file_to_note_sequence(sample_path)

stretch = 1
s = stretch_notes(s, stretch)

offset = 0
s = offset_notes(s, offset)

pm = note_seq.note_sequence_to_pretty_midi(s) 

preset = Preset(plot_width=400 , row_height=40, axis_y_major_tick_out=0, axis_y_label_standoff=5)
plotter = Plotter(preset, coloring=Coloring.INSTRUMENT, show_velocity=True,
                  plot_pitch_range_start=65, plot_pitch_range_stop=71,
                  plot_bar_range_start=3, plot_bar_range_stop=6)

plotter.show(pm, "midi_visual.html")
