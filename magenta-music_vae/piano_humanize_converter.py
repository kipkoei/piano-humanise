
from magenta.models.music_vae.data import BaseNoteSequenceConverter, ConverterTensors

import collections
import copy
import note_seq
from note_seq import sequences_lib
import numpy as np

PIANO_MIN_MIDI_PITCH = 21
PIANO_MAX_MIDI_PITCH = 108
MIN_MIDI_PITCH = 0
MAX_MIDI_PITCH = 127
MIDI_PITCHES = 128

MAX_INSTRUMENT_NUMBER = 127

MEL_PROGRAMS = range(0, 32)  # piano, chromatic percussion, organ, guitar
BASS_PROGRAMS = range(32, 40)
ELECTRIC_BASS_PROGRAM = 33

PIANO_PITCH_CLASSES = [[p] for p in range(PIANO_MIN_MIDI_PITCH, PIANO_MAX_MIDI_PITCH + 1)]
STYLES = ['b', 'c', 'r', 'l', 'm']

OUTPUT_VELOCITY = 80

CHORD_SYMBOL = note_seq.NoteSequence.TextAnnotation.CHORD_SYMBOL

class PianoHumanizeConverter(BaseNoteSequenceConverter):
  """Converts to and from duration/velocity/offset representations.

  In this setting, we represent drum sequences and performances
  as triples of (duration, velocity, offset). Each timestep refers to a fixed beat
  on a grid, which is by default spaced at 16th notes.

  Durations are scalar [0,).
  Velocities are continuous values in [0, 1].
  Offsets are continuous values in [-0.5, 0.5], rescaled to [-1, 1] for tensors.

  Each timestep contains this representation for each of a fixed list of
  piano notes, which by default is the list of the 88 keys on a piano. 
  With the default categories, the input and output
  at a single timestep is of length 88x4 = 264. So a single measure of drums
  at a 16th note grid is a matrix of shape (16, 264).

  Attributes:
    split_bars: Optional size of window to slide over full converted tensor.
    steps_per_quarter: The number of quantization steps per quarter note.
    quarters_per_bar: The number of quarter notes per bar.
    pitch_classes: A collection of collections, with each sub-collection
      containing the set of pitches representing a single class to group by.
    inference_pitch_classes: Pitch classes to use during inference. By default,
      uses same as `pitch_classes`.
    humanize: If True, flatten all input velocities and microtiming. The model
      then learns to map from a flattened input to the original sequence.
    hop_size: Number of steps to slide window.
    hits_as_controls: If True, pass in hits with the conditioning controls
      to force model to learn velocities and offsets.
    fixed_velocities: If True, flatten all input velocities.
    max_note_dropout_probability: If a value is provided, randomly drop out
      notes from the input sequences but not the output sequences.  On a per
      sequence basis, a dropout probability will be chosen uniformly between 0
      and this value such that some sequences will have fewer notes dropped
      out and some will have have more.  On a per note basis, lower velocity
      notes will be dropped out more often.
  """

  def __init__(self, split_bars=None, steps_per_quarter=4, quarters_per_bar=4,
               max_tensors_per_notesequence=8, pitch_classes=None,
               inference_pitch_classes=None, humanize=False, hop_size=None,
               hits_as_controls=False, fixed_velocities=False,
               max_note_dropout_probability=None):

    self._split_bars = split_bars
    self._steps_per_quarter = steps_per_quarter
    self._steps_per_bar = steps_per_quarter * quarters_per_bar

    self._humanize = humanize
    self._fixed_velocities = fixed_velocities

    self._hop_size = hop_size
    self._hits_as_controls = hits_as_controls

    def _classes_to_map(classes):
      class_map = {}
      for cls, pitches in enumerate(classes):
        for pitch in pitches:
          class_map[pitch] = cls
      return class_map

    self._pitch_classes = pitch_classes or PIANO_PITCH_CLASSES
    self._pitch_class_map = _classes_to_map(self._pitch_classes)
    self._infer_pitch_classes = inference_pitch_classes or self._pitch_classes
    self._infer_pitch_class_map = _classes_to_map(self._infer_pitch_classes)
    if len(self._pitch_classes) != len(self._infer_pitch_classes):
      raise ValueError(
          'Training and inference must have the same number of pitch classes. '
          'Got: %d vs %d.' % (
              len(self._pitch_classes), len(self._infer_pitch_classes)))
    self._num_keys = len(self._pitch_classes)

    if split_bars is None and hop_size is not None:
      raise ValueError(
          'Cannot set hop_size without setting split_bars')

    keys_per_output = self._num_keys
    # Output is 4 numbers per key (hit, duration, velocity, offset) and two numbers for the pedal
    output_depth = keys_per_output * 2 + 2
    input_depth = keys_per_output * 2 + len(STYLES)

    control_depth = 0
    # Set up controls for passing hits and durations as side information.
    if self._hits_as_controls:
      control_depth += self._num_keys * 2

    self._max_note_dropout_probability = max_note_dropout_probability
    self._note_dropout = max_note_dropout_probability is not None

    super(PianoHumanizeConverter, self).__init__(
        input_depth=input_depth,
        input_dtype=np.float32,
        output_depth=output_depth,
        output_dtype=np.float32,
        control_depth=control_depth,
        control_dtype=np.bool,
        end_token=False,
        presplit_on_time_changes=False,
        max_tensors_per_notesequence=max_tensors_per_notesequence)

  @property
  def pitch_classes(self):
    if self.is_inferring:
      return self._infer_pitch_classes
    return self._pitch_classes

  @property
  def pitch_class_map(self):  # pylint: disable=g-missing-from-attributes
    if self.is_inferring:
      return self._infer_pitch_class_map
    return self._pitch_class_map

  def _get_feature(self, note, feature, step_length=None):
    """Compute numeric value of hit/duration/velocity/offset for a note.

    For now, only allow one note per instrument per quantization time step.
    This means at 16th note resolution we can't represent some drumrolls etc.
    We just take the note with the highest velocity if there are multiple notes.

    Args:
      note: A Note object from a NoteSequence.
      feature: A string, either 'duration', 'velocity', or 'offset'.
      step_length: Time duration in seconds of a quantized step. This only needs
        to be defined when the feature is 'offset'.

    Raises:
      ValueError: Any feature other than 'duration', 'velocity', or 'offset'.

    Returns:
      The numeric value of the feature for the note.
    """

    def _get_offset(note, step_length):
      true_onset = note.start_time
      quantized_onset = step_length * note.quantized_start_step
      diff = quantized_onset - true_onset
      return diff/step_length

    def _get_duration(note, step_length):
      true_duration = note.end_time - note.start_time 
      return true_duration / step_length
      
      
    if feature == 'hit':
      if note:
        return 1.
      else:
        return 0.

    if feature == 'duration':
      if note:
        return _get_duration(note, step_length)
      else:
        return 0.
      
    elif feature == 'velocity':
      if note:
        return note.velocity/127.  # Switch from [0, 127] to [0, 1] for tensors.
      else:
        return 0.  # Set velocity to 0 if there's no note, this will be filled in later based on surrounding values

    elif feature == 'offset':
      if note:
        offset = _get_offset(note, step_length)
        return offset*2  # Switch from [-0.5, 0.5] to [-1, 1] for tensors.
      else:
        return 0.  # Default offset if there's no note is 0

    else:
      raise ValueError('Unlisted feature: ' + feature)

  def to_tensors(self, item, output_quantized = False, max_step=0):

    def _get_steps_hash(note_sequence):
      """Partitions all Notes in a NoteSequence by quantization step and drum.

      Creates a hash with each hash bucket containing a dictionary
      of all the notes at one time step in the sequence grouped by pitch.
      If there are no hits at a given time step, the hash value will be {}.

      Args:
        note_sequence: The NoteSequence object

      Returns:
        The fully constructed hash

      Raises:
        ValueError: If the sequence is not quantized
      """
      if not note_seq.sequences_lib.is_quantized_sequence(note_sequence):
        raise ValueError('NoteSequence must be quantized')

      h = collections.defaultdict(lambda: collections.defaultdict(list))
      c = collections.defaultdict(list)

      for note in note_sequence.notes:
        step = int(note.quantized_start_step)
        pitch = self.pitch_class_map[note.pitch]
        h[step][pitch].append(note)

      for control in note_sequence.control_changes:
        if control.control_number == 64:
          step = int(control.quantized_step)
          c[step].append(control)
      
      return h, c

    def _extract_windows(tensor, window_size, hop_size):
      """Slide a window across the first dimension of a 2D tensor."""
      return [tensor[i:i+window_size, :] for i in range(0, len(tensor) - window_size + 1, hop_size)]

    note_sequence = item
    style = note_sequence.filename[0]
    quantized_sequence = sequences_lib.quantize_note_sequence(note_sequence, self._steps_per_quarter)
    
    if (note_seq.steps_per_bar_in_quantized_sequence(quantized_sequence) != self._steps_per_bar):
      print("Time signature does not match model.")
      
    if not quantized_sequence.time_signatures:
      quantized_sequence.time_signatures.add(numerator=4, denominator=4)

    beat_length = 60. / quantized_sequence.tempos[0].qpm
    step_length = beat_length / quantized_sequence.quantization_info.steps_per_quarter

    steps_hash, steps_pedal = _get_steps_hash(quantized_sequence)

    if not quantized_sequence.notes:
      print("Warning: No notes in sequence")

    max_start_step = np.max([note.quantized_start_step for note in quantized_sequence.notes])

    # Round up so we pad to the end of the bar.
    total_bars = int(np.ceil((max_start_step + 1) / self._steps_per_bar))
    
    if max_step == 0:
      max_step = self._steps_per_bar * total_bars

    # Each of these stores a (total_beats, num_keys) matrix.
    hit_vectors = np.zeros((max_step, self._num_keys))
    duration_vectors = np.zeros((max_step, self._num_keys))
    velocity_vectors = np.zeros((max_step, self._num_keys))
    offset_vectors = np.zeros((max_step, self._num_keys))
    pedal_vectors = np.zeros((max_step, 2))
    style_vectors = np.zeros((max_step, len(STYLES)))
    
    style_vectors[:,STYLES.index(style)] = 1

    hit_steps = []
    # initialize the pedal to "up"
    pedal_state = 0.

    # Loop through timesteps.    
    for step in range(max_step):
      notes = steps_hash[step]
      
      hits = list()
      # Loop through each key.
      for key in range(self._num_keys):
        key_notes = notes[key]
        if len(key_notes) > 1:
          note = max(key_notes, key=lambda n: n.velocity)
        elif len(key_notes) == 1:
          note = key_notes[0]
        else:
          note = None
        
        hit_vectors[step, key] = self._get_feature(note, 'hit')
        duration_vectors[step, key] = self._get_feature(note, 'duration', step_length)
        velocity_vectors[step, key] = self._get_feature(note, 'velocity')
        offset_vectors[step, key] = self._get_feature(note, 'offset', step_length)
        
        if hit_vectors[step,key]:
          hits.append(key)    
      
      if len(hits) > 0:
        hit_steps.append(step)
      
      prev = 0
      hits.append(self._num_keys - 1)
      
      # Make changes gradual over the keyrange
      for next in hits:
        for key in range(prev + 1, next):
          duration_vectors[step, key] = (
            ((next - key) * duration_vectors[step, prev] + 
             (key - prev) * duration_vectors[step, next]) / 
            (next - prev))
          velocity_vectors[step, key] = (
            ((next - key) * velocity_vectors[step, prev] + 
             (key - prev) * velocity_vectors[step, next]) / 
            (next - prev))
          offset_vectors[step, key] = (
            ((next - key) * offset_vectors[step, prev] + 
             (key - prev) * offset_vectors[step, next]) / 
            (next - prev))
        prev = next
        
      # Add the last pedal action in this step
      for control in steps_pedal[step]:
        pedal_state = control.control_value / 127.
        
        if pedal_state < 0.7:
          pedal_vectors[step, 0] = 1. # Pedal lift
      
      # Set the output state of the pedal
      pedal_vectors[step, 1] = pedal_state
          
    hit_steps.append(max_step - 1)
    
    for key in range(self._num_keys):
      prev = 0
      
      # Make changes gradual over the steps
      for next in hit_steps:
        for step in range(prev + 1, next):
          duration_vectors[step, key] = (
            ((next - step) * duration_vectors[prev, key] + 
             (step - prev) * duration_vectors[next, key]) / 
            (next - prev))
          velocity_vectors[step, key] = (
            ((next - step) * velocity_vectors[prev, key] + 
             (step - prev) * velocity_vectors[next, key]) / 
            (next - prev))
          offset_vectors[step, key] = (
            ((next - step) * offset_vectors[prev, key] + 
             (step - prev) * offset_vectors[next, key]) / 
            (next - prev))
        prev = next
      
          
    # These are the input tensors for the encoder.
    in_hits = copy.deepcopy(hit_vectors)
    in_durations = copy.deepcopy(duration_vectors)
    in_velocities = copy.deepcopy(velocity_vectors)
    in_offsets = copy.deepcopy(offset_vectors)
    in_pedals = copy.deepcopy(pedal_vectors)

    if self._note_dropout:
      # Choose a uniform dropout probability for notes per sequence.
      note_dropout_probability = np.random.uniform(
          0.0, self._max_note_dropout_probability)
      # Drop out lower velocity notes with higher probability.
      velocity_dropout_weights = np.maximum(0.2, (1 - in_velocities))
      note_dropout_keep_mask = 1 - np.random.binomial(
          1, velocity_dropout_weights * note_dropout_probability)
      in_durations *= note_dropout_keep_mask
      in_velocities *= note_dropout_keep_mask
      in_offsets *= note_dropout_keep_mask

    if self._humanize:
      # in_durations[:] = 1.
      in_velocities[:] = 0.5
      in_offsets[:] = 0
      in_pedals[:] = 0

    if self._fixed_velocities:
      in_velocities[:] = 0.5

    # Now concatenate all 3 vectors into 1, eg (16, 27).
    output_seqs = np.concatenate([velocity_vectors, offset_vectors, pedal_vectors], axis=1)
    quantized_seqs = np.concatenate([in_velocities, in_offsets, in_pedals], axis=1)
    input_seqs = np.concatenate([in_hits, in_durations, style_vectors], axis=1)

    # Controls section.
    controls = []
    if self._hits_as_controls:
      controls.append(hit_vectors.astype(np.bool))
      controls.append(duration_vectors)
      
    controls = np.concatenate(controls, axis=-1) if controls else None

    if self._split_bars:
      window_size = self._steps_per_bar * self._split_bars
      hop_size = self._hop_size or window_size
      output_seqs = _extract_windows(output_seqs, window_size, hop_size)
      input_seqs = _extract_windows(input_seqs, window_size, hop_size)
      if controls is not None:
        controls = _extract_windows(controls, window_size, hop_size)
      if output_quantized:
        quantized_seqs = _extract_windows(quantized_seqs, window_size, hop_size)
    else:
      # Output shape will look like (1, 64, output_depth).
      output_seqs = [output_seqs]
      input_seqs = [input_seqs]
      if controls is not None:
        controls = [controls]

    tensors = ConverterTensors(inputs=input_seqs, outputs=output_seqs, controls=controls)
    if output_quantized:
      return tensors, quantized_seqs
    
    return tensors

  def from_tensors(self, samples, controls=None):
    def _zero_one_to_velocity(val):
      output = int(np.round(val*127))
      return np.clip(output, 0, 127)

    def _minus_1_1_to_offset(val):
      output = val/2
      return np.clip(output, -0.5, 0.5)

    output_sequences = []
    use_controls = controls is not None
    
    for sample, control in zip(samples, controls):
      n_timesteps = sample.shape[0]
      note_sequence = note_seq.NoteSequence()
      note_sequence.tempos.add(qpm=120)
      beat_length = 60. / note_sequence.tempos[0].qpm
      step_length = beat_length / self._steps_per_quarter

      # Each timestep should be a (1, output_depth) vector
      # representing n hits, durations, velocities, offsets, and pedal controls in order.
      pitch_range = range(len(self.pitch_classes))
      
      for i in range(n_timesteps):
        velocities, offsets, pedal = np.split(sample[i], [self._num_keys, self._num_keys*2])
      
        control_hits, control_durations =  np.split(control[i], 2)
        # Loop through the pitches
        for j in pitch_range:
          if (use_controls and control_hits[j]):# or (not use_controls and hits[j] > 0.5):
            # Create a new note
            note = note_sequence.notes.add()
            pitch = self.pitch_classes[j][0]
            note.pitch = pitch
            note.velocity = _zero_one_to_velocity(velocities[j])
            offset = _minus_1_1_to_offset(offsets[j])
            note.start_time = (i - offset) * step_length
            duration = control_durations[j] #if use_controls else durations[j]
            note.end_time = note.start_time + step_length * duration
            
        if pedal[0] > 0.5: # Pedal lift
          control_change = note_sequence.control_changes.add()
          # Make sure the pedal lift occurs before pressing down again
          control_change.time = i * step_length - step_length / 4
          control_change.control_value = 0
          control_change.control_number = 64
        
        control_change = note_sequence.control_changes.add()
        control_change.time = i * step_length
        control_change.control_value = _zero_one_to_velocity(pedal[1])
        control_change.control_number = 64

      output_sequences.append(note_sequence)

    return output_sequences

