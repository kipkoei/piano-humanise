from magenta.models.music_vae.lstm_models import BaseLstmDecoder
import tensorflow_probability as tfp
import tensorflow.compat.v1 as tf

class PianoLstmDecoder(BaseLstmDecoder):
  """Piano LSTM decoder with MSE loss for continuous values.

  At each timestep, this decoder outputs a vector of length (N_INSTRUMENTS*3).

  For each instrument, the model outputs a triple of (on/off, velocity, offset),
  with a binary representation for on/off, continuous values between 0 and 1
  for velocity, and continuous values between -0.5 and 0.5 for offset.
  """

  def _activate_outputs(self, flat_rnn_output):
    output_velocities, output_offsets, output_pedal = tf.split(
        flat_rnn_output, [88, 88, 2], axis=1)

    output_velocities = tf.nn.sigmoid(output_velocities)
    output_offsets = tf.nn.tanh(output_offsets)
    output_pedal = tf.nn.sigmoid(output_pedal)

    return output_velocities, output_offsets, output_pedal

  def _flat_reconstruction_loss(self, flat_x_target, flat_rnn_output):
    target_velocities, target_offsets, target_pedal = tf.split(
        flat_x_target, [88, 88, 2], axis=1)

    output_velocities, output_offsets, output_pedal = self._activate_outputs(flat_rnn_output)

    velocities_loss = tf.reduce_sum(tf.losses.mean_squared_error(
        target_velocities, output_velocities,
        reduction=tf.losses.Reduction.NONE), axis=1)

    offsets_loss = tf.reduce_sum(tf.losses.mean_squared_error(
        target_offsets, output_offsets,
        reduction=tf.losses.Reduction.NONE), axis=1)
    
    pedal_loss = tf.reduce_sum(tf.losses.mean_squared_error(
        target_pedal, output_pedal,
        reduction=tf.losses.Reduction.NONE), axis=1)

    # loss = hits_loss + durations_loss + 
    loss = velocities_loss + offsets_loss + pedal_loss

    metric_map = {
        'metrics/velocities_loss':
            tf.metrics.mean(velocities_loss),
        'metrics/offsets_loss':
            tf.metrics.mean(offsets_loss),
        'metrics/pedal_loss':
            tf.metrics.mean(pedal_loss)
    }

    return loss, metric_map

  def _sample(self, rnn_output, temperature=1.0):
    output_velocities, output_offsets, output_pedal = tf.split(
        rnn_output, [88, 88, 2], axis=1)

    output_velocities = tf.nn.sigmoid(output_velocities)
    output_offsets = tf.nn.tanh(output_offsets)
    output_pedal = tf.nn.sigmoid(output_pedal)    
    
    return tf.concat([output_velocities, output_offsets, output_pedal], axis=1)
