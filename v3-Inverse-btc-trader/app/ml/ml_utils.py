# File: v2-Inverse-btc-trader/app/ml/ml_utils.py

"""
Utility functions and classes for ML models:
- Focal loss
- Custom Attention layer
- Possibly other small utilities
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
from structlog import get_logger

logger = get_logger(__name__)

def focal_loss(gamma: float=2.0, alpha: float=0.25):
    """
    Focal loss to address class imbalance.

    Args:
        gamma (float): Focusing parameter for modulating factor (1-p).
        alpha (float): Weighting factor for the rare class.

    Returns:
        A focal loss function closure.
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1e-8
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return focal_loss_fixed

class AttentionLayer(Layer):
    """
    Custom attention layer for sequence data.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="normal"
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros"
        )
        super().build(input_shape)
    
    def call(self, x):
        # x has shape [batch_size, time_steps, features]
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)  # [batch_size, time_steps, 1]
        a = tf.nn.softmax(e, axis=1)  # [batch_size, time_steps, 1]
        output = tf.reduce_sum(x * a, axis=1)  # [batch_size, features]
        return output
