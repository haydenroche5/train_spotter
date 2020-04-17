import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import numpy as np
import cv2


class GradCamHeatMapper:
    def __init__(self, ref_model_dir):
        ref_model = load_model(ref_model_dir)
        last_conv_layer_name = ''
        for layer in ref_model.layers:
            config = layer.get_config()
            if config['name'].startswith('conv2d'):
                last_conv_layer_name = config['name']

        self.grad_model = Model([ref_model.inputs], [
            ref_model.get_layer(last_conv_layer_name).output, ref_model.output
        ])
        self.grad_model.layers[-1].activation = None

    def get_heat_map(self, img):
        with tf.GradientTape() as tape:
            inputs = tf.cast(img, tf.float32)
            conv_outputs, predictions = self.grad_model(np.array([img]))
            score = predictions[:, 0]

        grads = tape.gradient(score, conv_outputs)[0]

        cast_conv_outputs = tf.cast(conv_outputs > 0, 'float32')
        cast_grads = tf.cast(grads > 0, 'float32')
        guided_grads = cast_conv_outputs * cast_grads * grads

        conv_outputs = conv_outputs[0]
        guided_grads = guided_grads[0]

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)

        (w, h) = (img.shape[1], img.shape[0])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        epsilon = 1e-8
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + epsilon
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype('uint8')

        img = (img * 255).astype('uint8')

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        output = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

        return (heatmap, output)
