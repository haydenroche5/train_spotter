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
            conv_outputs, predictions = self.grad_model(np.array([img]))
            score = predictions[:, 0]

        output = conv_outputs[0]
        grads = tape.gradient(score, conv_outputs)[0]

        gate_f = tf.cast(output > 0, 'float32')
        gate_r = tf.cast(grads > 0, 'float32')
        guided_grads = tf.cast(output > 0, 'float32') * tf.cast(
            grads > 0, 'float32') * grads

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))

        cam = np.ones(output.shape[0:2], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        height, width = img.shape[0:2]
        cam = cv2.resize(cam.numpy(), (width, height))
        cam = np.maximum(cam, 0)
        heatmap = (cam - cam.min()) / (cam.max() - cam.min())

        cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

        output_image_bgr = cv2.addWeighted(
            cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 1,
            0)
        output_image_rgb = output_image_bgr[:, :, ::-1]

        return output_image_rgb
