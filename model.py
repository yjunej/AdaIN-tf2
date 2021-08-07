import tensorflow as tf
from tensorflow.python.keras.layers.convolutional import UpSampling2D
from config import Config


class AdaINStyleTransfer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.vgg19 = tf.keras.applications.vgg19.VGG19(
            weights="imagenet", include_top=False, input_shape=(Config.img_size)
        )
        self.vgg19.trainable = False
        self.encoder = tf.keras.Sequential(self.vgg19.layers[0:13])
        self.input_layer = self.vgg19.layers[0]
        self.encoder_conv_relu_1_1 = tf.keras.Sequential(self.vgg19.layers[0:2])
        self.encoder_conv_relu_2_1 = tf.keras.Sequential(self.vgg19.layers[2:5])
        self.encoder_conv_relu_3_1 = tf.keras.Sequential(self.vgg19.layers[5:8])
        self.encoder_conv_relu_4_1 = tf.keras.Sequential(self.vgg19.layers[8:13])

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2DTranspose(
                    256, (3, 3), (1, 1), padding="same", activation="relu"
                ),
                tf.keras.layers.UpSampling2D(size=(2, 2)),
                tf.keras.layers.Conv2DTranspose(
                    256, (3, 3), (1, 1), padding="same", activation="relu"
                ),
                tf.keras.layers.Conv2DTranspose(
                    256, (3, 3), (1, 1), padding="same", activation="relu"
                ),
                tf.keras.layers.Conv2DTranspose(
                    256, (3, 3), (1, 1), padding="same", activation="relu"
                ),
                tf.keras.layers.Conv2DTranspose(
                    128, (3, 3), (1, 1), padding="same", activation="relu"
                ),
                tf.keras.layers.UpSampling2D(size=(2, 2)),
                tf.keras.layers.Conv2DTranspose(
                    128, (3, 3), (1, 1), padding="same", activation="relu"
                ),
                tf.keras.layers.Conv2DTranspose(
                    64, (3, 3), (1, 1), padding="same", activation="relu"
                ),
                tf.keras.layers.UpSampling2D(size=(2, 2)),
                tf.keras.layers.Conv2DTranspose(
                    64, (3, 3), (1, 1), padding="same", activation="relu"
                ),
                tf.keras.layers.Conv2DTranspose(
                    3, (3, 3), (1, 1), padding="same", activation="relu"
                ),
            ]
        )

    def call(self, inputs):

        # STYLE TRANSFER NETWORK
        content, style = inputs
        c1 = self.encoder_conv_relu_1_1(content)
        c2 = self.encoder_conv_relu_2_1(c1)
        c3 = self.encoder_conv_relu_3_1(c2)
        c4 = self.encoder_conv_relu_4_1(c3)

        s1 = self.encoder_conv_relu_1_1(style)
        s2 = self.encoder_conv_relu_2_1(s1)
        s3 = self.encoder_conv_relu_3_1(s2)
        s4 = self.encoder_conv_relu_4_1(s3)

        t = adain(c4, s4)

        generated_img = self.decoder(t)

        g1 = self.encoder_conv_relu_1_1(generated_img)
        g2 = self.encoder_conv_relu_2_1(g1)
        g3 = self.encoder_conv_relu_3_1(g2)
        g4 = self.encoder_conv_relu_4_1(g3)

        return t, generated_img, (s1, s2, s3, s4), (g1, g2, g3, g4)


def adain(content, style):
    c_mean, c_var = tf.nn.moments(content, axes=[1, 2], keepdims=True)
    s_mean, s_var = tf.nn.moments(style, axes=[1, 2], keepdims=True)
    return (
        tf.multiply(
            tf.divide((content - c_mean), tf.sqrt(c_var + 1e-5)), tf.sqrt(s_var + 1e-5)
        )
        + s_mean
    )

