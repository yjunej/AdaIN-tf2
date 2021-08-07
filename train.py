from model import AdaINStyleTransfer, adain
from config import Config
import tensorflow as tf
import matplotlib.image as mpimg
import cv2
import PIL.Image as pimg


def train():
    content = mpimg.imread(Config.content_img_path)
    style = mpimg.imread(Config.content_style_path)

    content = tf.image.resize(tf.constant(content), [512, 512])
    style = tf.image.resize(tf.constant(style), [512, 512])
    content = content[tf.newaxis, ...] / 255.0
    style = style[tf.newaxis, ...] / 255.0

    #! Normalizing
    # style = style / 255

    model = AdaINStyleTransfer()
    optimizer = tf.keras.optimizers.Adam()

    l2_loss = tf.keras.losses.MeanSquaredError()

    for e in range(Config.epoch):
        crop_content = tf.image.random_crop(content[0], size=[256, 256, 3])
        crop_style = tf.image.random_crop(style[0], size=[256, 256, 3])

        crop_content = crop_content[tf.newaxis, ...]
        crop_style = crop_style[tf.newaxis, ...]

        with tf.GradientTape() as tape:
            t, generated_img, c_feature_maps, s_feature_maps = model(
                [crop_content, crop_style]
            )
            content_loss = l2_loss(s_feature_maps[-1], t)

            style_loss = 0
            for c_f, s_f in zip(c_feature_maps, s_feature_maps):
                c_mean, c_var = tf.nn.moments(c_f, axes=[1, 2])
                s_mean, s_var = tf.nn.moments(s_f, axes=[1, 2])
                s_loss = l2_loss(c_mean, s_mean) + l2_loss(
                    tf.sqrt(c_var), tf.sqrt(s_var)
                )
                style_loss += s_loss
            loss = content_loss + style_loss
        print(content_loss.numpy(), style_loss.numpy())

        if e % 10 == 0:
            _, gen_img, _, _ = model([content, style], training=False)
            cv2.imwrite(f"ys_results/{e}_epoch_gen_img.png", gen_img.numpy()[0])
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


if __name__ == "__main__":
    train()
