import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow INFO & WARNING logs

import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress deprecation and user warnings

import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Hide TensorFlow internal logs

import numpy as np
import tensorflow_hub as hub
import matplotlib.pyplot as plt

def fast_neural_style_transfer(content_image_path, style_image_path, save_path=None, show_plot=True):
    """
    Applies fast neural style transfer using a pre-trained TF Hub model.

    Parameters:
    - content_image_path (str): Path to the content image.
    - style_image_path (str): Path to the style image.
    - save_path (str): File path to save the stylized image.
    - show_plot (bool): Whether to display the comparison plot.

    Returns:
    - stylized_image (np.array): The generated stylized image.
    """

    # Load and preprocess images
    content_image = plt.imread(content_image_path).astype(np.float32)[np.newaxis, ...] / 255.
    style_image = plt.imread(style_image_path).astype(np.float32)[np.newaxis, ...] / 255.
    style_image = tf.image.resize(style_image, (256, 256))

    # For visualization
    content_image_display = plt.imread(content_image_path)
    style_image_display = plt.imread(style_image_path)

    # Load style transfer model from TF Hub
    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    # Apply style transfer
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0][0].numpy()

    if show_plot:
        plt.figure(figsize=(25, 30))

        plt.subplot(1, 3, 1)
        plt.imshow(content_image_display)
        plt.title("Content Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(style_image_display)
        plt.title("Style Image")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(stylized_image)
        plt.title("Stylized Image")
        plt.axis("off")

        if save_path:
            plt.savefig(save_path, transparent=True, bbox_inches='tight')
            print(f"Stylized image saved to: {save_path}")
        plt.show()
    elif save_path:
        # Save without showing
        plt.imsave(save_path, stylized_image)
        print(f"Stylized image saved to: {save_path}")

    return stylized_image
