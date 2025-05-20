import gradio as gr
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

# Check available GPU
gpus = tf.config.list_physical_devices('GPU')
gpu_available = len(gpus) > 0

# Load model from TF Hub
model_path = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
print("Loading model from TensorFlow Hub...")
hub_module = hub.load(model_path)

# Preprocess image
def preprocess_image(image):
    image = np.array(image).astype(np.float32)[np.newaxis, ...] / 255.0
    return image

# Style Transfer Function with GPU Option
def transfer_style_gradio(content_image, style_image, use_gpu):
    content_image = preprocess_image(content_image)
    style_image = preprocess_image(style_image)
    style_image = tf.image.resize(style_image, (256, 256))

    # Device context
    if use_gpu and gpu_available:
        device = '/GPU:0'
    else:
        device = '/CPU:0'

    print(f"Running on: {device}")

    with tf.device(device):
        outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
        stylized_image = outputs[0]

    # Convert output to image
    stylized_image = tf.squeeze(stylized_image, axis=0)
    stylized_image = (stylized_image * 255).numpy().astype(np.uint8)

    return Image.fromarray(stylized_image)

# Build Gradio Interface
gr.Interface(
    fn=transfer_style_gradio,
    inputs=[
        gr.Image(type="pil", label="Content Image"),
        gr.Image(type="pil", label="Style Image"),
        gr.Checkbox(label="Use GPU (if available)", value=True)
    ],
    outputs=gr.Image(label="Stylized Output"),
    title="Neural Style Transfer",
    description="Upload a content and style image",
    allow_flagging="never"
).launch()