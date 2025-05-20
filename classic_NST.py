import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19

def classical_neural_style_transfer(base_image_path, style_reference_image_path, result_prefix="generated_image", iterations=5000):
    """
    Perform classic neural style transfer.

    Parameters:
    - base_image_path (str): Path to the content image.
    - style_reference_image_path (str): Path to the style image.
    - result_prefix (str): Prefix for saved image filenames.
    - iterations (int): Number of optimization steps.

    Returns:
    - final_image (np.array): Stylized output image.
    """

    # Load image size
    width, height = keras.preprocessing.image.load_img(base_image_path).size
    img_nrows = 400
    img_ncols = int(width * img_nrows / height)

    # Loss weights
    total_variation_weight = 1e-6
    style_weight = 1e-6
    content_weight = 2.5e-8

    def preprocess_image(image_path):
        img = keras.preprocessing.image.load_img(image_path, target_size=(img_nrows, img_ncols))
        img = keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)
        return tf.convert_to_tensor(img)

    def deprocess_image(x):
        x = x.reshape((img_nrows, img_ncols, 3))
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype("uint8")
        return x

    def gram_matrix(x):
        x = tf.transpose(x, (2, 0, 1))
        features = tf.reshape(x, (tf.shape(x)[0], -1))
        return tf.matmul(features, tf.transpose(features))

    def style_loss(style, combination):
        S = gram_matrix(style)
        C = gram_matrix(combination)
        channels = 3
        size = img_nrows * img_ncols
        return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

    def content_loss(base, combination):
        return tf.reduce_sum(tf.square(combination - base))

    def total_variation_loss(x):
        a = tf.square(x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, 1:, : img_ncols - 1, :])
        b = tf.square(x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :])
        return tf.reduce_sum(tf.pow(a + b, 1.25))

    # Load model
    model = vgg19.VGG19(weights="imagenet", include_top=False)
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

    # Loss layer settings
    style_layer_names = [
        "block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1",
    ]
    content_layer_name = "block5_conv2"

    def compute_loss(combination_image, base_image, style_reference_image):
        input_tensor = tf.concat([base_image, style_reference_image, combination_image], axis=0)
        features = feature_extractor(input_tensor)
        loss = tf.zeros(shape=())
        layer_features = features[content_layer_name]
        base_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss += content_weight * content_loss(base_image_features, combination_features)
        for layer_name in style_layer_names:
            layer_features = features[layer_name]
            style_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            loss += (style_weight / len(style_layer_names)) * style_loss(style_features, combination_features)
        loss += total_variation_weight * total_variation_loss(combination_image)
        return loss

    @tf.function
    def compute_loss_and_grads(combination_image, base_image, style_reference_image):
        with tf.GradientTape() as tape:
            loss = compute_loss(combination_image, base_image, style_reference_image)
        grads = tape.gradient(loss, combination_image)
        return loss, grads

    # Prepare images
    base_image = preprocess_image(base_image_path)
    style_reference_image = preprocess_image(style_reference_image_path)
    combination_image = tf.Variable(preprocess_image(base_image_path))

    optimizer = keras.optimizers.SGD(
        keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
        )
    )

    print("Starting neural style transfer...")
    for i in range(1, iterations + 1):
        loss, grads = compute_loss_and_grads(combination_image, base_image, style_reference_image)
        optimizer.apply_gradients([(grads, combination_image)])

        if i % 1000 == 0:
            print(f"Iteration {i}: loss={loss:.2f}")
            img = deprocess_image(combination_image.numpy())
            fname = f"{result_prefix}_at_iteration_{i}.png"
            keras.preprocessing.image.save_img(fname, img)
            print(f"Saved image at: {fname}")

    # Final image
    final_image = deprocess_image(combination_image.numpy())
    final_fname = f"{result_prefix}_final.png"
    keras.preprocessing.image.save_img(final_fname, final_image)
    print(f"Final stylized image saved at: {final_fname}")

    return final_image