import tensorflow as tf

def main(input_dim=256, dropout_rate=0.2):
    input_layer = tf.keras.layers.Input(shape=(input_dim,), name='input_vector')

    # Encoder stem
    x = tf.keras.layers.Dense(input_dim, name='encoder_dense_stem')(input_layer)
    x = tf.keras.layers.BatchNormalization(name='encoder_bn_stem')(x)
    x = tf.keras.layers.Activation('relu', name='encoder_act_stem')(x)
    x = tf.keras.layers.Dropout(dropout_rate, name='encoder_drop_stem')(x)

    # Bottleneck (latent space) - increased capacity
    latent_units = max(32, input_dim // 4)
    latent_space = tf.keras.layers.Dense(latent_units, activation='relu', name='latent_space')(x)

    # Decoder stem
    decoded = tf.keras.layers.Dense(max(32, input_dim // 2), name='decoder_dense_stem')(latent_space)
    decoded = tf.keras.layers.BatchNormalization(name='decoder_bn_stem')(decoded)
    decoded = tf.keras.layers.Activation('relu', name='decoder_act_stem')(decoded)
    decoded = tf.keras.layers.Dropout(dropout_rate, name='decoder_drop_stem')(decoded)

    # Output layer (mapping to centroid space)
    output_reconstruction = tf.keras.layers.Dense(input_dim, activation='linear', name='output_centroid')(decoded)

    # Optional classification output for better separability
    return tf.keras.models.Model(inputs=input_layer, outputs=output_reconstruction, name='cluster_autoencoder_deep')