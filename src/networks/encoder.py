import keras
from keras import layers, regularizers

def define_encoder(latent_dim, input_shape):
    # Input
    encoder_inputs = keras.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(encoder_inputs)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(10, 10))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation='relu')(x)
    x = layers.Dense(units=960, activation='relu', use_bias=False, kernel_regularizer=regularizers.OrthogonalRegularizer(factor=1.0))(x)

    # Encoder Output
    encoder_outputs = layers.Dense(units=latent_dim, activation='relu', use_bias=False, kernel_regularizer=regularizers.OrthogonalRegularizer(factor=1.0))(x)

    # Model
    encoder = keras.Model(encoder_inputs, encoder_outputs, name='encoder')
    return encoder
