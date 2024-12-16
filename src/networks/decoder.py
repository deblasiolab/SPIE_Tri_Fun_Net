import keras
from keras import layers, regularizers

def define_decoder(latent_dim):
    # Input
    decoder_inputs = keras.Input(shape=(latent_dim,))

    # Decoder
    x = layers.Dense(units=960, activation='relu', use_bias=False, kernel_regularizer=regularizers.OrthogonalRegularizer(factor=1.0))(decoder_inputs)
    x = layers.Dense(units=128, activation='relu')(x)
    x = layers.Dense(units=20*134*16, activation='relu')(x)
    x = layers.Reshape(target_shape=(20, 134, 16))(x)
    x = layers.UpSampling2D(size=(10, 10), interpolation='nearest')(x)
    x = layers.Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)

    # Decoder Output
    decoder_outputs = layers.Conv2D(filters=1, kernel_size=(3, 3), strides=1, padding='same', activation='sigmoid')(x)

    # Model
    decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')
    return decoder