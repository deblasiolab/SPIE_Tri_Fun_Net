import keras
from keras import layers, regularizers, ops
from src.constants import SEED

class Sampling(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(SEED)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon
    
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
    z_mean = layers.Dense(units=latent_dim)(x)
    z_log_var = layers.Dense(units=latent_dim)(x)
    z = Sampling()([z_mean, z_log_var])

    # Model
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
    return encoder
