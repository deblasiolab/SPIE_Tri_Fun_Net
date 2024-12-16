import keras

class ZLA_EncoderWrapper(keras.layers.Layer):
    def __init__(self, encoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder

    def call(self, inputs):
        _, _, z_latent = self.encoder(inputs)
        return z_latent

class ZME_EncoderWrapper(keras.layers.Layer):
    def __init__(self, encoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder

    def call(self, inputs):
        z_mean, _, _ = self.encoder(inputs)
        return z_mean