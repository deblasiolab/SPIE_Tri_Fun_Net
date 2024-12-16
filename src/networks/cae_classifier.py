import tensorflow as tf
import keras

class CAE_CLASSIFIER(keras.Model):
    def __init__(self, encoder, decoder, classifier, **kwargs):
        super().__init__(**kwargs)
        # Sub-networks
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier

        # Initialize trackers
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.classification_loss_tracker = keras.metrics.Mean(name='classification_loss')

        # Initialize accuracy
        self.accuracy_metric = keras.metrics.SparseCategoricalAccuracy(name='accuracy')

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.classification_loss_tracker,
            self.accuracy_metric,
        ]

    def call(self, inputs):
        # Forward pass through the model
        latent_z = self.encoder(inputs)
        reconstruction = self.decoder(latent_z)
        classification = self.classifier(latent_z)
        return (reconstruction, classification)

    def train_step(self, data):
        # Unpack data
        x_imgs, y_labels = data

        with tf.GradientTape() as tape:
            # Forward pass
            latent_z = self.encoder(x_imgs)
            reconstruction = self.decoder(latent_z)
            classification = self.classifier(latent_z)

            # Reconstruction loss (MAE)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_mean(tf.abs(x_imgs - reconstruction), axis=[1, 2])
            )

            # Classification loss (Cross-Entropy)
            classification_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y_labels, classification)
            )

            # Combine losses
            total_loss = reconstruction_loss + classification_loss

            # Calculate accuracy
            self.accuracy_metric.update_state(y_labels, classification)

        # Backpropagation
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.classification_loss_tracker.update_state(classification_loss)

        # Return metrics
        return {
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'classification_loss': self.classification_loss_tracker.result(),
            'accuracy': self.accuracy_metric.result(),
        }

    def test_step(self, data):
        # Unpack data
        x_imgs, y_labels = data

        # Forward pass
        latent_z = self.encoder(x_imgs)
        reconstruction = self.decoder(latent_z)
        classification = self.classifier(latent_z)

        # Reconstruction loss (MAE)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_mean(tf.abs(x_imgs - reconstruction), axis=[1, 2])
        )

        # Classification loss (Cross-Entropy)
        classification_loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y_labels, classification)
        )

        # Combine losses
        total_loss = reconstruction_loss + classification_loss

        # Calculate accuracy
        self.accuracy_metric.update_state(y_labels, classification)

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.classification_loss_tracker.update_state(classification_loss)

        # Return metrics
        return {
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'classification_loss': self.classification_loss_tracker.result(),
            'accuracy': self.accuracy_metric.result(),
        }