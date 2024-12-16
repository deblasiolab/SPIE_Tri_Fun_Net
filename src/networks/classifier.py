import keras
from keras import layers, regularizers

def define_classifier(latent_dim, num_classes):
    # Input
    classifier_inputs = keras.Input(shape=(latent_dim,))

    # Classifier
    x = layers.Dense(units=32, activation='relu', use_bias=False, kernel_regularizer=regularizers.OrthogonalRegularizer(factor=1.0))(classifier_inputs)

    # Classifier Output
    classifier_outputs = layers.Dense(units=num_classes, activation='softmax')(x)

    # Model
    classifier = keras.Model(classifier_inputs, classifier_outputs, name='classifier')
    return classifier   