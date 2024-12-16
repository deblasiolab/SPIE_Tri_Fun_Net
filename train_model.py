import json
import os
import argparse
import numpy as np
import keras
from keras.models import Sequential
from src.utils.numpy_data_load import NumpyDataLoader
from src.utils.visualize_image_dimensions import visualize_image_dimensions
from src.utils.slitless_preprocessing import SlitlessPreprocessor
from src.utils.string_label_encoder import StringLabelEncoder
from src.utils.visualize_class_samples import visualize_class_samples
from src.utils.analyze_data_splits import analyze_data_splits
from src.networks.encoder import define_encoder
from src.networks.decoder import define_decoder
from src.networks.classifier import define_classifier
from src.networks.cae_classifier import CAE_CLASSIFIER
from src.utils.plot_training_history import plot_training_history
from src.utils.classification_performance_summary import classification_performance_summary
from src.utils.visualize_latent_space_2d import visualize_latent_space_2d
from src.constants import LATENT_DIM, BATCH_SIZE, LEARNING_RATE, EPOCHS

# Set up argument parser
parser = argparse.ArgumentParser(description='Train model with specified fold')
parser.add_argument('fold', type=int, help='Fold index for cross-validation')
args = parser.parse_args()

# Define model directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  
CAE_MODEL_DIR = os.path.join(ROOT_DIR, 'trained_models', f'SPIE_TrifunNet_fold_{args.fold}/')
os.makedirs(CAE_MODEL_DIR, exist_ok=True)

# data
# Load data
print("loading fold: ", args.fold)
data_loader = NumpyDataLoader(fold_idx=args.fold)
x_train, y_train = data_loader.load_training_data()
x_val, y_val = data_loader.load_validation_data()
x_test, y_test = data_loader.load_testing_data()
print(x_train.shape, x_val.shape, x_test.shape)
print(y_train.shape, y_val.shape, y_test.shape)
print(np.unique(y_train))

visualize_image_dimensions(x_train, save_path=os.path.join(CAE_MODEL_DIR, 'mean_slit_dimensions.png'))

analyze_data_splits(x_train, y_train, x_val, y_val, x_test, y_test, save_path=os.path.join(CAE_MODEL_DIR, 'data_splits_fold.png'))

# example slitless-spectographs:
selected_indices = visualize_class_samples(x_train, y_train, save_path=os.path.join(CAE_MODEL_DIR, 'example_slitless_spectographs_fold.png'))

# Preprocessing Slitless
# Preprocess data
x_preprocessor = SlitlessPreprocessor().fit(x_train)
x_train = x_preprocessor.transform(x_train)
x_val = x_preprocessor.transform(x_val)
x_test = x_preprocessor.transform(x_test)

selected_indices = visualize_class_samples(x_train, y_train, selected_indices_arr=selected_indices)

# Encode labels
y_encoder = StringLabelEncoder().fit(y_train)
y_train = y_encoder.transform(y_train)
y_val = y_encoder.transform(y_val)
y_test = y_encoder.transform(y_test)

# ADD Channel Dimension
x_train = np.expand_dims(x_train, axis=-1)
x_val = np.expand_dims(x_val, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Model sub-networks
encoder = define_encoder(latent_dim=LATENT_DIM, input_shape=x_train.shape[1:])
decoder = define_decoder(latent_dim=LATENT_DIM)
classifier = define_classifier(latent_dim=LATENT_DIM, num_classes=np.unique(y_train).shape[0])

# Define model
cae_classifier = CAE_CLASSIFIER(encoder, decoder, classifier)
cae_classifier.compile(optimizer=keras.optimizers.SGD(learning_rate=LEARNING_RATE))

# Save best on validation
checkpoint_callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(CAE_MODEL_DIR, 'best_val_accuracy.weights.h5'),
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_weights_only=True,
        verbose=0
    ),
]

# Train model
history = cae_classifier.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_val, y_val),
    callbacks=checkpoint_callbacks,
    verbose=1
)

# Convert history.history to a JSON-serializable format
history_dict = {key: [float(x) for x in value] for key, value in history.history.items()}

# Save the serialized history to a JSON file
with open(os.path.join(CAE_MODEL_DIR, 'training_history.json'), 'w') as f:
    json.dump(history_dict, f)

# Load the JSON history later
with open(os.path.join(CAE_MODEL_DIR, 'training_history.json'), 'r') as f:
    loaded_history = json.load(f)

plot_training_history(loaded_history, save_path=os.path.join(CAE_MODEL_DIR, 'training_history_fold.png'))
                      
# Load the best model by validation
encoder = define_encoder(latent_dim=LATENT_DIM, input_shape=x_train.shape[1:])
decoder = define_decoder(latent_dim=LATENT_DIM)
classifier = define_classifier(latent_dim=LATENT_DIM, num_classes=np.unique(y_train).shape[0])
cae_classifier = CAE_CLASSIFIER(encoder, decoder, classifier)

weights_path = os.path.join(CAE_MODEL_DIR, 'best_val_accuracy.weights.h5')
cae_classifier.load_weights(weights_path)

encoder_classifier = Sequential([
    cae_classifier.encoder,   # Add encoder
    cae_classifier.classifier  # Add classifier
])

classification_performance_summary(encoder_classifier, x_train, y_train, y_encoder, save_path=os.path.join(CAE_MODEL_DIR, 'classification_performance_summary_train.png'))
classification_performance_summary(encoder_classifier, x_val, y_val, y_encoder, save_path=os.path.join(CAE_MODEL_DIR, 'classification_performance_summary_val.png'))
classification_performance_summary(encoder_classifier, x_test, y_test, y_encoder, save_path=os.path.join(CAE_MODEL_DIR, 'classification_performance_summary_test.png'))
latent_viz_data = visualize_latent_space_2d(encoder=cae_classifier.encoder, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test, label_encoder=y_encoder, save_path=os.path.join(CAE_MODEL_DIR, 'latent_space_2d_fold.png'))