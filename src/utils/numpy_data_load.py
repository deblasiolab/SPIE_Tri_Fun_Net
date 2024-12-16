import numpy as np
from src.constants import SLIT_IMGS_PATH, SLIT_LABL_PATH, FOLDS_FOLDER_PATH

class NumpyDataLoader:
    def __init__(self, fold_idx):
        # Load data
        self.img_features = np.load(SLIT_IMGS_PATH).astype(np.float32)
        self.img_labels = np.load(SLIT_LABL_PATH)

        # Load folds
        self.train_fold = np.load(f'{FOLDS_FOLDER_PATH}/train_indices_fold_{fold_idx}.npy')
        self.val_fold = np.load(f'{FOLDS_FOLDER_PATH}/val_indices_fold_{fold_idx}.npy')
        self.test_fold = np.load(f'{FOLDS_FOLDER_PATH}/test_indices_fold_{fold_idx}.npy')

    def load_training_data(self):
        # Return training data
        return self.img_features[self.train_fold], self.img_labels[self.train_fold]

    def load_validation_data(self):
        # Return validation data
        return self.img_features[self.val_fold], self.img_labels[self.val_fold]

    def load_testing_data(self):
        # Return test data
        return self.img_features[self.test_fold], self.img_labels[self.test_fold]