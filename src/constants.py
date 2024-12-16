# Import libraries
import os

# Constants
DATA_PATH = '/datastore/researchdata/mnt/lcedillo/SPIE_trifun_net/data/raw_fits'
SAVE_PATH = '/datastore/researchdata/mnt/lcedillo/SPIE_trifun_net/data/numpy_vectors'
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER_PATH = os.path.join(ROOT_DIR, 'data', 'numpy_vectors')
FOLDS_FOLDER_PATH = os.path.join(ROOT_DIR, 'data', 'numpy_folds')
PLOTS_FOLDER_PATH = os.path.join(ROOT_DIR, 'data', 'fold_plots')
SLIT_IMGS_PATH = os.path.join(DATA_FOLDER_PATH, 'x_raw_slitless_spectro_single_channel_images.npy')
SLIT_LABL_PATH = os.path.join(DATA_FOLDER_PATH, 'y_satellite_string_class_labels.npy')
TOTAL_FOLDS = 10
SEED = 42
VAL_SPLIT_RATIO = 0.1  

# Model
LATENT_DIM = 64
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
EPOCHS = 1000

