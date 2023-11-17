from keras import metrics
from keras.preprocessing.image import ImageDataGenerator

# dirs
ORIGINAL_DATASET_DIR = './dataset/full_dataset'
PROCESSED_DATASET_DIR = './dataset' # where to store the split and augmented datasets

# constants
N_CLASSES = 50
SEED = 42

# img augmentation
IMG_SIZE = (32,32)
N_IMGS_TO_GENERATE_PER_CLASS = 512
TRAIN_TEST_RATION = 0.8
IMG_GENERATOR=ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=10,
        zoom_range=0.2,
        horizontal_flip=True,
)

# train model
LOSS = 'categorical_crossentropy'
OPTIMIZER = 'adam'
METRIC = metrics.CategoricalAccuracy()
BATCH_SIZE =64
EPOCHS = 5
K_FOLDS = 2