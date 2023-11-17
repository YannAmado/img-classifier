from keras.preprocessing.image import ImageDataGenerator
from letter_classifier.create_model import Model
from letter_classifier.pipeline import Pipeline
from letter_classifier.config import *

model_initializer = Model()
# models = [model_initializer.mobile_net(),
#           model_initializer.deep_neural_network(),
#           model_initializer.double_convolutional()]
models=[model_initializer.mobile_net()]
pipeline = Pipeline(models=models)
pipeline.split_dataset(dataset_dir=ORIGINAL_DATASET_DIR, train_ratio=TRAIN_TEST_RATION)

pipeline.augment_dataset(subset='train', img_generator=IMG_GENERATOR, n_images_to_create=N_IMGS_TO_GENERATE_PER_CLASS)
img_generator=ImageDataGenerator()
pipeline.augment_dataset(subset='test', img_generator=img_generator, n_images_to_create=N_IMGS_TO_GENERATE_PER_CLASS)

pipeline.train_models(batch_size=BATCH_SIZE, epochs=EPOCHS, k_folds=K_FOLDS, loss=LOSS, optimizer=OPTIMIZER, metrics=METRIC)
