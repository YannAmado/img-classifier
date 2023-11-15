from keras.preprocessing.image import ImageDataGenerator
from letter_classifier.create_model import Model
from letter_classifier.pipeline import Pipeline

IMG_SIZE = (128,128)
model_initializer = Model(img_size=IMG_SIZE)
# models = [model_initializer.mobile_net(),
#           model_initializer.deep_neural_network(),
#           model_initializer.double_convolutional()]
models=[model_initializer.mobile_net()]
pipeline = Pipeline(img_size=IMG_SIZE, models=models)
pipeline.split_dataset(dataset_dir='./dataset/full_dataset', train_ratio=0.8)

img_generator=ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=10,
        zoom_range=0.2,
        horizontal_flip=True,
)
pipeline.augment_dataset(subset='train', img_generator=img_generator, n_images_to_create=512)
img_generator=ImageDataGenerator()
pipeline.augment_dataset(subset='test', img_generator=img_generator, n_images_to_create=512)

pipeline.train_models(batch_size=64, epochs=5, k_folds=3)
