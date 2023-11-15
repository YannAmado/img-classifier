import keras
import tensorflow as tf
import numpy as np

from sklearn.model_selection import cross_validate



class ModelTraining:
    def __init__(self, train_ds, num_classes, img_size, batch_size, epochs, seed=42):
        self.train_ds = train_ds
        self.num_classes = num_classes
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed


    def make_train_df(self, train_dir):
        train_ds = keras.utils.image_dataset_from_directory(
            train_dir,
            label_mode='categorical',
            validation_split=0.0,
            subset="training",
            seed=self.seed,
            image_size=(self.img_size[0], self.img_size[1]),
            batch_size=self.batch_size)

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        return train_ds

    def cross_validate(self, model, k_folds):
        scores = cross_validate(model, self.train_ds, cv=k_folds,
                scoring = 'accuracy',
                return_train_score = True)
        return scores


    def test_model(self, model, test_dir):
        test_ds = keras.utils.image_dataset_from_directory(
            test_dir,
            validation_split=0,
            subset="train",
            seed=self.seed,
            image_size=(self.img_size[0], self.img_size[1]),
            batch_size=self.batch_size)

        predictions = model.predict(test_ds)
        score = tf.nn.softmax(predictions)
        return score