import keras
import tensorflow as tf
import numpy as np

from sklearn.model_selection import KFold
from keras import metrics



class ModelTraining:
    def __init__(self, num_classes, img_size, batch_size, epochs, seed=42):
        self.num_classes = num_classes
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed

    def make_train_df(self, train_dir):
        train_ds = keras.utils.image_dataset_from_directory(
            train_dir,
            label_mode='categorical',
            seed=self.seed,
            image_size=(self.img_size[0], self.img_size[1]),
            batch_size=self.batch_size)

        self.train_X = np.concatenate(list(train_ds.map(lambda x, y: x)))
        self.train_y = np.concatenate(list(train_ds.map(lambda x, y: y)))

    def cross_validate(self, model, k_folds, loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=metrics.CategoricalAccuracy()):
        kfold = KFold(n_splits=k_folds, shuffle=True)
        for i, (train, test) in enumerate(kfold.split(self.train_X, self.train_y)):
            print(f'FOLD {i+1}:')
            model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics)
            model.fit(self.train_X[train], self.train_y[train],
                      batch_size=self.batch_size,
                      epochs=self.epochs,
                      validation_data=(self.train_X[test], self.train_y[test]))
            print('====================================================================================')


    def test_model(self, model, test_dir):
        test_ds = keras.utils.image_dataset_from_directory(
            test_dir,
            seed=self.seed,
            image_size=(self.img_size[0], self.img_size[1]),
            batch_size=self.batch_size)

        predictions = model.predict(test_ds)
        score = tf.nn.softmax(predictions)
        return score