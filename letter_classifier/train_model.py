import keras
import numpy as np

from sklearn.model_selection import KFold
from keras import metrics
from .config import IMG_SIZE



class ModelTraining:
    def __init__(self, batch_size, epochs, seed=42):
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed

    def make_train_df(self, train_dir):
        train_ds = keras.utils.image_dataset_from_directory(
            train_dir,
            label_mode='categorical',
            seed=self.seed,
            image_size=(IMG_SIZE[0], IMG_SIZE[1]),
            batch_size=self.batch_size)

        self.train_X = np.concatenate(list(train_ds.map(lambda x, y: x)))
        self.train_y = np.concatenate(list(train_ds.map(lambda x, y: y)))

    def cross_validate(self, model, k_folds, loss, optimizer, metrics):
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
            image_size=(IMG_SIZE[0], IMG_SIZE[1]),
            batch_size=self.batch_size)

        predictions = model.predict(test_ds)
        score = softmax(predictions)
        return score

    @staticmethod
    def _softmax(x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)
