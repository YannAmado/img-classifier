from .data_preparation import split_dataset
from .data_augmentation import make_dataframe, make_and_store_images
from .train_model import ModelTraining
from .config import PROCESSED_DATASET_DIR


class Pipeline:
    def __init__(self, models, seed=42):
        self.models = models
        self.seed = seed

    def split_dataset(self, dataset_dir, train_ratio=0.8):
        split_dataset(dataset_dir, './dataset',train_ratio, 1-train_ratio, self.seed)

    def augment_dataset(self, subset, img_generator, n_images_to_create=512):
        sdir = PROCESSED_DATASET_DIR + f'/{subset}'
        df = make_dataframe(sdir)
        print(df.head())
        print('length of dataframe is ', len(df))

        augdir = PROCESSED_DATASET_DIR + f'/{subset}_augmented'  # directory to store the images if it does not exist it will be created
        make_and_store_images(df, augdir, n_images_to_create, gen=img_generator, color_mode='rgb', save_prefix='aug_',
                              save_format='jpg')

    def train_models(self, batch_size, epochs, k_folds, loss, optimizer, metrics):
        model_training = ModelTraining(batch_size, epochs, seed=self.seed)
        model_training.make_train_df('./dataset/train_augmented')
        for model in self.models:
            print(f'MODELO: {model}, comeco da cross validation com K={k_folds}')
            model_training.cross_validate(model, k_folds, loss, optimizer, metrics)
            print('====================================================================================')
