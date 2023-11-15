from .data_preparation import split_dataset
from .data_augmentation import make_dataframe, make_and_store_images
from .train_model import ModelTraining


class Pipeline:
    def __init__(self, img_size, models, num_classes=50, seed=42):
        self.img_size = img_size
        self.models = models
        self.num_classes = num_classes
        self.seed = seed

    def split_dataset(self, dataset_dir, train_ratio=0.8):
        split_dataset(dataset_dir, './dataset',train_ratio, 1-train_ratio, self.seed)

    def augment_dataset(self, subset, img_generator, n_images_to_create=512):
        sdir = f'./dataset/{subset}'
        df = make_dataframe(sdir)
        print(df.head())
        print('length of dataframe is ', len(df))

        augdir = f'./dataset/{subset}_augmented'  # directory to store the images if it does not exist it will be created
        make_and_store_images(df, augdir, n_images_to_create, self.img_size, gen=img_generator, color_mode='rgb', save_prefix='aug_',
                              save_format='jpg')

    def train_models(self, batch_size, epochs, k_folds):
        model_training = ModelTraining(self.num_classes, self.img_size, batch_size, epochs, seed=self.seed)
        model_training.make_train_df('./dataset/train_augmented')
        for model in self.models:
            print(f'MODELO: {model}, comeco da cross validation com K={k_folds}')
            model_training.cross_validate(model, k_folds)
            print('====================================================================================')
