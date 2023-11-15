# pip install split_folders
import splitfolders # or import splitfolders

def split_dataset(input_folder, output_folder, train_ratio, test_ratio, seed):
    splitfolders.ratio(input_folder, output=output_folder, seed=seed, ratio=(train_ratio, .0, test_ratio)) # ratio of split are in order of train/val/test. You can change to whatever you want. For train/val sets only, you could do .75, .25 for example.