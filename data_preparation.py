# pip install split_folders
import splitfolders # or import splitfolders
input_folder = "./dataset/whole_dataset"
output = "./dataset" #where you want the split datasets saved. one will be created if it does not exist or none is set

train_ratio = 0.8
test_ratio = 0.2
splitfolders.ratio(input_folder, output=output, seed=42, ratio=(train_ratio, .0, test_ratio)) # ratio of split are in order of train/val/test. You can change to whatever you want. For train/val sets only, you could do .75, .25 for example.