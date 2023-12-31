import os
import pandas as pd
import shutil
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator

from .config import IMG_SIZE

def make_dataframe(sdir):
    # sdir is the directory when the class subdirectories are stored
    filepaths=[]
    labels=[]
    classlist=sorted(os.listdir(sdir) )
    for klass in classlist:
        classpath=os.path.join(sdir, klass)
        if os.path.isdir(classpath):
            flist=sorted(os.listdir(classpath))
            desc=f'{klass:25s}'
            for f in tqdm(flist, ncols=130,desc=desc, unit='files', colour='blue'):
                fpath=os.path.join(classpath,f)
                filepaths.append(fpath)
                labels.append(klass)
    Fseries=pd.Series(filepaths, name='filepaths')
    Lseries=pd.Series(labels, name='labels')
    df=pd.concat([Fseries, Lseries], axis=1)
    # return a dataframe with columns filepaths, labels
    return df

def make_and_store_images(df, augdir, n,  gen, color_mode='rgb', save_prefix='aug-',save_format='jpg'):
    #augdir is the full path where augmented images will be stored
    #n is the number of augmented images that will be created for each class that has less than n image samples
    # img_size  is a tupple(height,width) that specifies the size of the augmented images
    # color_mode is 'rgb by default'
    # save_prefix is the prefix augmented images are identified with by default it is 'aug-'
    #save_format is the format augmented images will be save in, by default it is 'jpg'
    # see documentation of ImageDataGenerator at https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator for details
    df=df.copy()
    if os.path.isdir(augdir):# start with an empty directory
        shutil.rmtree(augdir)
    os.mkdir(augdir)  # if directory does not exist create it
    for label in df['labels'].unique():
        classpath=os.path.join(augdir,label)
        os.mkdir(classpath) # make class directories within aug directory
    # create and store the augmented images
    total=0
    # in ImageDateGenerator select the types of augmentation you desire  below are some examples
    groups=df.groupby('labels') # group by class
    for label in df['labels'].unique():  # for every class
        classdir=os.path.join(augdir, label)
        group=groups.get_group(label)  # a dataframe holding only rows with the specified label
        sample_count=len(group)   # determine how many samples there are in this class
        aug_img_count=0
        aug_gen=gen.flow_from_dataframe( group,  x_col='filepaths', y_col=None, target_size=IMG_SIZE,
                                        class_mode=None, batch_size=1, shuffle=False,
                                        save_to_dir=classdir, save_prefix=save_prefix, color_mode=color_mode,
                                        save_format=save_format)

        while aug_img_count<n:
            images=next(aug_gen)
            aug_img_count += len(images)
        total +=aug_img_count
    print('Total Augmented images created =', total)

# sdir=r'../dataset/train'
# df=make_dataframe(sdir)
# print (df.head())
# print ('length of dataframe is ',len(df))
#
# augdir=r'./dataset/train_augmented' # directory to store the images if it does not exist it will be created
# n=500 # How many images to create
# img_size=IMG_SIZE # image size (height,width) of augmented images
#

#
# img_generator=ImageDataGenerator(
#         width_shift_range=0.1,
#         height_shift_range=0.1,
#         shear_range=10,
#         zoom_range=0.2,
#         horizontal_flip=True,
# )
#
# make_and_store_images(df, augdir, n,  img_size, gen=img_generator,  color_mode='rgb', save_prefix='aug_',save_format='jpg')