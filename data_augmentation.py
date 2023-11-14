from tf.keras.preprocessing.Image import ImageDataGenerator

# train_data_gen = ImageDataGenerator(
#     zca_whitening=False,
#     zca_epsilon=1e-06,
#     rotation_range=0,
#     width_shift_range=0.0,
#     height_shift_range=0.0,
#     brightness_range=None,
#     shear_range=0.0,
#     zoom_range=0.0,
#     channel_shift_range=0.0,
#     fill_mode='nearest',
#     cval=0.0,
#     horizontal_flip=False,
#     vertical_flip=False,
#     rescale=None,
#     preprocessing_function=None
# )

train_data_gen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=10,
    zoom_range=0.2,
    horizontal_flip=True,
)

test_data_gen = ImageDataGenerator()

train_generator = train_data_gen.flow_from_directory('./dataset/train', target_size=(32,32), batch_size=32, shuffle=True)
test_generator = test_data_gen.flow_from_directory('./dataset/test', target_size=(32,32), batch_size=32, shuffle=False)