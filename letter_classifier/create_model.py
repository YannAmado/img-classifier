import keras

from keras import applications, layers

from .config import IMG_SIZE, N_CLASSES

class Model:
    def mobile_net(self):
        normalization_layer =layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        layer1 = applications.MobileNetV2(
            weights='imagenet',
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            include_top=False
        )
        layer2 = layers.Flatten()
        layer3 = layers.Dense(N_CLASSES, activation='softmax')
        model = keras.Sequential([normalization_layer, layer1, layer2, layer3])

        layer1.trainable = False
        return model

    def deep_neural_network(self):
        normalization_layer =layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        layer1 = layers.Dense(IMG_SIZE[0]*IMG_SIZE[1], activation='softmax')
        layer2 = layers.Dense(64, activation='softmax')
        layer3 = layers.Dense(N_CLASSES, activation='softmax')
        model = keras.Sequential([normalization_layer, layer1, layer2, layer3])
        return model

    def double_convolutional(self):
        normalization_layer =layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        conv1 = layers.Conv2D(32, 3)
        conv2 = layers.Conv2D(64, 3)
        fc1 = layers.Dense(64 * 32 * 32, activation='softmax')
        fc2 = layers.Dense(512, activation='softmax')
        pool = layers.MaxPooling2D(2, 2)
        relu = layers.ReLU()
        dropout = layers.Dropout(0.5)
        model = keras.Sequential([normalization_layer, conv1, conv2, fc1, fc2, pool, relu, dropout])
        return model