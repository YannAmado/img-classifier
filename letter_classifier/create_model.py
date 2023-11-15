import keras
import torch.nn as nn

from keras import applications, layers, metrics

class Model:
    def __init__(self,
                 num_classes,
                 img_size,
                 loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=metrics.CategoricalAccuracy()):
        self.num_classes = num_classes
        self.img_size = img_size
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def mobile_net(self):
        scaling_layer =layers.Rescaling(1./255, input_shape=(self.img_size, 3)),
        layer1 = applications.MobileNetV2(
            weights='imagenet',
            input_shape=(128, 128, 3),
            include_top=False
        )
        layer2 = layers.Flatten()
        layer3 = layers.Dense(7, activation='softmax')

        model = keras.Sequential([scaling_layer, layer1, layer2, layer3])

        layer1.trainable = False

        model = self.compile_model(model)
        return model

    def double_convolutional(self):
        normalization_layer =layers.Rescaling(1./255, input_shape=(self.img_size, 3)),
        conv1 = nn.Conv2d(3, 32, 3, padding=1)
        conv2 = nn.Conv2d(32, 64, 3, padding=1)
        fc1 = nn.Linear(64 * 32 * 32, 512)
        fc2 = nn.Linear(512, self.num_classes)
        pool = nn.MaxPool2d(2, 2)
        relu = nn.ReLU()
        dropout = nn.Dropout(0.5)
        model = keras.Sequential([normalization_layer, conv1, conv2, fc1, fc2, pool, relu, dropout])

        model = self.compile_model(model)
        return model

    def deep_neural_network(self):
        scaling_layer =layers.Rescaling(1./255, input_shape=(self.img_size, 3)),
        layer1 = layers.Dense(self.img_size[0]*self.img_size[1], activation='softmax')
        layer2 = layers.Dense(64, activation='softmax')
        layer3 = layers.Dense(self.num_classes, activation='softmax')
        model = keras.Sequential([scaling_layer, layer1, layer2, layer3])
        model = self.compile_model(model)
        return model

    def compile_model(self, model):
        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=self.metrics)
        return model
