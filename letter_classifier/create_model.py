import keras

from keras import applications, layers, metrics

class Model:
    def __init__(self,
                 img_size,
                 loss='categorical_crossentropy',
                 optimizer='adam',
                 num_classes=50,
                 metrics=metrics.CategoricalAccuracy()):
        self.num_classes = num_classes
        self.img_size = img_size
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def mobile_net(self):
        normalization_layer =layers.Rescaling(1./255, input_shape=(self.img_size[0], self.img_size[1], 3))
        layer1 = applications.MobileNetV2(
            weights='imagenet',
            input_shape=(self.img_size[0], self.img_size[1], 3),
            include_top=False
        )
        layer2 = layers.Flatten()
        layer3 = layers.Dense(self.num_classes, activation='softmax')

        model = keras.Sequential([normalization_layer, layer1, layer2, layer3])

        layer1.trainable = False

        model = self.compile_model(model)
        return model

    def deep_neural_network(self):
        normalization_layer =layers.Rescaling(1./255, input_shape=(self.img_size[0], self.img_size[1], 3))
        layer1 = layers.Dense(self.img_size[0]*self.img_size[1], activation='softmax')
        layer2 = layers.Dense(64, activation='softmax')
        layer3 = layers.Dense(self.num_classes, activation='softmax')
        model = keras.Sequential([normalization_layer, layer1, layer2, layer3])
        model = self.compile_model(model)
        return model

    def double_convolutional(self):
        normalization_layer =layers.Rescaling(1./255, input_shape=(self.img_size[0], self.img_size[1], 3))
        conv1 = layers.Conv2D(32, 3)
        conv2 = layers.Conv2D(64, 3)
        fc1 = layers.Dense(64 * 32 * 32, activation='softmax')
        fc2 = layers.Dense(512, activation='softmax')
        pool = layers.MaxPooling2D(2, 2)
        relu = layers.ReLU()
        dropout = layers.Dropout(0.5)
        model = keras.Sequential([normalization_layer, conv1, conv2, fc1, fc2, pool, relu, dropout])

        model = self.compile_model(model)
        return model

    def compile_model(self, model):
        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=self.metrics)
        return model
