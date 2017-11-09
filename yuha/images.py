from keras.preprocessing import image

from constant import Image


class ImageGeneratorKeras:
    def __init__(self, config):
        self._config = config
        self.resize = self._config[Image.IMAGE][Image.RESIZE]
        self.num_channel = self._config[Image.IMAGE][Image.NUM_CHANNEL]
        self.rescale = self._config[Image.IMAGE][Image.RESCALE]
        self.batch_size = self._config[Image.IMAGE][Image.BATCH_SIZE]

    def load_train_data(self, path, classes=None):
        datagen = image.ImageDataGenerator(
            rescale=1. / self.rescale,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        datagenerator = datagen.flow_from_directory(
            path,
            target_size=tuple(self.resize),
            color_mode='rgb' if self.num_channel == 3 else 'grayscale',
            classes=classes,
            class_mode='categorical',
            batch_size=self.batch_size
        )
        return datagenerator
