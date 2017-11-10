import tensorflow as tf
import keras

from yuha.config import config
from yuha.constant import Train, Image
from yuha.images import ImageGeneratorKeras
from yuha.network import Models
from yuha.util import MODEL_PATH, MODEL_WEIGHT


class Pipelines(object):
    def __init__(self, config=config, train=True):
        self._config = config
        self.num_epoch = self._config[Train.TRAIN][Train.NUM_EPOCH]
        self.model = getattr(Models, self._config[Train.TRAIN][Train.MODEL])

    def keras_pipeline(self):
        test = ImageGeneratorKeras(self._config).load_train_data('images/testing_data')
        train = ImageGeneratorKeras(self._config).load_train_data('images/training_data')
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            self.model = self.model(input_shape=(244, 244, 3), classes=4)

            callback_ = keras.callbacks.ModelCheckpoint(
                'models/model_checkpoint.hdf5',
                monitor='val_loss',
                verbose=0,
                save_best_only=True,
                save_weights_only=False,
                mode='auto',
                period=1
            )

            # early_stopping = keras.callbacks.EarlyStopping(
            #     monitor='val_loss',
            #     min_delta=0.00001,
            #     patience=0,
            #     verbose=0,
            #     mode='auto'
            # )

            tensorboard = keras.callbacks.TensorBoard(
                log_dir='logs',
                histogram_freq=0,
                batch_size=self._config[Image.IMAGE][Image.BATCH_SIZE],
                write_graph=True,
                write_grads=True,
                write_images=True,
                embeddings_freq=0
            )

            self.model.fit_generator(
                train,
                steps_per_epoch=self._config[Image.IMAGE][Image.BATCH_SIZE],
                epochs=self.num_epoch,
                validation_data=test,
                validation_steps=64,
                use_multiprocessing=True,
                callbacks=[callback_, tensorboard]
            )

            self.model.save(MODEL_PATH)
            self.model.save_weights(MODEL_WEIGHT)

            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    pipeline = Pipelines(train=False)
    pipeline.keras_pipeline()
