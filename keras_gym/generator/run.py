import keras
import tensorflow as tf
from keras_gym.keras_model.dense import build_model
from .generator import MyGenerator


def get_gpu():
    from keras import backend as K
    K.tensorflow_backend._get_available_gpus()


def enable_gpu():
    config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 56})
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)


def run():
    enable_gpu()
    train_generator = MyGenerator(image_filenames=['a.jpg', 'b.jpg', 'c.jpg'],
                                  labels=[0, 9, 5],
                                  batch_size=1)

    eval_generator = MyGenerator(image_filenames=['b.jpg'],
                                 labels=[9],
                                 batch_size=1)

    model = build_model()
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_generator.__len__(),
                        epochs=10,
                        verbose=1,
                        validation_data=eval_generator,
                        validation_steps=eval_generator.__len__(),
                        use_multiprocessing=True,
                        workers=16,
                        max_queue_size=32)


def train(model, x_train, y_train):
    model.fit(x_train, y_train, epochs=5, batch_size=32)


def evaluate(model, x_eval, y_eval):
    loss_and_metrics = model.evaluate(x_eval, y_eval, batch_size=128)


def predict(model, x_test):
    classes = model.predict(x_test, batch_size=128)


if __name__ == "__main__":
    run()
