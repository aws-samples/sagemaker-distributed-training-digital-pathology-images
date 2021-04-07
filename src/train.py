import argparse
import json
import logging
import os

from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization, \
    GlobalAveragePooling2D, Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.resnet50 import ResNet50
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score

HEIGHT = 512
WIDTH = 512
DEPTH = 3
NUM_CLASSES = 3
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 150


def _dataset_parser(value):
    image_feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(value, image_feature_description)
    image = tf.io.decode_raw(example['image_raw'], tf.float32)
    image = tf.cast(image, tf.float32)
    image.set_shape([DEPTH * HEIGHT * WIDTH])
    image = tf.cast(tf.reshape(image, [HEIGHT, WIDTH, DEPTH]), tf.float32)
    label = tf.cast(example['label'], tf.int32)

    return image, tf.one_hot(label, NUM_CLASSES)


def _dataset_parser_with_slide(value):
    image_feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'slide_string': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(value, image_feature_description)
    image = tf.io.decode_raw(example['image_raw'], tf.float32)
    image = tf.cast(image, tf.float32)
    image.set_shape([DEPTH * HEIGHT * WIDTH])
    image = tf.cast(tf.reshape(image, [HEIGHT, WIDTH, DEPTH]), tf.float32)
    label = tf.cast(example['label'], tf.int32)
    slide = example['slide_string']

    return image, label, slide


def get_filenames(input_data_dir, hvd):
    return glob.glob('{}/*.tfrecords'.format(input_data_dir))


def valid_input_fn(hvd, mpi=False):
    if mpi:
        return _input(args.epochs, args.batch_size, None, 'valid', hvd)
    else:
        return _input(args.epochs, args.batch_size, args.valid, 'valid')


def test_input_fn():
    return _input(args.epochs, args.batch_size, args.test, 'test')


def train_input_fn(hvd, mpi=False):
    if mpi:
        return _input(args.epochs, args.batch_size, None, 'train', hvd)
    else:
        return _input(args.epochs, args.batch_size, args.train, 'train')


def _input(epochs, batch_size, channel, channel_name, hvd=None):
    if hvd != None:
        channel_name = '{}_{}'.format(channel_name, hvd.rank() % 4)

    print("The channel name is ", channel_name)
    channel_input_dir = args.training_env['channel_input_dirs'][channel_name]
    print("The corresponding input directory is ", channel_input_dir)
    mode = args.data_config[channel_name]['TrainingInputMode']
    if mode == 'Pipe':
        from sagemaker_tensorflow import PipeModeDataset
        dataset = PipeModeDataset(channel=channel_name, record_format='TFRecord')
    else:
        filenames = get_filenames(channel_input_dir, hvd)
        print("The correpsonding filenames are", filenames)
        dataset = tf.data.TFRecordDataset(filenames)

    if 'test' in channel_name:
        dataset = dataset.map(_dataset_parser_with_slide)
    else:
        dataset = dataset.repeat(epochs)
        dataset = dataset.map(_dataset_parser)

    if 'train' in channel_name:
        # Ensure that the capacity is sufficiently large to provide good random shuffling.
        buffer_size = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * 0.4) + 3 * batch_size
        dataset = dataset.shuffle(buffer_size=buffer_size)

    # Batch it up (only for train and valid)
    if 'test' not in channel_name:
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(10)

    return dataset


def model_def(learning_rate, mpi=False, hvd=False):
    inputs = Input(shape=(HEIGHT, WIDTH, DEPTH), name='inputs')
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, DEPTH),
                          input_tensor=inputs)
    base_model.trainable = False

    x1 = base_model.output
    x1 = GlobalAveragePooling2D()(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Dense(512, activation='relu')(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Dense(NUM_CLASSES, activation='softmax')(x1)

    model = Model(inputs=[base_model.input], outputs=[x1])
    for layer in model.layers[:20]:
        layer.trainable = False
    for layer in model.layers[20:]:
        layer.trainable = True

    size = hvd.size() if mpi else 1

    opt = SGD(lr=learning_rate * size)

    if mpi:
        opt = hvd.DistributedOptimizer(opt)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'], experimental_run_tf_function=False)
    return model


def main(args):
    mpi = False
    if 'sagemaker_mpi_enabled' in args.fw_params:
        if args.fw_params['sagemaker_mpi_enabled']:
            import horovod.keras as hvd
            mpi = True
            # Horovod: initialize Horovod.
            hvd.init()

            # Horovod: pin GPU to be used to process local rank (one GPU per process)
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if gpus:
                tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    else:
        hvd = None

    callbacks = []
    if mpi:
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        callbacks.append(hvd.callbacks.MetricAverageCallback())

        if hvd.rank() == 0:
            callbacks.append(ModelCheckpoint(args.output_dir + '/checkpoint-{epoch}.ckpt',
                                             save_weights_only=True,
                                             verbose=2))
    else:
        callbacks.append(ModelCheckpoint(args.output_dir + '/checkpoint-{epoch}.ckpt',
                                         save_weights_only=True,
                                         verbose=2))

    current_host = os.environ['SM_CURRENT_HOST']
    print("The current horovod rank is ", hvd.rank())
    print("the current host is ", current_host)

    print("Training dataset being loaded -----------------")
    train_dataset = train_input_fn(hvd, mpi)

    print("valid dataset being loaded -----------------")
    valid_dataset = valid_input_fn(hvd, mpi)

    print("Test dataset being loaded -----------------")
    test_dataset = test_input_fn()

    logging.info("configuring model")
    model = model_def(args.learning_rate, mpi, hvd)

    logging.info("Starting training")

    size = 1
    if mpi:
        size = hvd.size()
    print("the size is ", size)

    # Fit the model
    model.fit(train_dataset,
              steps_per_epoch=((args.num_train // args.batch_size) // size),
              epochs=args.epochs,
              validation_data=valid_dataset,
              validation_steps=((args.num_val // args.batch_size) // size),
              callbacks=callbacks,
              verbose=2)

    # Evaluate the model at rank 0
    if not mpi or (mpi and hvd.rank() == 0):
        print("-------------------------Evaluation begins ----------------------------------------------------")

        # Accumulate per-slide predictions
        pred_dict = {}
        for i, element in enumerate(test_dataset):
            if (i + 1) % 1000 == 0:
                print("Computing scores for tile {}...".format(i + 1))
                logging.info("Computing scores for slide {}...".format(i + 1))

            image = element[0].numpy()
            label = element[1].numpy()
            slide = element[2].numpy().decode()

            if slide not in pred_dict.keys():
                pred_dict[slide] = {'y_true': label, 'y_pred': []}
            pred = model.predict(np.expand_dims(image, axis=0))[0]
            pred_dict[slide]['y_pred'].append(pred)

        # Aggregate per-slide predictions
        y_true = []
        y_pred = []
        for key, value in pred_dict.items():
            slide_true = value['y_true']
            pred_scores_list = value['y_pred']
            mean_pred_scores = np.mean(np.vstack(pred_scores_list), axis=0)
            mean_pred_class = np.argmax(mean_pred_scores)

            y_true.append(slide_true)
            y_pred.append(mean_pred_class)

            print('Slide {}: True Label = {}, Prediction = {}'.format(key, slide_true, mean_pred_class))
            logging.info('Slide {}: True Label = {}, Prediction = {}'.format(key, slide_true, mean_pred_class))

        acc = accuracy_score(y_true, y_pred)
        print('Per-Slide Test accuracy: {}'.format(acc))
        logging.info('Per-Slide Test accuracy: {}'.format(acc))

    if mpi:
        if hvd.rank() == 0:
            model_path = '{}/00000001'.format(args.model_output_dir)
            model.save(model_path)
    else:
        model_path = '{}/00000001'.format(args.model_output_dir)
        model.save(model_path)
        model.save(args.model_output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train',
        type=str,
        required=False,
        default=os.environ.get('SM_CHANNEL_TRAIN'),
        help='The directory where the input training data is stored.')
    parser.add_argument(
        '--valid',
        type=str,
        required=False,
        default=os.environ.get('SM_CHANNEL_VALID'),
        help='The directory where the input validation data is stored.')
    parser.add_argument(
        '--test',
        type=str,
        required=False,
        default=os.environ.get('SM_CHANNEL_TEST'),
        help='The directory where the input test data is stored.')
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='The directory where the model will be stored.')
    parser.add_argument(
        '--model_output_dir',
        type=str,
        default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument(
        '--output-dir',
        type=str,
        default=os.environ.get('SM_OUTPUT_DIR'))
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=2e-4,
        help='Weight decay for convolutions.')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help="""\
        This is the inital learning rate value. The learning rate will decrease
        during training. For more details check the model_fn implementation in
        this file.\
        """)
    parser.add_argument(
        '--epochs',
        type=int,
        default=2,
        help='The number of steps to use for training.')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for training.')
    parser.add_argument(
        '--data-config',
        type=json.loads,
        default=os.environ.get('SM_INPUT_DATA_CONFIG')
    )
    parser.add_argument(
        '--fw-params',
        type=json.loads,
        default=os.environ.get('SM_FRAMEWORK_PARAMS')
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default='0.9'
    )
    parser.add_argument(
        '--training-env',
        type=json.loads,
        default=os.environ.get('SM_TRAINING_ENV')
    )
    parser.add_argument(
        '--num-val',
        type=float
    )
    parser.add_argument(
        '--num-train',
        type=float)
    parser.add_argument(
        '--num-test',
        type=float)

    args = parser.parse_args()
    main(args)
