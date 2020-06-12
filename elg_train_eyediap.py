#!/usr/bin/env python3
"""Main script for training a model for gaze estimation."""
import argparse

import coloredlogs
import tensorflow as tf


N_SEED = 32

import random as rn
import argparse
import pickle

from experiment_helper import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras import backend as K
from images_data_augmenter_seqaware import ImageDataAugmenter


def str2bool(v):
    """
    Convert string to boolean
    :param v: string
    :return: boolean
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def init_main():
    """
    Defines the type of input arguments expected.
    Definition of each of them is included in "help" variable of each argument
    :return: parsed input arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-exp", "--experiment", dest="experiment", default="NFEL5836", help="Experiment name")
    parser.add_argument("-dp", "--dropout", dest="dropout", type=float, default=0.3, help="Dropout value")
    parser.add_argument("-aug", "--augmentation", dest="augmentation", type=str2bool, default=True,
                        help="True if Data augmentation is activated")
    parser.add_argument("-bs", "--batch_size", dest="batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-lr", "--learning_rate", dest="learning_rate",type=float, default=0.0001, help="Learning rate")
    parser.add_argument("-epochs", "--epochs", dest="n_epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("-data", "--data_file", dest="data_files", default=[], help="Data file", action="append",
                        nargs=2)
    parser.add_argument("-gt", "--gt_files", dest="gt_files", default=[], help="Ground truth files", action="append",
                        nargs=2)
    parser.add_argument("-vgt", "--vector_gt_files", dest="vector_gt_files", default=[],
                        help="Vector ground truth files", action="append", nargs=2)
    parser.add_argument("-feats", "--face_features", dest="face_features_file", default=[], help="Face features file",
                        nargs=2)
    parser.add_argument("-test", "--test_folders", dest="test_folders", default=[], help="Test folders",
                        action="append", nargs=10)
    parser.add_argument("-vp", "--validation_participants", dest="val_parts", type=int, default=0,
                        help="Number of participants to perform validation on")
    parser.add_argument("-mlb", "--max_look_back", dest="max_look_back", type=int, default=4,
                        help="Maximum number of frames to take into account before current frame, in sequence mode")
    parser.add_argument("-t", "--title", dest="title", default="", help="Experiment description")
    parser.add_argument("-p", "--path", dest="path", type=str, default="", help="Path")
    parser.add_argument("-mp", "--multi_processing", dest="multi_processing", type=str2bool, default=False,
                        help="True if GPU multi processing is activated")
    return parser.parse_args()


if __name__ == '__main__':
    '''
    # Set global log level
    parser = argparse.ArgumentParser(description='Train a gaze estimation model.')
    parser.add_argument('-v', type=str, help='logging level', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    args = parser.parse_args()
    coloredlogs.install(
        datefmt='%d/%m %H:%M',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )

    # Initialize Tensorflow session
    tf.logging.set_verbosity(tf.logging.ERROR)
    gpu_options = tf.GPUOptions(allow_growth=True)
    for i in range(0, 15):
        # Specify which people to train on, and which to test on
        person_id = 'p%02d' % i
        other_person_ids = ['p%02d' % j for j in range(15) if i != j]
    '''


    # Parse input arguments
    print("Parsing arguments...")
    args = init_main()

    # Read data and ground truth (both 2D and 3D)       读取数据
    print("Reading input files...")
    data, gt, vgt, _ = read_input(args.data_files, args.path, args.gt_files, args.vector_gt_files)

    # Read face features
    print("Reading face features...")
    face_features = read_face_features_file(args.face_features_file)

    # Get train-validation split                         将数据分为训练和测试数据
    print("Splitting data in train and validation sets...")
    train, validation = train_valtest_split(data, vgt, face_features, args.test_folders, args.val_parts)
    '''
    # Get experiment details and methods
    print("Get experiment and define associated model...")
    experiment = ExperimentHelper.get_experiment(args.experiment)
    '''
    print("Preparing data...")
    variables = {'max_look_back': args.max_look_back}
    train, validation, variables = experiment.prepare_data(train, validation, variables)

    # Make sure that from this point on experiments are reproducible (not valid with multi_processing)
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed=N_SEED)
    rn.seed(N_SEED)

    augmenter = None
    if args.augmentation:
        print("Loading augmentation...")
        # Define data augmenter
        augmenter = ImageDataAugmenter(rotation_range=0,
                                       width_shift_range=5,  # pixels
                                       height_shift_range=5,  # pixels
                                       zoom_range=[0.98, 1.02],  # %
                                       horizontal_flip=True,
                                       illumination_range=[0.4, 1.75],
                                       gaussian_noise_range=0.03)
        print("Augmentation is on.")
    print(augmenter)  # Just checking  

    # Shuffle
    print("Initiate data generators...")
    train = unison_shuffled_copy(train)
    print("Training: ", len(train.x))
    experiment.init_data_gen_train(train, args.batch_size, augmenter, True, True)
    if validation is not None:
        validation = unison_shuffled_copy(validation)
        print("Test: ", len(validation.x))
        experiment.init_data_gen_val(validation, args.batch_size, None, False)


    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:

        # Declare some parameters
        batch_size = 32

        # Define some model-specific parameters
        elg_first_layer_stride = 1
        elg_num_modules = 3
        elg_num_feature_maps = 32

        # Define training data source
        from datasources import UnityEyes
        from datasources import HDF5Source
        unityeyes = UnityEyes(
            session,
            batch_size=batch_size,
            data_format='NCHW',
            unityeyes_path='/home/zhangbochen/imgs',
            min_after_dequeue=1000,
            generate_heatmaps=True,
            shuffle=True,
            staging=True,
            eye_image_shape=(36, 60),
            heatmaps_scale=1.0 / elg_first_layer_stride,
        )
        unityeyes.set_augmentation_range('translation', 2.0, 10.0)
        unityeyes.set_augmentation_range('rotation', 1.0, 10.0)
        unityeyes.set_augmentation_range('intensity', 0.5, 20.0)
        unityeyes.set_augmentation_range('blur', 0.1, 1.0)
        unityeyes.set_augmentation_range('scale', 0.01, 0.1)
        unityeyes.set_augmentation_range('rescale', 1.0, 0.5)
        unityeyes.set_augmentation_range('num_line', 0.0, 2.0)
        unityeyes.set_augmentation_range('heatmap_sigma', 7.5, 2.5)

        # Define model
        from models import ELG
        model = ELG(
            # Tensorflow session
            # Note: The same session must be used for the model and the data sources.
            session,

            # Model configuration parameters
            # first_layer_stride describes how much the input image is downsampled before producing
            #                    feature maps for eventual heatmaps regression
            # num_modules defines the number of hourglass modules, and thus the number of times repeated
            #             coarse-to-fine refinement is done.
            # num_feature_maps describes how many feature maps are refined over the entire network.
            first_layer_stride=elg_first_layer_stride,
            num_feature_maps=elg_num_feature_maps,
            num_modules=elg_num_modules,

            # The learning schedule describes in which order which part of the network should be
            # trained and with which learning rate.
            #
            # A standard network would have one entry (dict) in this argument where all model
            # parameters are optimized. To do this, you must specify which variables must be
            # optimized and this is done by specifying which prefixes to look for.
            # The prefixes are defined by using `tf.variable_scope`.
            #
            # The loss terms which can be specified depends on model specifications, specifically
            # the `loss_terms` output of `BaseModel::build_model`.
            learning_schedule=[
                {
                    'loss_terms_to_optimize': {
                        'heatmaps_mse': ['hourglass'],
                        'radius_mse': ['radius'],
                        'combined_loss': ['hourglass', 'densenet'],
                    },
                    'metrics': ['gaze_mse', 'gaze_ang'],
                    'learning_rate': 1e-3,
                },
            ],

            # Data sources for training (and testing).
            train_data={'eyediap_t': train},
            
            test_data={'eyediap_v': train},
        )

        # Train this model for a set number of epochs
        model.train(
            num_epochs=100,
        )
