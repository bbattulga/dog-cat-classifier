import argparse


def get_args():
    """Argument parser.

    Returns:
        Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        dest='job_dir',
        type=str,
        help='GCS location to write checkpoints and export models')
    parser.add_argument(
        '--train-file',
        type=str,
        help='Training file local or GCS')
    parser.add_argument(
        '--train-labels-file',
        type=str,
        help='Training labels file local or GCS')
    parser.add_argument(
        '--test-file',
        type=str,
        help='Test file local or GCS')
    parser.add_argument(
        '--test-labels-file',
        type=str,
        help='Test file local or GCS')
    parser.add_argument(
        '--num-epochs',
        type=float,
        default=5,
        help='number of times to go through the data, default=5')
    parser.add_argument(
        '--batch-size',
        default=128,
        type=int,
        help='number of records to read during each training step, default=128')
    parser.add_argument(
        '--learning-rate',
        default=.01,
        type=float,
        help='learning rate for gradient descent, default=.001')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')

    # user params
    parser.add_argument(
        '--local',
        default=False)
    return parser.parse_args()


args = get_args()
