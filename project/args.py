from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(
        'Project Arguments'
    )

    parser.add_argument('--experiment', required=True, type=str,
                        help="Name of the experiment. All training data/logs/checkpoints/etc will be saved in the directory experiments/[args.experiment]/")
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 4).')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate (default: 1e-3).', dest='lr')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--seed', default=42, type=int,
                        help='seed for initializing training.')

    parser.add_argument('--timesteps', '-T', type=int, default=40,
                        help="Number of timesteps for one inference.")

    # Image dimensions
    parser.add_argument('--height', type=int,
                        help="Height dimension of the input", default=176)
    parser.add_argument('--width', type=int,
                        help="Width dimension of the input", default=240)

    return parser
