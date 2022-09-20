"""
Command line arguments
"""
import argparse
import warnings

from domainlab import arg_parser


def parse_cmd_args():
    """
    Args for command line definition
    """
    parser = arg_parser.mk_parser_main()
    parser.add_argument('--d_dim', type=int, default=7,
                        help='number of domains (or clusters)')
    parser.add_argument('--pre_tr', type=float, default=0.5, help="threshold for pretraining: pretraining finishes "
                                                                  "when validation clustering accuracy "
                                                                  "exceeds the pre_tr value")
    parser.add_argument('--L', type=int, default=3, help="number of MC runs")
    parser.add_argument('--prior', type = str, default="Bern", help='specifies whether binary or continuous-valued '
                                                                    ' input data.Input either "Bern" for Bernoulli or '
                                                                    '"Gaus" for Gaussian prior distribution for the data.')
    parser.add_argument('--model', type = str, default="linear", help = "specify 'linear' for a fully-connected or "
                                                                        "'cnn' for a convolutional model architecture" )
    parser.add_argument('--pretrain', type = str, default = "False", help = "turn on/off pretraining (boolean flag)")

    args = parser.parse_args()

    return args
