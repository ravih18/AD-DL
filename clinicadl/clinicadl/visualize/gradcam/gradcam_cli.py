"""
clinicadl gradcam model_path caps_dir dataset_type tsv_path -cpu --nproc --verbose
"""

def gradcam_func(args):
    from .gradcam_vae import attention_map
    attention_map(args)

def create_gradcam_parser(subparser):

    from clinicadl.utils.cli_utils import parent_parser

    gradcam_parser = subparser.add_parser(
        'gradcam',
        parents=[parent_parser],
        help='''Create gradcams on testset for defined model.'''
    )

    gradcam_parser.add_argument(
        "output_dir", type=str,
        help="Path to the trained model folder."
    )

    gradcam_parser.add_argument(
        'caps_dir',
        help='Data using CAPS structure.',
        default=None
    )

    gradcam_parser.add_argument(
        'mode',
        help='''Create gradcams on testset for defined model.''',
        choices=['image', 'slice', 'patch', 'roi'],
        type=str
    )

    gradcam_parser.add_argument(
        'preprocessing',
        help='Defines the type of preprocessing of CAPS data.',
        choices=['t1-linear', 't1-extensive', 't1-volume', 'shepplogan'],
        type=str
    )

    gradcam_parser.add_argument(
        'tsv_path',
        help='TSV path with subjects/sessions to process.',
        default=None
    )

    gradcam_parser.add_argument(
        'target_layer',
        help='Target conv to plot gradcam',
        default=2
    )

    gradcam_parser.add_argument(
        '-cpu', '--use_cpu', action='store_true',
        help='If provided, will use CPU instead of GPU.',
        default=False
    )

    gradcam_parser.add_argument(
        '-np', '--nproc',
        help='Number of cores used during the training.',
        type=int, default=2
    )

    gradcam_parser.add_argument(
        '--batch_size',
        default=2, type=int,
        help='Batch size for training.'
    )

    gradcam_parser.add_argument(
        '--unnormalize', '-un',
        help='Disable default MinMaxNormalization.',
        action="store_true",
        default=False
    )
    
    gradcam_parser.add_argument(
        '--diagnoses', '-d',
        help='List of diagnoses that will be selected for training.',
        default=['AD', 'CN'], nargs='+', type=str,
        choices=['AD', 'BV', 'CN', 'MCI', 'sMCI', 'pMCI']
    )

    gradcam_parser.add_argument(
        "--baseline", action="store_true", default=False,
        help="If provided, only the baseline sessions are used for training."
    )

    gradcam_parser.add_argument(
        "--multi_cohort",
        help="Performs multi-cohort training. In this case, caps_dir and tsv_path must be paths to TSV files.",
        action="store_true",
        default=False
    )


    gradcam_parser.set_defaults(func=gradcam_func)
