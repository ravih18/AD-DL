"""

"""

def synthesis_func(args):
    from .synthesis_vae import evaluate_vae
    evaluate_vae(args)

def latent_viz_func(args):
    from .synthesis_vae import plot_latent_space

def create_synthesis_parser(subparser):
    
    from clinicadl.utils.cli_utils import parent_parser

    eval_parser = subparser.add_parser(
        'evalvae',
        parents=[parent_parser],
        help='''Evaluation of VAE'''
    )

    eval_parser.add_argument(
        "output_dir", type=str,
        help="Path to the trained model folder."
    )

    eval_parser.add_argument(
        'caps_dir',
        help='Data using CAPS structure.',
        default=None
    )

    eval_parser.add_argument(
        'mode',
        help='''Create synthesiss on testset for defined model.''',
        choices=['image', 'slice', 'patch', 'roi'],
        type=str
    )

    eval_parser.add_argument(
        'preprocessing',
        help='Defines the type of preprocessing of CAPS data.',
        choices=['t1-linear', 't1-extensive', 't1-volume', 'shepplogan'],
        type=str
    )

    eval_parser.add_argument(
        'tsv_path',
        help='TSV path with subjects/sessions to process.',
        default=None
    )

    eval_parser.add_argument(
        '-cpu', '--use_cpu', action='store_true',
        help='If provided, will use CPU instead of GPU.',
        default=False
    )

    eval_parser.add_argument(
        '-np', '--nproc',
        help='Number of cores used during the training.',
        type=int, default=2
    )

    eval_parser.add_argument(
        '--batch_size',
        default=2, type=int,
        help='Batch size for training.'
    )

    eval_parser.add_argument(
        '--unnormalize', '-un',
        help='Disable default MinMaxNormalization.',
        action="store_true",
        default=False
    )
    
    eval_parser.add_argument(
        '--diagnoses', '-d',
        help='List of diagnoses that will be selected for training.',
        default=['AD', 'CN'], nargs='+', type=str,
        choices=['AD', 'BV', 'CN', 'MCI', 'sMCI', 'pMCI']
    )

    eval_parser.add_argument(
        "--baseline", action="store_true", default=False,
        help="If provided, only the baseline sessions are used for training."
    )

    eval_parser.add_argument(
        "--multi_cohort",
        help="Performs multi-cohort training. In this case, caps_dir and tsv_path must be paths to TSV files.",
        action="store_true",
        default=False
    )

    synthesis_parser = eval_parser.add_subparser(
        'synthesis',
        parents=[parent_parser],
        help='''Synthesize PH images on testset for defined model.'''
    )
    synthesis_parser.set_defaults(func=synthesis_func)

    latentviz_parser = eval_parser.add_subparser(
        'latentviz',
        parents=[parent_parser],
        help='''Synthesize PH images on testset for defined model.'''
    )
    latentviz_parser.set_defaults(func=latent_viz_func)
