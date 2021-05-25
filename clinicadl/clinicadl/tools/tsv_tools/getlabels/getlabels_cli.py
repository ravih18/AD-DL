def tsv_getlabels_func(args):
    from .getlabels import get_labels

    get_labels(
        args.merged_tsv,
        args.missing_mods,
        args.results_path,
        diagnoses=args.diagnoses,
        modality=args.modality,
        restriction_path=args.restriction_path,
        time_horizon=args.time_horizon,
        variables_of_interest=args.variables_of_interest,
        remove_smc=not args.keep_smc,
        verbose=args.verbose
    )


def create_getlabels_cli(tsv_subparser):

    from clinicadl.utils.cli_utils import parent_parser
    
    tsv_getlabels_subparser = tsv_subparser.add_parser(
        'getlabels',
        parents=[parent_parser],
        help='Get labels in separate tsv files.')

    tsv_getlabels_subparser.add_argument(
        "merged_tsv",
        help="Path to the file obtained by the command clinica iotools merge-tsv.",
        type=str)
    tsv_getlabels_subparser.add_argument(
        "missing_mods",
        help="Path to the folder where the outputs of clinica iotools missing-mods are.",
        type=str)
    tsv_getlabels_subparser.add_argument(
        "results_path",
        type=str,
        help="Path to the folder where tsv files are extracted.")

    # Optional arguments
    tsv_getlabels_subparser.add_argument(
        "--modality", "-mod",
        help="Modality to select sessions. Sessions which do not include the modality will be excluded.",
        default="t1w", type=str)
    tsv_getlabels_subparser.add_argument(
        "--diagnoses",
        help="Labels that must be extracted from merged_tsv.",
        nargs="+", type=str, choices=['AD', 'BV', 'CN', 'MCI', 'sMCI', 'pMCI'], default=['AD', 'CN'])
    tsv_getlabels_subparser.add_argument(
        "--time_horizon",
        help="Time horizon to analyse stability of MCI subjects.",
        default=36, type=int)
    tsv_getlabels_subparser.add_argument(
        "--restriction_path",
        help="Path to a tsv containing the sessions that can be included.",
        type=str, default=None)
    tsv_getlabels_subparser.add_argument(
        "--variables_of_interest",
        help="Variables of interest that will be kept in the final lists."
                "Default will keep the diagnosis, age and the sex needed for the split procedure.",
        type=str, nargs="+", default=None)
    tsv_getlabels_subparser.add_argument(
        "--keep_smc",
        help="This flag allows to keep SMC participants, else they are removed.",
        default=False, action="store_true"
    )

    tsv_getlabels_subparser.set_defaults(func=tsv_getlabels_func)
