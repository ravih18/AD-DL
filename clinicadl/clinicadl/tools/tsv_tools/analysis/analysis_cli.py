def tsv_analysis_func(args):
    from .analysis import demographics_analysis

    demographics_analysis(
        args.merged_tsv,
        args.formatted_data_path,
        args.results_path,
        diagnoses=args.diagnoses
    )

def create_analysis_cli(tsv_subparser):

    tsv_analysis_subparser = tsv_subparser.add_parser(
        'analysis',
        help='Produces a demographic analysis of the extracted labels.')

    tsv_analysis_subparser.add_argument(
        "merged_tsv",
        help="Path to the file obtained by the command clinica iotools merge-tsv.",
        type=str)
    tsv_analysis_subparser.add_argument(
        "formatted_data_path",
        help="Path to the folder containing data extracted by clinicadl tsvtool getlabels.",
        type=str)
    tsv_analysis_subparser.add_argument(
        "results_path",
        help="Path to the output tsv file (filename included).",
        type=str)

    # Modality selection
    tsv_analysis_subparser.add_argument(
        "--diagnoses",
        help="Labels selected for the demographic analysis.",
        default=['AD', 'CN'], nargs="+", type=str, choices=['AD', 'BV', 'CN', 'MCI', 'sMCI', 'pMCI'])

    tsv_analysis_subparser.set_defaults(func=tsv_analysis_func)
