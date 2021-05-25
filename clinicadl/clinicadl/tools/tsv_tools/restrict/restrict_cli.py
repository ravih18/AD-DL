def tsv_restrict_func(args):
    from .restrict import aibl_restriction, oasis_restriction

    if args.dataset == "AIBL":
        aibl_restriction(args.merged_tsv, args.results_path)
    elif args.dataset == "OASIS":
        oasis_restriction(args.merged_tsv, args.results_path)

def create_restrict_cli(tsv_subparser):

    tsv_restrict_subparser = tsv_subparser.add_parser(
        'restrict',
        help='Reproduce restrictions applied to AIBL and OASIS datasets')

    tsv_restrict_subparser.add_argument(
        "dataset",
        help="dataset on which the restriction is performed.",
        choices=["AIBL", "OASIS"],
        type=str)

    tsv_restrict_subparser.add_argument(
        "merged_tsv",
        help="Path to the file obtained by the command clinica iotools merge-tsv.",
        type=str)
    tsv_restrict_subparser.add_argument(
        "results_path",
        help="Path to the output tsv file (filename included).",
        type=str)

    tsv_restrict_subparser.set_defaults(func=tsv_restrict_func)
