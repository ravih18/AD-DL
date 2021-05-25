def tsv_kfold_func(args):
    from .kfold import split_diagnoses

    split_diagnoses(
        args.formatted_data_path,
        n_splits=args.n_splits,
        subset_name=args.subset_name,
        MCI_sub_categories=args.MCI_sub_categories,
        stratification=args.stratification,
        verbose=args.verbose
    )

def create_kfold_cli(tsv_subparser):

    from clinicadl.utils.cli_utils import parent_parser
    
    tsv_kfold_subparser = tsv_subparser.add_parser(
        'kfold',
        parents=[parent_parser],
        help='Performs a k-fold split on participant level.')

    tsv_kfold_subparser.add_argument(
        "formatted_data_path",
        help="Path to the folder containing data extracted by clinicadl tsvtool getlabels.",
        type=str)

    # Optional arguments
    tsv_kfold_subparser.add_argument(
        "--n_splits",
        help="Number of folds in the k-fold split."
                "If 0, there is no training set and the whole dataset is considered as a test set.",
        type=int, default=5)
    tsv_kfold_subparser.add_argument(
        "--MCI_sub_categories",
        help="Deactivate default managing of MCI sub-categories to avoid data leakage.",
        action="store_false", default=True)
    tsv_kfold_subparser.add_argument(
        "--subset_name",
        help="Name of the subset that is complementary to train.",
        type=str, default="validation")
    tsv_kfold_subparser.add_argument(
        "--stratification",
        help="Name of a variable used to stratify the k-fold split.",
        type=str, default=None)

    tsv_kfold_subparser.set_defaults(func=tsv_kfold_func)
