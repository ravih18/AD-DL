def tsv_split_func(args):
    from .split import split_diagnoses

    split_diagnoses(
        args.formatted_data_path,
        n_test=args.n_test,
        subset_name=args.subset_name,
        MCI_sub_categories=args.MCI_sub_categories,
        p_age_threshold=args.p_age_threshold,
        p_sex_threshold=args.p_sex_threshold,
        ignore_demographics=args.ignore_demographics,
        train_with_cn=args.train_with_cn,
        verbose=args.verbose,
        categorical_split_variable=args.categorical_split_variable
    )

def create_split_cli(tsv_subparser):

    from clinicadl.utils.cli_utils import parent_parser

    tsv_split_subparser = tsv_subparser.add_parser(
        'split',
        parents=[parent_parser],
        help='Performs one stratified shuffle split on participant level.')

    tsv_split_subparser.add_argument(
        "formatted_data_path",
        help="Path to the folder containing data extracted by clinicadl tsvtool getlabels.",
        type=str)

    # Optional arguments
    tsv_split_subparser.add_argument(
        "--n_test",
        help="If >= 1, number of subjects to put in set with name 'subset_name'. "
                "If < 1, proportion of subjects to put set with name 'subset_name'. "
                "If 0, no training set is created and the whole dataset is considered as one set with name 'subset_name.",
        type=float, default=100.)
    tsv_split_subparser.add_argument(
        "--MCI_sub_categories",
        help="Deactivate default managing of MCI sub-categories to avoid data leakage.",
        action="store_false", default=True)
    tsv_split_subparser.add_argument(
        "--p_sex_threshold", "-ps",
        help="The threshold used for the chi2 test on sex distributions.",
        default=0.80, type=float)
    tsv_split_subparser.add_argument(
        "--p_age_threshold", "-pa",
        help="The threshold used for the T-test on age distributions.",
        default=0.80, type=float)
    tsv_split_subparser.add_argument(
        "--subset_name",
        help="Name of the subset that is complementary to train.",
        type=str, default="test")
    tsv_split_subparser.add_argument(
        "--ignore_demographics",
        help="If True do not use age and sex to create the splits.",
        default=False, action="store_true"
    )
    tsv_split_subparser.add_argument(
        "--train_with_cn",
        help="If True will construct a training set with only CN and put CN, MCI and AD in test",
        default=False, action="store_true"
    )
    tsv_split_subparser.add_argument(
        "--categorical_split_variable",
        help="Name of a categorical variable used for a stratified shuffle split "
                "(in addition to age and sex selection).",
        default=None, type=str
    )

    tsv_split_subparser.set_defaults(func=tsv_split_func)