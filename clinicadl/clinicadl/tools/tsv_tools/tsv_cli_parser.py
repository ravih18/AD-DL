def create_tsv_parser(subparser):
    
    tsv_parser = subparser.add_parser(
            'tsvtool',
            help='''Handle tsv files for metadata processing and data splits.''')

    tsv_subparser = tsv_parser.add_subparsers(
        title='''Task to execute with tsv tool:''',
        description='''What kind of task do you want to use with tsv tool?
                (restrict, getlabels, split, kfold, analysis).''',
        dest='tsv_task',
        help='''****** Tasks proposed by clinicadl tsv tool ******''')

    tsv_subparser.required = True

    from .restrict.restrict_cli import create_restrict_cli
    create_restrict_cli(tsv_subparser)

    from .getlabels.getlabels_cli import create_getlabels_cli
    create_getlabels_cli(tsv_subparser)

    from .split.split_cli import create_split_cli
    create_split_cli(tsv_subparser)

    from .kfold.kfold_cli import create_kfold_cli
    create_kfold_cli(tsv_subparser)

    from .analysis.analysis_cli import create_analysis_cli
    create_analysis_cli(tsv_subparser)