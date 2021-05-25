import argparse

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument('--verbose', '-v', action='count', default=0)