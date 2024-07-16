# encoding: utf-8

"""
Helpers to parse some argparse stuff... created this while tired.. so might become useful after a while because I
realize that I tried to hard and got only so far

"""

import os
import warnings


def parse_path(arg_path):
    """
    Returns the absolute path of a relative one. Parses home (~) correctly.

    Example usage:
        parse_path('~/Documents')
        parse_path('')
        parse_path('Documents')
        parse_path('Documents/test')  # Path does not exist

    :param arg_path: input string, represents a path
    :return: absolute path of given path
    """
    if arg_path:
        if '~' in arg_path:
            temp_path = os.path.expanduser(arg_path)
        else:
            temp_path = os.path.abspath(arg_path)
    else:
        temp_path = os.path.expanduser('~')
        warnings.warn('No input given: Defaulting to home dir')

    if not os.path.exists(temp_path):
        warnings.warn('Given path does not exist.')

    return temp_path
