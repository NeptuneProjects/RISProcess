#!/usr/bin/env python3

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from RISProcess import io
from RISProcess import workflows
from RISProcess.processing import SignalProcessing

# def test1():
#     parser = argparse.ArgumentParser(description="Test argparse")
#     parser.add_argument("--n", type=float, default=8.)
#     args = parser.parse_args()
#     print("Hello world, this is Script I.")
#     print(np.square(args.n))
#
# def test2():
#     parser = argparse.ArgumentParser(description="Test argparse")
#     parser.add_argument("--n", type=float, default=64.)
#     args = parser.parse_args()
#     print("Hello world, this is Script II.")
#     print(np.sqrt(args.n))

def process():
    parser = argparse.ArgumentParser(
        description="Command-line tool for processing RIS seismic data."
    )
    parser.add_argument("path", help="Path to config file")
    args = parser.parse_args()
    config_file = args.path
    config = configparser.ConfigParser()
    config.read(config_file)
    params = io.config(config)
    print(params)
