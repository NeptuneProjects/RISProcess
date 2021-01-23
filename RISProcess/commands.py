#!/usr/bin/env python3

import argparse

import numpy as np

def test1():
    parser = argparse.ArgumentParser(description="Test argparse")
    parser.add_argument("number", type=float, default=8.0)
    args = parser.parse_args()
    print("Hello world, this is Script I.")
    print(np.square(args.number))

def test2():
    print("Hello world, this is Script II.")
