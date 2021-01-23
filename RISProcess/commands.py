#!/usr/bin/env python3

import argparse

import numpy as np

def test1():
    parser = argparse.ArgumentParser(description="Test argparse")
    parser.add_argument("--n", type=float, default=8.)
    args = parser.parse_args()
    print("Hello world, this is Script I.")
    print(np.square(args.n))

def test2():
    parser = argparse.ArgumentParser(description="Test argparse")
    parser.add_argument("--n", type=float, default=64.)
    args = parser.parse_args()
    print("Hello world, this is Script II.")
    print(np.sqrt(args.n))
