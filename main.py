"""Runs the interpreter on the file specified as the first argument"""
import sys

import interpreter

if __name__ == '__main__':
    filename = sys.argv[1]
    interpreter.run(filename)

