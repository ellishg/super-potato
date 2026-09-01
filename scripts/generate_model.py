#!/usr/bin/env python3

import os
import sys

# TODO: Eventually this will produce a real model
with open(sys.argv[1], "wb") as f:
    f.write("The rest of this model is random data:\n".encode())
    f.write(os.urandom(4 * 1024 * 1024))