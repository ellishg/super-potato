#!/usr/bin/env bash

# https://huggingface.co/ellishg/tinyllamas/blob/main/README.md
wget -O $1 https://huggingface.co/ellishg/tinyllamas/resolve/main/stories3_5M-Q8_0-v32k.bin
wget -O $2 https://huggingface.co/ellishg/tinyllamas/resolve/main/tok32000.bin
