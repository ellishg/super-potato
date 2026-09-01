#!/usr/bin/env bash

# https://huggingface.co/karpathy/tinyllamas/blob/main/stories260K/readme.md
wget -O $1 https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.bin
wget -O $2 https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.bin