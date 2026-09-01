# super-potato

This is an attempt to run an LLM on an esp32p4. The implementation was copied from https://github.com/karpathy/llama2.c.

# Dependencies
Install and activate eim
https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/index.html

# Testing
https://github.com/espressif/esp-emulator
```
idf.py set-target esp32p4
idf.py merge-bin
esp-emu --chip esp32p4 --firmware build/merged-binary.bin
```