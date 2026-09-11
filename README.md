# super-potato

This is an attempt to run an LLM on an esp32p4. The implementation was copied from https://github.com/karpathy/llama2.c.

# Dependencies
Install and activate eim
https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/index.html

# Testing
https://github.com/espressif/esp-emulator
```
idf.py -D IDF_TOOLCHAIN=clang set-target esp32p4
idf.py -D IDF_TOOLCHAIN=clang merge-bin
esp-emu --chip esp32p4 --firmware build/merged-binary.bin
```

# Perf History
* achieved tok/s: 3.564400
  * New quantized 14M model
  * https://huggingface.co/ellishg/tinyllamas/tree/main
  * https://github.com/ellishg/llama2.c
* achieved tok/s: 182.286
  * Unroll loop
* achieved tok/s: 141.573
  * -ffast-math
* achieved tok/s: 140.869
  * CONFIG_COMPILER_OPTIMIZATION_PERF=y
* achieved tok/s: 139.544
  * Switched to clang toolchain
* achieved tok/s: 85.500

# Useful Links
* https://github.com/karpathy/llama2.c
* https://github.com/slvDev/esp32-ai
* https://github.com/espressif/esp-idf
