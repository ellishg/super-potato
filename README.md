# super-potato

This is an attempt to run an LLM on an esp32p4. The implementation was copied from https://github.com/karpathy/llama2.c.

# Dependencies
Install and activate eim
https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/index.html

# Building
```
idf.py build
```

# Running
```
idf.py flash monitor
# Use `Ctrl + ]` to exit
```

# Emulator
https://github.com/espressif/esp-emulator
To run on an emulator, first remove `CONFIG_ESP32P4_SELECTS_REV_LESS_V3` from `sdkconfig.defaults` and fullclean.
```
idf.py merge-bin
esp-emu --chip esp32p4 --firmware build/merged-binary.bin
```

# Perf History
* achieved tok/s: 0.608930
  * Refactored assembly
* achieved tok/s: 0.608929
  * Write more custom assembly
* achieved tok/s: 0.571630
  * Use custom esp32 SIMD instructions in some cases
* achieved tok/s: 0.515770
  * Hand-written dot product assembly and GCC
* achieved tok/s: 0.514566
  * Real hardware! esp32-p4-pico
  * https://docs.waveshare.com/ESP32-P4-Pico
* achieved tok/s: 3.502858
  * Fix watchdog warnings
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
