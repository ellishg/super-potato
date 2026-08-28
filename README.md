# super-potato

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