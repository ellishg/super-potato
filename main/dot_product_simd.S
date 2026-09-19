// https://github.com/espressif/esp-dsp/blob/3c8ac0fdfec83740b783e200862c8d0c056de0ad/modules/dotprod/fixed/dsps_dp_s8_arp4.S

// https://godbolt.org/z/67TPP311T
// esp_err_t dsps_dp_s8_ansi(const int8_t *src1, const int8_t *src2, int32_t *dest, int len)
// {
//     int32_t acc = 0;
//     for (int i = 0 ; i < len ; i++) {
//         acc += src1[i] * src2[i];
//     }
//     *dest = acc;
//     return ESP_OK;
// }

  .text
  .align 4
  .global dsps_dp_s8_arp4
  .type dsps_dp_s8_arp4,@function

// void dsps_dp_s8_arp4(const int8_t *src1, const int8_t *src2, int32_t *dest, int len);
dsps_dp_s8_arp4:
  // a0 = src1
  // a1 = src2
  // a2 = dest
  // a3 = len
  // a4 = acc

  li    a4, 0
  add   a3, a3, a0

.Lloop:
  lb    a5, 0(a0)
  lb    a6, 0(a1)
  // acc += src1[i] * src2[i]
  mul   a5, a6, a5
  add   a4, a4, a5

  addi  a0, a0, 1
  addi  a1, a1, 1
  // if (src1 < src1 + len) goto loop
  blt   a0, a3, .Lloop

  // *dst = acc
  sw    a4, 0(a2)
  ret