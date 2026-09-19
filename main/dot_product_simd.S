// https://github.com/espressif/esp-dl/blob/master/tools/agents/skills/esp32p4-pie-simd/references/instructions.md

// https://godbolt.org/z/67TPP311T
// int32_t dsps_dp_s8_ansi(const int8_t *a, const int8_t *b, int len)
// {
//     int32_t acc = 0;
//     for (int i = 0 ; i < len ; i++)
//         acc += a[i] * b[i];
//     return acc;
// }

  .text
  .align 4
  .global dsps_dp_s8_arp4
  .type dsps_dp_s8_arp4,@function

// int32_t dsps_dp_s8_arp4(const int8_t *a, const int8_t *b, int len);
dsps_dp_s8_arp4:
  // a0, q0 = a
  // a1, q1 = b
  // a2     = len

  srli  a2, a2, 4
  esp.zero.xacc

  // Load 16 bytes into vectors
  esp.vld.128.ip    q0, a0, 16
  esp.vld.128.ip    q1, a1, 16

  // Loop len/16 times (.Lloop_end is the last instruction in each iteration)
  esp.lp.setup      0, a2, .Lloop_end
    esp.vmulas.s8.xacc.ld.ip    q0, a0, 16, q0, q1
.Lloop_end: esp.vld.128.ip    q1, a1, 16

  esp.movx.r.xacc.l   a0
  ret
