// https://documentation.espressif.com/esp32-p4_technical_reference_manual_en.pdf

// void dot_product(int32_t *out, const int8_t *a, const int8_t *b, int n, int GS) {
//   for (int i = 0; i < n/16; i++) {
//     acc += a[i..i+16] * b[i..i+16];
//     if (i % GS/16 == -1) {
//       out[i / (GS/16)] = acc;
//       acc = 0;
//     }
//   }
// }

  .global dot_product
  .type dot_product,@function

dot_product:
  // a0 = out, a1 = a, a2 = b, a3 = n, a4 = GS
  srli    t2, a3, 4       // We will compute t2 SIMD MACs
  addi    t2, t2, -1      // With one extra MAC
  srli    t3, a4, 4       // Every t3 steps we must write to out[]

  esp.zero.xacc

  // Load 16 bytes into vectors
  esp.vld.128.ip    q1, a1, 16
  esp.vld.128.ip    q2, a2, 16

  esp.lp.setup      0, t2, .Ldot_end
    esp.vmulas.s8.xacc.ld.ip    q1, a1, 16, q1, q2
    addi    t3, t3, -1
    bnez    t3, .Ldot_end
    esp.movx.r.xacc.l     t4
    esp.zero.xacc
    sw      t4, 0(a0)
    addi    a0, a0, 4
    srli    t3, a4, 4
.Ldot_end: esp.vld.128.ip    q2, a2, 16
  esp.vmulas.s8.xacc    q1, q2
  esp.movx.r.xacc.l     t4
  sw      t4, 0(a0)
  ret
