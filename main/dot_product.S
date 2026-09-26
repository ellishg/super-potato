// https://documentation.espressif.com/esp32-p4_technical_reference_manual_en.pdf

// void dot_product(int32_t *out, const int8_t *a, const int8_t *b, int n, int GS) {
//   for (int i = 0; i < n/GS; i++) {
//     int32_t acc = 0;
//     for (int j = 0; j < GS/16; j++) {
//       idx = i * GS/16 + j;
//       acc += a[idx..idx+16] * b[idx..idx+16];
//     }
//     out[i] = acc;
//   }

  .global dot_product
  .type dot_product,@function

dot_product:
  // a0 = out, a1 = a, a2 = b, a3 = n, a4 = GS
  div     t2, a3, a4
  srli    t3, a4, 4
  addi    t3, t3, -1

.Lloop:
  esp.zero.xacc

  // Load 16 bytes into vectors
  esp.vld.128.ip            q1, a1, 16
  esp.vld.128.ip            q2, a2, 16

  esp.lp.setup      0, t3, .Lend
    esp.vmulas.s8.xacc.ld.ip    q1, a1, 16, q1, q2
.Lend: .Ldot_end: esp.vld.128.ip    q2, a2, 16
  esp.vmulas.s8.xacc        q1, q2

  esp.movx.r.xacc.l         t4
  sw      t4, 0(a0)
  addi    a0, a0, 4

  addi    t2, t2, -1
  bnez    t2, .Lloop
  ret
