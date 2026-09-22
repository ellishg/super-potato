// https://documentation.espressif.com/esp32-p4_technical_reference_manual_en.pdf

// float matmul_asm(const int8_t *w, const float *ws, const int8_t *x, const float *xs, int n, int GS) {
//   float acc = 0.f;
//   for (int j = 0; j < n; j += GS) {
//     int32_t ival = 0;
//     for (int k = 0; k < GS; k++)
//       ival += w[j + k] * x[j + k];
//     acc += (float)ival * ws[j / GS] * xs[j / GS];
//   }
//   return acc;
// }

  .global matmul_asm
  .type matmul_asm,@function

matmul_asm:
  // a0 = w, a1 = ws, a2 = x, a3 = xs, a4 = n, a5 = GS
  li      t0, 0
  fcvt.s.w fa0, t0                 // float acc = 0.0f
  mv      t6, a5


.Lmatmul_group:
  beqz    a4, .Lmatmul_done
  srli    t2, t6, 4                 // Number of 16-byte SIMD chunks.
  addi    t2, t2, -1

  // Load 16 bytes into vectors
  esp.vld.128.ip    q0, a0, 16
  esp.vld.128.ip    q1, a2, 16

  esp.zero.xacc

  // Loop len/16 times (.Ldot_end is the last instruction in each iteration)
  esp.lp.setup      0, t2, .Ldot_end
    esp.vmulas.s8.xacc.ld.ip    q0, a0, 16, q0, q1
.Ldot_end: esp.vld.128.ip    q1, a2, 16
  esp.vmulas.s8.xacc    q0, q1

  esp.movx.r.xacc.l   a5

  fcvt.s.w ft0, a5                  // (float)ival
  flw     ft1, 0(a1)                // ws[group]
  flw     ft2, 0(a3)                // xs[group]
  fmul.s  ft0, ft0, ft1
  fmul.s  ft0, ft0, ft2
  fadd.s  fa0, fa0, ft0
  addi    a1, a1, 4
  addi    a3, a3, 4
  sub     a4, a4, t6
  j       .Lmatmul_group

.Lmatmul_done:
  ret