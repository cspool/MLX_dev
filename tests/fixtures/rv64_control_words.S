.text
.option norvc
.option norelax
.globl controller_words
controller_words:
  addi a2, a0, 0
  addi a0, a0, 1
  add a2, a0, a1
  mul a2, a0, a1
  slt a2, a1, a0
  xori a2, a2, 1
  fle.s a2, fa0, fa1
  fclass.s a3, fa0
  andi a3, a3, 768
  sltu a3, zero, a3
  fclass.s a4, fa1
  andi a4, a4, 768
  sltu a4, zero, a4
  xori a4, a4, 1
  and a3, a3, a4
  flt.s a2, fa1, fa0
  or a2, a2, a3
  beq a2, zero, 1f
  addi s4, s5, 0
  fsgnj.s fa1, fa0, fa0
1:
  fcvt.s.l fa0, a0, rne
