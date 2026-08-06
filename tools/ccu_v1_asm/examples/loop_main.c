/*
 * C-style CCU V1 program covering every ISA opcode.
 * Operands match examples/all_opcodes.s so casm and numeric asm must cmp equal.
 *
 * Assembler context (CcuV1CasmCtx) accumulates packed 32B instructions;
 * loop() fills CcuV1Loop binary fields directly.
 */

void main()
{
    /* ---- LOAD ---- */
    load_sqeargs_to_gsa(1, 2);
    load_sqeargs_to_xn(3, 4);
    load_imd_to_gsa(5, 0x1122334455667788);
    load_imd_to_xn(6, 0x1000, 0);
    load_gsa_xn(7, 8, 9);
    load_gsa_gsa(1, 2, 3);
    load_xx(4, 5, 6);

    /* ---- CTRL ---- */
    loop(0, 10, 11);
    loop_group(0, 12, 13, 1);
    set_cke(0, 1, 0xffff, 2, 0x0001);
    clear_cke(1, 3, 0x00ff, 4, 0x0002);
    jmp(14, 15, 0xabcdef);

    /* ---- TRANS ---- */
    trans_loc_mem_to_loc_ms(0, 1, 2, 3, 0, 0, 1, 1, 0x1, 2, 0x2);
    trans_rmt_mem_to_loc_ms(1, 2, 3, 4, 1, 1, 0, 3, 0x3, 4, 0x4);
    trans_loc_ms_to_loc_mem(5, 6, 7, 8, 2, 0, 1, 5, 0x5, 6, 0x6);
    trans_loc_ms_to_rmt_mem(1, 2, 3, 4, 3, 0, 1, 0, 0, 0, 0);
    trans_rmt_ms_to_loc_mem(2, 3, 4, 5, 4, 1, 1, 7, 0x7, 8, 0x8);
    trans_loc_ms_to_loc_ms(1, 2, 3, 5, 0, 1, 9, 0x9, 10, 0xa);
    trans_rmt_ms_to_loc_ms(3, 4, 5, 6, 0, 0, 0, 0, 0, 0);
    trans_loc_ms_to_rmt_ms(5, 6, 7, 7, 1, 0xb, 0, 1, 2, 0xc, 3, 0xd);
    trans_rmt_mem_to_loc_mem(1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0);
    trans_loc_mem_to_rmt_mem(2, 3, 4, 5, 6, 1, 0xab, 0xc, 0x5, 1, 1, 1, 4, 0xe, 5, 0xf);
    trans_loc_mem_to_loc_mem(1, 2, 3, 4, 5, 2, 0, 1, 6, 0x10, 7, 0x11);

    /* ---- SYNC ---- */
    sync_cke(1, 2, 0xffff, 0, 0, 1, 0x1, 2, 0x2);
    sync_gsa(3, 4, 1, 5, 0x20, 1, 6, 0x21, 7, 0x22);
    sync_xn(8, 9, 2, 10, 0x30, 0, 11, 0x31, 12, 0x32);

    /* ---- REDUCE ---- */
    add({0, 1, 2, 0, 0, 0, 0, 0}, 3, 1, 4, 13, 0, 1, 0x1, 2, 0x2);
    max({4, 5, 6, 7, 0, 0, 0, 0}, 4, 2, 14, 1, 3, 0x3, 4, 0x4);
    min({8, 9, 10, 0, 0, 0, 0, 0}, 3, 1, 15, 0, 5, 0x5, 6, 0x6);
}
