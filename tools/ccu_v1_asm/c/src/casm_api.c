/**
 * Instruction intrinsics: allocate ctx->inst and write payload fields directly.
 * No string formatting / emit_scalars / assembler round-trip.
 */
#include "ccu_v1_casm_api.h"

#include <string.h>

static CcuV1CasmCtx *require_ctx(void)
{
    CcuV1CasmCtx *ctx = ccu_v1_casm_current();
    if (!ctx || ctx->failed) {
        return NULL;
    }
    return ctx;
}

/* type/code match isa.c opcode table — no mnemonic lookup on the hot path. */
static int emit(CcuV1CasmCtx *ctx, uint8_t type, uint16_t code)
{
    if (ccu_v1_casm_emit(ctx, type, code) != 0) {
        return -1;
    }
    return 0;
}

void load_sqeargs_to_gsa(uint16_t gsa, uint16_t sqe)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_LOAD_TYPE, 0x0) != 0) {
        return;
    }
    ctx->inst->load_sqe_gsa.gsa = gsa;
    ctx->inst->load_sqe_gsa.sqe = sqe;
}

void load_sqeargs_to_xn(uint16_t xn, uint16_t sqe)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_LOAD_TYPE, 0x1) != 0) {
        return;
    }
    ctx->inst->load_sqe_xn.xn = xn;
    ctx->inst->load_sqe_xn.sqe = sqe;
}

void load_imd_to_gsa(uint16_t gsa, uint64_t imm)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_LOAD_TYPE, 0x2) != 0) {
        return;
    }
    ctx->inst->load_imd_gsa.gsa = gsa;
    ctx->inst->load_imd_gsa.imm = imm;
}

void load_imd_to_xn(uint16_t xn, uint64_t imm, uint16_t sec)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_LOAD_TYPE, 0x3) != 0) {
        return;
    }
    ctx->inst->load_imd_xn.xn = xn;
    ctx->inst->load_imd_xn.imm = imm;
    ctx->inst->load_imd_xn.sec = sec;
}

void load_gsa_xn(uint16_t gsad, uint16_t gsam, uint16_t xn)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_LOAD_TYPE, 0x4) != 0) {
        return;
    }
    ctx->inst->load_gsa_xn.gsad = gsad;
    ctx->inst->load_gsa_xn.gsam = gsam;
    ctx->inst->load_gsa_xn.xn = xn;
}

void load_gsa_gsa(uint16_t gsad, uint16_t gsam, uint16_t gsan)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_LOAD_TYPE, 0x5) != 0) {
        return;
    }
    ctx->inst->load_gsa_gsa.gsad = gsad;
    ctx->inst->load_gsa_gsa.gsam = gsam;
    ctx->inst->load_gsa_gsa.gsan = gsan;
}

void load_xx(uint16_t xd, uint16_t xm, uint16_t xn)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_LOAD_TYPE, 0x6) != 0) {
        return;
    }
    ctx->inst->load_xx.xd = xd;
    ctx->inst->load_xx.xm = xm;
    ctx->inst->load_xx.xn = xn;
}

void loop(uint16_t start, uint16_t end, uint16_t xn)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx) {
        return;
    }
    (void)ccu_v1_casm_loop(ctx, start, end, xn);
}

void loop_group(uint16_t start_loop, uint16_t xn, uint16_t xm, uint16_t hiperf)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_CTRL_TYPE, 0x1) != 0) {
        return;
    }
    ctx->inst->loop_group.start_loop = start_loop;
    ctx->inst->loop_group.xn = xn;
    ctx->inst->loop_group.xm = xm;
    ctx->inst->loop_group.hiperf = hiperf & 1u;
}

void set_cke(uint16_t clear, uint16_t set_id, uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_CTRL_TYPE, 0x2) != 0) {
        return;
    }
    ctx->inst->set_cke.clear = clear & 1u;
    ctx->inst->set_cke.set_id = set_id;
    ctx->inst->set_cke.set_mask = set_mask;
    ctx->inst->set_cke.wait_id = wait_id;
    ctx->inst->set_cke.wait_mask = wait_mask;
}

void clear_cke(uint16_t clear, uint16_t clear_id, uint16_t clear_mask, uint16_t wait_id, uint16_t wait_mask)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_CTRL_TYPE, 0x4) != 0) {
        return;
    }
    ctx->inst->clear_cke.clear = clear & 1u;
    ctx->inst->clear_cke.clear_id = clear_id;
    ctx->inst->clear_cke.clear_mask = clear_mask;
    ctx->inst->clear_cke.wait_id = wait_id;
    ctx->inst->clear_cke.wait_mask = wait_mask;
}

void jmp(uint16_t dst_xn, uint16_t cond_xn, uint32_t expect)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_CTRL_TYPE, 0x5) != 0) {
        return;
    }
    ctx->inst->jmp.dst_xn = dst_xn;
    ctx->inst->jmp.cond_xn = cond_xn;
    ctx->inst->jmp.expect = expect;
}

void trans_loc_mem_to_loc_ms(uint16_t ms, uint16_t gsa, uint16_t xn, uint16_t len_xn, uint16_t ch, uint16_t clear,
                             uint16_t len_en, uint16_t set_id, uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_TRANS_TYPE, 0x0) != 0) {
        return;
    }
    ctx->inst->trans_mem_to_ms.ms = ms;
    ctx->inst->trans_mem_to_ms.gsa = gsa;
    ctx->inst->trans_mem_to_ms.xn = xn;
    ctx->inst->trans_mem_to_ms.len_xn = len_xn;
    ctx->inst->trans_mem_to_ms.ch = ch;
    ctx->inst->trans_mem_to_ms.clear = clear & 1u;
    ctx->inst->trans_mem_to_ms.len_en = len_en & 1u;
    ctx->inst->trans_mem_to_ms.set_id = set_id;
    ctx->inst->trans_mem_to_ms.set_mask = set_mask;
    ctx->inst->trans_mem_to_ms.wait_id = wait_id;
    ctx->inst->trans_mem_to_ms.wait_mask = wait_mask;
}

void trans_rmt_mem_to_loc_ms(uint16_t ms, uint16_t gsa, uint16_t xn, uint16_t len_xn, uint16_t ch, uint16_t clear,
                             uint16_t len_en, uint16_t set_id, uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_TRANS_TYPE, 0x1) != 0) {
        return;
    }
    ctx->inst->trans_mem_to_ms.ms = ms;
    ctx->inst->trans_mem_to_ms.gsa = gsa;
    ctx->inst->trans_mem_to_ms.xn = xn;
    ctx->inst->trans_mem_to_ms.len_xn = len_xn;
    ctx->inst->trans_mem_to_ms.ch = ch;
    ctx->inst->trans_mem_to_ms.clear = clear & 1u;
    ctx->inst->trans_mem_to_ms.len_en = len_en & 1u;
    ctx->inst->trans_mem_to_ms.set_id = set_id;
    ctx->inst->trans_mem_to_ms.set_mask = set_mask;
    ctx->inst->trans_mem_to_ms.wait_id = wait_id;
    ctx->inst->trans_mem_to_ms.wait_mask = wait_mask;
}

void trans_loc_ms_to_loc_mem(uint16_t gsa, uint16_t xn, uint16_t ms, uint16_t len_xn, uint16_t ch, uint16_t clear,
                             uint16_t len_en, uint16_t set_id, uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_TRANS_TYPE, 0x2) != 0) {
        return;
    }
    ctx->inst->trans_ms_to_mem.gsa = gsa;
    ctx->inst->trans_ms_to_mem.xn = xn;
    ctx->inst->trans_ms_to_mem.ms = ms;
    ctx->inst->trans_ms_to_mem.len_xn = len_xn;
    ctx->inst->trans_ms_to_mem.ch = ch;
    ctx->inst->trans_ms_to_mem.clear = clear & 1u;
    ctx->inst->trans_ms_to_mem.len_en = len_en & 1u;
    ctx->inst->trans_ms_to_mem.set_id = set_id;
    ctx->inst->trans_ms_to_mem.set_mask = set_mask;
    ctx->inst->trans_ms_to_mem.wait_id = wait_id;
    ctx->inst->trans_ms_to_mem.wait_mask = wait_mask;
}

void trans_loc_ms_to_rmt_mem(uint16_t gsa, uint16_t xn, uint16_t ms, uint16_t len_xn, uint16_t ch, uint16_t clear,
                             uint16_t len_en, uint16_t set_id, uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_TRANS_TYPE, 0x3) != 0) {
        return;
    }
    ctx->inst->trans_ms_to_mem.gsa = gsa;
    ctx->inst->trans_ms_to_mem.xn = xn;
    ctx->inst->trans_ms_to_mem.ms = ms;
    ctx->inst->trans_ms_to_mem.len_xn = len_xn;
    ctx->inst->trans_ms_to_mem.ch = ch;
    ctx->inst->trans_ms_to_mem.clear = clear & 1u;
    ctx->inst->trans_ms_to_mem.len_en = len_en & 1u;
    ctx->inst->trans_ms_to_mem.set_id = set_id;
    ctx->inst->trans_ms_to_mem.set_mask = set_mask;
    ctx->inst->trans_ms_to_mem.wait_id = wait_id;
    ctx->inst->trans_ms_to_mem.wait_mask = wait_mask;
}

void trans_rmt_ms_to_loc_mem(uint16_t gsa, uint16_t xn, uint16_t ms, uint16_t len_xn, uint16_t ch, uint16_t clear,
                             uint16_t len_en, uint16_t set_id, uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_TRANS_TYPE, 0x4) != 0) {
        return;
    }
    ctx->inst->trans_ms_to_mem.gsa = gsa;
    ctx->inst->trans_ms_to_mem.xn = xn;
    ctx->inst->trans_ms_to_mem.ms = ms;
    ctx->inst->trans_ms_to_mem.len_xn = len_xn;
    ctx->inst->trans_ms_to_mem.ch = ch;
    ctx->inst->trans_ms_to_mem.clear = clear & 1u;
    ctx->inst->trans_ms_to_mem.len_en = len_en & 1u;
    ctx->inst->trans_ms_to_mem.set_id = set_id;
    ctx->inst->trans_ms_to_mem.set_mask = set_mask;
    ctx->inst->trans_ms_to_mem.wait_id = wait_id;
    ctx->inst->trans_ms_to_mem.wait_mask = wait_mask;
}

void trans_loc_ms_to_loc_ms(uint16_t dst_ms, uint16_t src_ms, uint16_t len_xn, uint16_t ch, uint16_t clear,
                            uint16_t len_en, uint16_t set_id, uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_TRANS_TYPE, 0x5) != 0) {
        return;
    }
    ctx->inst->trans_loc_ms_loc_ms.dst_ms = dst_ms;
    ctx->inst->trans_loc_ms_loc_ms.src_ms = src_ms;
    ctx->inst->trans_loc_ms_loc_ms.len_xn = len_xn;
    ctx->inst->trans_loc_ms_loc_ms.ch = ch;
    ctx->inst->trans_loc_ms_loc_ms.clear = clear & 1u;
    ctx->inst->trans_loc_ms_loc_ms.len_en = len_en & 1u;
    ctx->inst->trans_loc_ms_loc_ms.set_id = set_id;
    ctx->inst->trans_loc_ms_loc_ms.set_mask = set_mask;
    ctx->inst->trans_loc_ms_loc_ms.wait_id = wait_id;
    ctx->inst->trans_loc_ms_loc_ms.wait_mask = wait_mask;
}

void trans_rmt_ms_to_loc_ms(uint16_t loc_ms, uint16_t rmt_ms, uint16_t len_xn, uint16_t ch, uint16_t clear,
                            uint16_t len_en, uint16_t set_id, uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_TRANS_TYPE, 0x6) != 0) {
        return;
    }
    ctx->inst->trans_rmt_ms_loc_ms.loc_ms = loc_ms;
    ctx->inst->trans_rmt_ms_loc_ms.rmt_ms = rmt_ms;
    ctx->inst->trans_rmt_ms_loc_ms.len_xn = len_xn;
    ctx->inst->trans_rmt_ms_loc_ms.ch = ch;
    ctx->inst->trans_rmt_ms_loc_ms.clear = clear & 1u;
    ctx->inst->trans_rmt_ms_loc_ms.len_en = len_en & 1u;
    ctx->inst->trans_rmt_ms_loc_ms.set_id = set_id;
    ctx->inst->trans_rmt_ms_loc_ms.set_mask = set_mask;
    ctx->inst->trans_rmt_ms_loc_ms.wait_id = wait_id;
    ctx->inst->trans_rmt_ms_loc_ms.wait_mask = wait_mask;
}

void trans_loc_ms_to_rmt_ms(uint16_t rmt_ms, uint16_t loc_ms, uint16_t len_xn, uint16_t ch, uint16_t rmt_set_id,
                            uint16_t rmt_set_mask, uint16_t clear, uint16_t len_en, uint16_t set_id, uint16_t set_mask,
                            uint16_t wait_id, uint16_t wait_mask)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_TRANS_TYPE, 0x7) != 0) {
        return;
    }
    ctx->inst->trans_loc_ms_rmt_ms.rmt_ms = rmt_ms;
    ctx->inst->trans_loc_ms_rmt_ms.loc_ms = loc_ms;
    ctx->inst->trans_loc_ms_rmt_ms.len_xn = len_xn;
    ctx->inst->trans_loc_ms_rmt_ms.ch = ch;
    ctx->inst->trans_loc_ms_rmt_ms.rmt_set_id = rmt_set_id;
    ctx->inst->trans_loc_ms_rmt_ms.rmt_set_mask = rmt_set_mask;
    ctx->inst->trans_loc_ms_rmt_ms.clear = clear & 1u;
    ctx->inst->trans_loc_ms_rmt_ms.len_en = len_en & 1u;
    ctx->inst->trans_loc_ms_rmt_ms.set_id = set_id;
    ctx->inst->trans_loc_ms_rmt_ms.set_mask = set_mask;
    ctx->inst->trans_loc_ms_rmt_ms.wait_id = wait_id;
    ctx->inst->trans_loc_ms_rmt_ms.wait_mask = wait_mask;
}

void trans_rmt_mem_to_loc_mem(uint16_t loc_gsa, uint16_t loc_xn, uint16_t rmt_gsa, uint16_t rmt_xn, uint16_t len_xn,
                              uint16_t ch, uint16_t udf, uint16_t reduce_dtype, uint16_t reduce_op, uint16_t clear,
                              uint16_t len_en, uint16_t reduce_en, uint16_t set_id, uint16_t set_mask, uint16_t wait_id,
                              uint16_t wait_mask)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_TRANS_TYPE, 0x8) != 0) {
        return;
    }
    ctx->inst->trans_rmt_mem_loc_mem.loc_gsa = loc_gsa;
    ctx->inst->trans_rmt_mem_loc_mem.loc_xn = loc_xn;
    ctx->inst->trans_rmt_mem_loc_mem.rmt_gsa = rmt_gsa;
    ctx->inst->trans_rmt_mem_loc_mem.rmt_xn = rmt_xn;
    ctx->inst->trans_rmt_mem_loc_mem.len_xn = len_xn;
    ctx->inst->trans_rmt_mem_loc_mem.ch = ch;
    ctx->inst->trans_rmt_mem_loc_mem.udf = udf & 0xFFu;
    ctx->inst->trans_rmt_mem_loc_mem.reduce_dtype = reduce_dtype & 0xFu;
    ctx->inst->trans_rmt_mem_loc_mem.reduce_op = reduce_op & 0xFu;
    ctx->inst->trans_rmt_mem_loc_mem.clear = clear & 1u;
    ctx->inst->trans_rmt_mem_loc_mem.len_en = len_en & 1u;
    ctx->inst->trans_rmt_mem_loc_mem.reduce_en = reduce_en & 1u;
    ctx->inst->trans_rmt_mem_loc_mem.set_id = set_id;
    ctx->inst->trans_rmt_mem_loc_mem.set_mask = set_mask;
    ctx->inst->trans_rmt_mem_loc_mem.wait_id = wait_id;
    ctx->inst->trans_rmt_mem_loc_mem.wait_mask = wait_mask;
}

void trans_loc_mem_to_rmt_mem(uint16_t rmt_gsa, uint16_t rmt_xn, uint16_t loc_gsa, uint16_t loc_xn, uint16_t len_xn,
                              uint16_t ch, uint16_t udf, uint16_t reduce_dtype, uint16_t reduce_op, uint16_t clear,
                              uint16_t len_en, uint16_t reduce_en, uint16_t set_id, uint16_t set_mask, uint16_t wait_id,
                              uint16_t wait_mask)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_TRANS_TYPE, 0x9) != 0) {
        return;
    }
    ctx->inst->trans_loc_mem_rmt_mem.rmt_gsa = rmt_gsa;
    ctx->inst->trans_loc_mem_rmt_mem.rmt_xn = rmt_xn;
    ctx->inst->trans_loc_mem_rmt_mem.loc_gsa = loc_gsa;
    ctx->inst->trans_loc_mem_rmt_mem.loc_xn = loc_xn;
    ctx->inst->trans_loc_mem_rmt_mem.len_xn = len_xn;
    ctx->inst->trans_loc_mem_rmt_mem.ch = ch;
    ctx->inst->trans_loc_mem_rmt_mem.udf = udf & 0xFFu;
    ctx->inst->trans_loc_mem_rmt_mem.reduce_dtype = reduce_dtype & 0xFu;
    ctx->inst->trans_loc_mem_rmt_mem.reduce_op = reduce_op & 0xFu;
    ctx->inst->trans_loc_mem_rmt_mem.clear = clear & 1u;
    ctx->inst->trans_loc_mem_rmt_mem.len_en = len_en & 1u;
    ctx->inst->trans_loc_mem_rmt_mem.reduce_en = reduce_en & 1u;
    ctx->inst->trans_loc_mem_rmt_mem.set_id = set_id;
    ctx->inst->trans_loc_mem_rmt_mem.set_mask = set_mask;
    ctx->inst->trans_loc_mem_rmt_mem.wait_id = wait_id;
    ctx->inst->trans_loc_mem_rmt_mem.wait_mask = wait_mask;
}

void trans_loc_mem_to_loc_mem(uint16_t dst_gsa, uint16_t dst_xn, uint16_t src_gsa, uint16_t src_xn, uint16_t len_xn,
                              uint16_t ch, uint16_t clear, uint16_t len_en, uint16_t set_id, uint16_t set_mask,
                              uint16_t wait_id, uint16_t wait_mask)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_TRANS_TYPE, 0xA) != 0) {
        return;
    }
    ctx->inst->trans_loc_mem_loc_mem.dst_gsa = dst_gsa;
    ctx->inst->trans_loc_mem_loc_mem.dst_xn = dst_xn;
    ctx->inst->trans_loc_mem_loc_mem.src_gsa = src_gsa;
    ctx->inst->trans_loc_mem_loc_mem.src_xn = src_xn;
    ctx->inst->trans_loc_mem_loc_mem.len_xn = len_xn;
    ctx->inst->trans_loc_mem_loc_mem.ch = ch;
    ctx->inst->trans_loc_mem_loc_mem.clear = clear & 1u;
    ctx->inst->trans_loc_mem_loc_mem.len_en = len_en & 1u;
    ctx->inst->trans_loc_mem_loc_mem.set_id = set_id;
    ctx->inst->trans_loc_mem_loc_mem.set_mask = set_mask;
    ctx->inst->trans_loc_mem_loc_mem.wait_id = wait_id;
    ctx->inst->trans_loc_mem_loc_mem.wait_mask = wait_mask;
}

void sync_cke(uint16_t rmt_cke, uint16_t loc_cke, uint16_t loc_mask, uint16_t ch, uint16_t clear, uint16_t set_id,
              uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_TRANS_TYPE, 0xB) != 0) {
        return;
    }
    ctx->inst->sync_cke.rmt_cke = rmt_cke;
    ctx->inst->sync_cke.loc_cke = loc_cke;
    ctx->inst->sync_cke.loc_mask = loc_mask;
    ctx->inst->sync_cke.ch = ch;
    ctx->inst->sync_cke.clear = clear & 1u;
    ctx->inst->sync_cke.set_id = set_id;
    ctx->inst->sync_cke.set_mask = set_mask;
    ctx->inst->sync_cke.wait_id = wait_id;
    ctx->inst->sync_cke.wait_mask = wait_mask;
}

void sync_gsa(uint16_t rmt_gsa, uint16_t loc_gsa, uint16_t ch, uint16_t rmt_set_id, uint16_t rmt_set_mask,
              uint16_t clear, uint16_t set_id, uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_TRANS_TYPE, 0xC) != 0) {
        return;
    }
    ctx->inst->sync_gsa.rmt_gsa = rmt_gsa;
    ctx->inst->sync_gsa.loc_gsa = loc_gsa;
    ctx->inst->sync_gsa.ch = ch;
    ctx->inst->sync_gsa.rmt_set_id = rmt_set_id;
    ctx->inst->sync_gsa.rmt_set_mask = rmt_set_mask;
    ctx->inst->sync_gsa.clear = clear & 1u;
    ctx->inst->sync_gsa.set_id = set_id;
    ctx->inst->sync_gsa.set_mask = set_mask;
    ctx->inst->sync_gsa.wait_id = wait_id;
    ctx->inst->sync_gsa.wait_mask = wait_mask;
}

void sync_xn(uint16_t rmt_xn, uint16_t loc_xn, uint16_t ch, uint16_t rmt_set_id, uint16_t rmt_set_mask, uint16_t clear,
             uint16_t set_id, uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_TRANS_TYPE, 0xD) != 0) {
        return;
    }
    ctx->inst->sync_xn.rmt_xn = rmt_xn;
    ctx->inst->sync_xn.loc_xn = loc_xn;
    ctx->inst->sync_xn.ch = ch;
    ctx->inst->sync_xn.rmt_set_id = rmt_set_id;
    ctx->inst->sync_xn.rmt_set_mask = rmt_set_mask;
    ctx->inst->sync_xn.clear = clear & 1u;
    ctx->inst->sync_xn.set_id = set_id;
    ctx->inst->sync_xn.set_mask = set_mask;
    ctx->inst->sync_xn.wait_id = wait_id;
    ctx->inst->sync_xn.wait_mask = wait_mask;
}

void add(CcuMs ms, uint16_t count, uint16_t cast, uint16_t dtype, uint16_t len_xn, uint16_t clear, uint16_t set_id,
         uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_REDUCE_TYPE, 0x0) != 0) {
        return;
    }
    memcpy(ctx->inst->add.ms, ms.v, sizeof(ms.v));
    ctx->inst->add.len_xn = len_xn;
    ctx->inst->add.clear = clear & 1u;
    ctx->inst->add.count = count & 7u;
    ctx->inst->add.cast = cast & 3u;
    ctx->inst->add.dtype = dtype & 31u;
    ctx->inst->add.set_id = set_id;
    ctx->inst->add.set_mask = set_mask;
    ctx->inst->add.wait_id = wait_id;
    ctx->inst->add.wait_mask = wait_mask;
}

void max(CcuMs ms, uint16_t count, uint16_t dtype, uint16_t len_xn, uint16_t clear, uint16_t set_id, uint16_t set_mask,
         uint16_t wait_id, uint16_t wait_mask)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_REDUCE_TYPE, 0x1) != 0) {
        return;
    }
    memcpy(ctx->inst->maxmin.ms, ms.v, sizeof(ms.v));
    ctx->inst->maxmin.len_xn = len_xn;
    ctx->inst->maxmin.clear = clear & 1u;
    ctx->inst->maxmin.count = count & 7u;
    ctx->inst->maxmin.dtype = dtype & 31u;
    ctx->inst->maxmin.set_id = set_id;
    ctx->inst->maxmin.set_mask = set_mask;
    ctx->inst->maxmin.wait_id = wait_id;
    ctx->inst->maxmin.wait_mask = wait_mask;
}

void min(CcuMs ms, uint16_t count, uint16_t dtype, uint16_t len_xn, uint16_t clear, uint16_t set_id, uint16_t set_mask,
         uint16_t wait_id, uint16_t wait_mask)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx || emit(ctx, CCU_V1_REDUCE_TYPE, 0x2) != 0) {
        return;
    }
    memcpy(ctx->inst->maxmin.ms, ms.v, sizeof(ms.v));
    ctx->inst->maxmin.len_xn = len_xn;
    ctx->inst->maxmin.clear = clear & 1u;
    ctx->inst->maxmin.count = count & 7u;
    ctx->inst->maxmin.dtype = dtype & 31u;
    ctx->inst->maxmin.set_id = set_id;
    ctx->inst->maxmin.set_mask = set_mask;
    ctx->inst->maxmin.wait_id = wait_id;
    ctx->inst->maxmin.wait_mask = wait_mask;
}
