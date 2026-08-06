/**
 * Instruction intrinsics: fill packed binary into the current assembler context.
 */
#include "ccu_v1_casm_api.h"

#include <string.h>

static void fail_no_ctx(void)
{
    /* no active context — ignore (host must call begin first) */
}

static CcuV1CasmCtx *require_ctx(void)
{
    CcuV1CasmCtx *ctx = ccu_v1_casm_current();
    if (!ctx || ctx->failed) {
        return NULL;
    }
    return ctx;
}

static void emit_scalars(const char *name, int n, const uint64_t *vals)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx) {
        fail_no_ctx();
        return;
    }
    CcuV1CasmArg args[CCU_V1_CASM_MAX_ARGS];
    for (int i = 0; i < n; ++i) {
        args[i].is_list = 0;
        args[i].scalar = vals[i];
        args[i].list_n = 0;
    }
    if (ccu_v1_casm_call(ctx, name, args, n) != 0) {
        ctx->failed = 1;
    }
}

static void emit_ms_then_scalars(const char *name, CcuMs ms, int nscalar, const uint64_t *vals)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx) {
        fail_no_ctx();
        return;
    }
    CcuV1CasmArg args[CCU_V1_CASM_MAX_ARGS];
    args[0].is_list = 1;
    args[0].list_n = CCU_V1_MS_MAX;
    memcpy(args[0].list, ms.v, sizeof(ms.v));
    for (int i = 0; i < nscalar; ++i) {
        args[1 + i].is_list = 0;
        args[1 + i].scalar = vals[i];
        args[1 + i].list_n = 0;
    }
    if (ccu_v1_casm_call(ctx, name, args, 1 + nscalar) != 0) {
        ctx->failed = 1;
    }
}

void load_sqeargs_to_gsa(uint16_t gsa, uint16_t sqe)
{
    uint64_t v[] = {gsa, sqe};
    emit_scalars("load_sqeargs_to_gsa", 2, v);
}

void load_sqeargs_to_xn(uint16_t xn, uint16_t sqe)
{
    uint64_t v[] = {xn, sqe};
    emit_scalars("load_sqeargs_to_xn", 2, v);
}

void load_imd_to_gsa(uint16_t gsa, uint64_t imm)
{
    uint64_t v[] = {gsa, imm};
    emit_scalars("load_imd_to_gsa", 2, v);
}

void load_imd_to_xn(uint16_t xn, uint64_t imm, uint16_t sec)
{
    uint64_t v[] = {xn, imm, sec};
    emit_scalars("load_imd_to_xn", 3, v);
}

void load_gsa_xn(uint16_t gsad, uint16_t gsam, uint16_t xn)
{
    uint64_t v[] = {gsad, gsam, xn};
    emit_scalars("load_gsa_xn", 3, v);
}

void load_gsa_gsa(uint16_t gsad, uint16_t gsam, uint16_t gsan)
{
    uint64_t v[] = {gsad, gsam, gsan};
    emit_scalars("load_gsa_gsa", 3, v);
}

void load_xx(uint16_t xd, uint16_t xm, uint16_t xn)
{
    uint64_t v[] = {xd, xm, xn};
    emit_scalars("load_xx", 3, v);
}

void loop(uint16_t start, uint16_t end, uint16_t xn)
{
    CcuV1CasmCtx *ctx = require_ctx();
    if (!ctx) {
        fail_no_ctx();
        return;
    }
    /* Fill LOOP binary fields directly into the active context. */
    if (ccu_v1_casm_loop(ctx, start, end, xn) != 0) {
        ctx->failed = 1;
    }
}

void loop_group(uint16_t start_loop, uint16_t xn, uint16_t xm, uint16_t hiperf)
{
    uint64_t v[] = {start_loop, xn, xm, hiperf};
    emit_scalars("loop_group", 4, v);
}

void set_cke(uint16_t clear, uint16_t set_id, uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask)
{
    uint64_t v[] = {clear, set_id, set_mask, wait_id, wait_mask};
    emit_scalars("set_cke", 5, v);
}

void clear_cke(uint16_t clear, uint16_t clear_id, uint16_t clear_mask, uint16_t wait_id, uint16_t wait_mask)
{
    uint64_t v[] = {clear, clear_id, clear_mask, wait_id, wait_mask};
    emit_scalars("clear_cke", 5, v);
}

void jmp(uint16_t dst_xn, uint16_t cond_xn, uint32_t expect)
{
    uint64_t v[] = {dst_xn, cond_xn, expect};
    emit_scalars("jmp", 3, v);
}

void trans_loc_mem_to_loc_ms(uint16_t ms, uint16_t gsa, uint16_t xn, uint16_t len_xn, uint16_t ch, uint16_t clear,
                             uint16_t len_en, uint16_t set_id, uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask)
{
    uint64_t v[] = {ms, gsa, xn, len_xn, ch, clear, len_en, set_id, set_mask, wait_id, wait_mask};
    emit_scalars("trans_loc_mem_to_loc_ms", 11, v);
}

void trans_rmt_mem_to_loc_ms(uint16_t ms, uint16_t gsa, uint16_t xn, uint16_t len_xn, uint16_t ch, uint16_t clear,
                             uint16_t len_en, uint16_t set_id, uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask)
{
    uint64_t v[] = {ms, gsa, xn, len_xn, ch, clear, len_en, set_id, set_mask, wait_id, wait_mask};
    emit_scalars("trans_rmt_mem_to_loc_ms", 11, v);
}

void trans_loc_ms_to_loc_mem(uint16_t gsa, uint16_t xn, uint16_t ms, uint16_t len_xn, uint16_t ch, uint16_t clear,
                             uint16_t len_en, uint16_t set_id, uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask)
{
    uint64_t v[] = {gsa, xn, ms, len_xn, ch, clear, len_en, set_id, set_mask, wait_id, wait_mask};
    emit_scalars("trans_loc_ms_to_loc_mem", 11, v);
}

void trans_loc_ms_to_rmt_mem(uint16_t gsa, uint16_t xn, uint16_t ms, uint16_t len_xn, uint16_t ch, uint16_t clear,
                             uint16_t len_en, uint16_t set_id, uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask)
{
    uint64_t v[] = {gsa, xn, ms, len_xn, ch, clear, len_en, set_id, set_mask, wait_id, wait_mask};
    emit_scalars("trans_loc_ms_to_rmt_mem", 11, v);
}

void trans_rmt_ms_to_loc_mem(uint16_t gsa, uint16_t xn, uint16_t ms, uint16_t len_xn, uint16_t ch, uint16_t clear,
                             uint16_t len_en, uint16_t set_id, uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask)
{
    uint64_t v[] = {gsa, xn, ms, len_xn, ch, clear, len_en, set_id, set_mask, wait_id, wait_mask};
    emit_scalars("trans_rmt_ms_to_loc_mem", 11, v);
}

void trans_loc_ms_to_loc_ms(uint16_t dst_ms, uint16_t src_ms, uint16_t len_xn, uint16_t ch, uint16_t clear,
                            uint16_t len_en, uint16_t set_id, uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask)
{
    uint64_t v[] = {dst_ms, src_ms, len_xn, ch, clear, len_en, set_id, set_mask, wait_id, wait_mask};
    emit_scalars("trans_loc_ms_to_loc_ms", 10, v);
}

void trans_rmt_ms_to_loc_ms(uint16_t loc_ms, uint16_t rmt_ms, uint16_t len_xn, uint16_t ch, uint16_t clear,
                            uint16_t len_en, uint16_t set_id, uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask)
{
    uint64_t v[] = {loc_ms, rmt_ms, len_xn, ch, clear, len_en, set_id, set_mask, wait_id, wait_mask};
    emit_scalars("trans_rmt_ms_to_loc_ms", 10, v);
}

void trans_loc_ms_to_rmt_ms(uint16_t rmt_ms, uint16_t loc_ms, uint16_t len_xn, uint16_t ch, uint16_t rmt_set_id,
                            uint16_t rmt_set_mask, uint16_t clear, uint16_t len_en, uint16_t set_id, uint16_t set_mask,
                            uint16_t wait_id, uint16_t wait_mask)
{
    uint64_t v[] = {rmt_ms, loc_ms, len_xn, ch, rmt_set_id, rmt_set_mask, clear, len_en, set_id, set_mask, wait_id,
                    wait_mask};
    emit_scalars("trans_loc_ms_to_rmt_ms", 12, v);
}

void trans_rmt_mem_to_loc_mem(uint16_t loc_gsa, uint16_t loc_xn, uint16_t rmt_gsa, uint16_t rmt_xn, uint16_t len_xn,
                              uint16_t ch, uint16_t udf, uint16_t reduce_dtype, uint16_t reduce_op, uint16_t clear,
                              uint16_t len_en, uint16_t reduce_en, uint16_t set_id, uint16_t set_mask, uint16_t wait_id,
                              uint16_t wait_mask)
{
    uint64_t v[] = {loc_gsa, loc_xn, rmt_gsa, rmt_xn, len_xn, ch, udf, reduce_dtype, reduce_op, clear, len_en,
                    reduce_en, set_id, set_mask, wait_id, wait_mask};
    emit_scalars("trans_rmt_mem_to_loc_mem", 16, v);
}

void trans_loc_mem_to_rmt_mem(uint16_t rmt_gsa, uint16_t rmt_xn, uint16_t loc_gsa, uint16_t loc_xn, uint16_t len_xn,
                              uint16_t ch, uint16_t udf, uint16_t reduce_dtype, uint16_t reduce_op, uint16_t clear,
                              uint16_t len_en, uint16_t reduce_en, uint16_t set_id, uint16_t set_mask, uint16_t wait_id,
                              uint16_t wait_mask)
{
    uint64_t v[] = {rmt_gsa, rmt_xn, loc_gsa, loc_xn, len_xn, ch, udf, reduce_dtype, reduce_op, clear, len_en,
                    reduce_en, set_id, set_mask, wait_id, wait_mask};
    emit_scalars("trans_loc_mem_to_rmt_mem", 16, v);
}

void trans_loc_mem_to_loc_mem(uint16_t dst_gsa, uint16_t dst_xn, uint16_t src_gsa, uint16_t src_xn, uint16_t len_xn,
                              uint16_t ch, uint16_t clear, uint16_t len_en, uint16_t set_id, uint16_t set_mask,
                              uint16_t wait_id, uint16_t wait_mask)
{
    uint64_t v[] = {dst_gsa, dst_xn, src_gsa, src_xn, len_xn, ch, clear, len_en, set_id, set_mask, wait_id, wait_mask};
    emit_scalars("trans_loc_mem_to_loc_mem", 12, v);
}

void sync_cke(uint16_t rmt_cke, uint16_t loc_cke, uint16_t loc_mask, uint16_t ch, uint16_t clear, uint16_t set_id,
              uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask)
{
    uint64_t v[] = {rmt_cke, loc_cke, loc_mask, ch, clear, set_id, set_mask, wait_id, wait_mask};
    emit_scalars("sync_cke", 9, v);
}

void sync_gsa(uint16_t rmt_gsa, uint16_t loc_gsa, uint16_t ch, uint16_t rmt_set_id, uint16_t rmt_set_mask,
              uint16_t clear, uint16_t set_id, uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask)
{
    uint64_t v[] = {rmt_gsa, loc_gsa, ch, rmt_set_id, rmt_set_mask, clear, set_id, set_mask, wait_id, wait_mask};
    emit_scalars("sync_gsa", 10, v);
}

void sync_xn(uint16_t rmt_xn, uint16_t loc_xn, uint16_t ch, uint16_t rmt_set_id, uint16_t rmt_set_mask, uint16_t clear,
             uint16_t set_id, uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask)
{
    uint64_t v[] = {rmt_xn, loc_xn, ch, rmt_set_id, rmt_set_mask, clear, set_id, set_mask, wait_id, wait_mask};
    emit_scalars("sync_xn", 10, v);
}

void add(CcuMs ms, uint16_t count, uint16_t cast, uint16_t dtype, uint16_t len_xn, uint16_t clear, uint16_t set_id,
         uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask)
{
    uint64_t v[] = {count, cast, dtype, len_xn, clear, set_id, set_mask, wait_id, wait_mask};
    emit_ms_then_scalars("add", ms, 9, v);
}

void max(CcuMs ms, uint16_t count, uint16_t dtype, uint16_t len_xn, uint16_t clear, uint16_t set_id, uint16_t set_mask,
         uint16_t wait_id, uint16_t wait_mask)
{
    uint64_t v[] = {count, dtype, len_xn, clear, set_id, set_mask, wait_id, wait_mask};
    emit_ms_then_scalars("max", ms, 8, v);
}

void min(CcuMs ms, uint16_t count, uint16_t dtype, uint16_t len_xn, uint16_t clear, uint16_t set_id, uint16_t set_mask,
         uint16_t wait_id, uint16_t wait_mask)
{
    uint64_t v[] = {count, dtype, len_xn, clear, set_id, set_mask, wait_id, wait_mask};
    emit_ms_then_scalars("min", ms, 8, v);
}
