/**
 * Intrinsics for CCU assembly programs (e.g. examples/loop_main.c).
 *
 * Usage:
 *   #include "ccu_v1_casm_api.h"
 *   void main(void) { loop(0, 10, 11); ... }
 *
 * Host (casm_host) does:
 *   create ctx → begin → call main (as ccu_user_main) → end → write_file
 */
#ifndef CCU_V1_CASM_API_H
#define CCU_V1_CASM_API_H

#include "ccu_v1_casm.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint16_t v[CCU_V1_MS_MAX];
} CcuMs;

#define MS(...) ((CcuMs){.v = {__VA_ARGS__}})

/* User entry: compile loop_main.c with -Dmain=ccu_user_main */
void ccu_user_main(void);

/* ---- LOAD ---- */
void load_sqeargs_to_gsa(uint16_t gsa, uint16_t sqe);
void load_sqeargs_to_xn(uint16_t xn, uint16_t sqe);
void load_imd_to_gsa(uint16_t gsa, uint64_t imm);
void load_imd_to_xn(uint16_t xn, uint64_t imm, uint16_t sec);
void load_gsa_xn(uint16_t gsad, uint16_t gsam, uint16_t xn);
void load_gsa_gsa(uint16_t gsad, uint16_t gsam, uint16_t gsan);
void load_xx(uint16_t xd, uint16_t xm, uint16_t xn);

/* ---- CTRL ---- */
void loop(uint16_t start, uint16_t end, uint16_t xn);
void loop_group(uint16_t start_loop, uint16_t xn, uint16_t xm, uint16_t hiperf);
void set_cke(uint16_t clear, uint16_t set_id, uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask);
void clear_cke(uint16_t clear, uint16_t clear_id, uint16_t clear_mask, uint16_t wait_id, uint16_t wait_mask);
void jmp(uint16_t dst_xn, uint16_t cond_xn, uint32_t expect);

/* ---- TRANS ---- */
void trans_loc_mem_to_loc_ms(uint16_t ms, uint16_t gsa, uint16_t xn, uint16_t len_xn, uint16_t ch, uint16_t clear,
                             uint16_t len_en, uint16_t set_id, uint16_t set_mask, uint16_t wait_id,
                             uint16_t wait_mask);
void trans_rmt_mem_to_loc_ms(uint16_t ms, uint16_t gsa, uint16_t xn, uint16_t len_xn, uint16_t ch, uint16_t clear,
                             uint16_t len_en, uint16_t set_id, uint16_t set_mask, uint16_t wait_id,
                             uint16_t wait_mask);
void trans_loc_ms_to_loc_mem(uint16_t gsa, uint16_t xn, uint16_t ms, uint16_t len_xn, uint16_t ch, uint16_t clear,
                             uint16_t len_en, uint16_t set_id, uint16_t set_mask, uint16_t wait_id,
                             uint16_t wait_mask);
void trans_loc_ms_to_rmt_mem(uint16_t gsa, uint16_t xn, uint16_t ms, uint16_t len_xn, uint16_t ch, uint16_t clear,
                             uint16_t len_en, uint16_t set_id, uint16_t set_mask, uint16_t wait_id,
                             uint16_t wait_mask);
void trans_rmt_ms_to_loc_mem(uint16_t gsa, uint16_t xn, uint16_t ms, uint16_t len_xn, uint16_t ch, uint16_t clear,
                             uint16_t len_en, uint16_t set_id, uint16_t set_mask, uint16_t wait_id,
                             uint16_t wait_mask);
void trans_loc_ms_to_loc_ms(uint16_t dst_ms, uint16_t src_ms, uint16_t len_xn, uint16_t ch, uint16_t clear,
                            uint16_t len_en, uint16_t set_id, uint16_t set_mask, uint16_t wait_id,
                            uint16_t wait_mask);
void trans_rmt_ms_to_loc_ms(uint16_t loc_ms, uint16_t rmt_ms, uint16_t len_xn, uint16_t ch, uint16_t clear,
                            uint16_t len_en, uint16_t set_id, uint16_t set_mask, uint16_t wait_id,
                            uint16_t wait_mask);
void trans_loc_ms_to_rmt_ms(uint16_t rmt_ms, uint16_t loc_ms, uint16_t len_xn, uint16_t ch, uint16_t rmt_set_id,
                            uint16_t rmt_set_mask, uint16_t clear, uint16_t len_en, uint16_t set_id,
                            uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask);
void trans_rmt_mem_to_loc_mem(uint16_t loc_gsa, uint16_t loc_xn, uint16_t rmt_gsa, uint16_t rmt_xn, uint16_t len_xn,
                              uint16_t ch, uint16_t udf, uint16_t reduce_dtype, uint16_t reduce_op, uint16_t clear,
                              uint16_t len_en, uint16_t reduce_en, uint16_t set_id, uint16_t set_mask,
                              uint16_t wait_id, uint16_t wait_mask);
void trans_loc_mem_to_rmt_mem(uint16_t rmt_gsa, uint16_t rmt_xn, uint16_t loc_gsa, uint16_t loc_xn, uint16_t len_xn,
                              uint16_t ch, uint16_t udf, uint16_t reduce_dtype, uint16_t reduce_op, uint16_t clear,
                              uint16_t len_en, uint16_t reduce_en, uint16_t set_id, uint16_t set_mask,
                              uint16_t wait_id, uint16_t wait_mask);
void trans_loc_mem_to_loc_mem(uint16_t dst_gsa, uint16_t dst_xn, uint16_t src_gsa, uint16_t src_xn, uint16_t len_xn,
                              uint16_t ch, uint16_t clear, uint16_t len_en, uint16_t set_id, uint16_t set_mask,
                              uint16_t wait_id, uint16_t wait_mask);

/* ---- SYNC ---- */
void sync_cke(uint16_t rmt_cke, uint16_t loc_cke, uint16_t loc_mask, uint16_t ch, uint16_t clear, uint16_t set_id,
              uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask);
void sync_gsa(uint16_t rmt_gsa, uint16_t loc_gsa, uint16_t ch, uint16_t rmt_set_id, uint16_t rmt_set_mask,
              uint16_t clear, uint16_t set_id, uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask);
void sync_xn(uint16_t rmt_xn, uint16_t loc_xn, uint16_t ch, uint16_t rmt_set_id, uint16_t rmt_set_mask, uint16_t clear,
             uint16_t set_id, uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask);

/* ---- REDUCE ---- */
void add(CcuMs ms, uint16_t count, uint16_t cast, uint16_t dtype, uint16_t len_xn, uint16_t clear, uint16_t set_id,
         uint16_t set_mask, uint16_t wait_id, uint16_t wait_mask);
void max(CcuMs ms, uint16_t count, uint16_t dtype, uint16_t len_xn, uint16_t clear, uint16_t set_id, uint16_t set_mask,
         uint16_t wait_id, uint16_t wait_mask);
void min(CcuMs ms, uint16_t count, uint16_t dtype, uint16_t len_xn, uint16_t clear, uint16_t set_id, uint16_t set_mask,
         uint16_t wait_id, uint16_t wait_mask);

#ifdef __cplusplus
}
#endif

#endif /* CCU_V1_CASM_API_H */
