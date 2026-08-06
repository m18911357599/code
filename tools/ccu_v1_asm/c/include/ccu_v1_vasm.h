/**
 * Variable-aware CCU V1 assembler: named resources, auto ID alloc/reuse, metainfo.
 */
#ifndef CCU_V1_VASM_H
#define CCU_V1_VASM_H

#include "ccu_v1_asm.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CCU_V1_RES_XN = 0,
    CCU_V1_RES_GSA,
    CCU_V1_RES_MS,
    CCU_V1_RES_CKE,
    CCU_V1_RES_CH,
    CCU_V1_RES_SQE,
    CCU_V1_RES_COUNT
} CcuV1ResType;

typedef struct {
    uint16_t limits[CCU_V1_RES_COUNT]; /* max IDs per type (exclusive upper bound) */
} CcuV1VasmConfig;

void ccu_v1_vasm_config_default(CcuV1VasmConfig *cfg);

typedef struct {
    char name[64];
    CcuV1ResType type;
    int pinned;          /* 1 if user forced id */
    int16_t id;          /* allocated / pinned id; -1 before alloc */
    int live_start;      /* inclusive instruction index */
    int live_end;        /* inclusive */
    int use_count;
} CcuV1VarInfo;

typedef struct {
    CcuV1VarInfo *vars;
    size_t var_count;
    size_t var_cap;

    CcuV1Program program; /* lowered numeric instructions */
    char *lowered_asm;    /* optional textual dump of lowered asm */
    size_t instr_count;

    uint16_t used_peak[CCU_V1_RES_COUNT];
    CcuV1VasmConfig cfg;
} CcuV1VasmResult;

void ccu_v1_vasm_result_init(CcuV1VasmResult *r);
void ccu_v1_vasm_result_free(CcuV1VasmResult *r);

/**
 * Assemble variable-aware source:
 *   - Declarations:  .xn name  |  .var xn name  |  .xn name = 3 (pinned)
 *   - Operands: identifiers resolve to allocated IDs; numbers stay literals
 *   - Auto-declare on first use when value is an identifier (type from field)
 * Emits binary program + variable table. Optionally fills lowered_asm.
 */
int ccu_v1_vasm_assemble(const char *text, size_t text_len, const CcuV1VasmConfig *cfg,
                         CcuV1VasmResult *out, char *errmsg, size_t errmsg_sz);

/** Write metainfo JSON describing name/type/id/liveness. */
int ccu_v1_vasm_write_metainfo(const CcuV1VasmResult *r, FILE *out);

const char *ccu_v1_res_type_name(CcuV1ResType t);
int ccu_v1_res_type_parse(const char *s, CcuV1ResType *out);

#ifdef __cplusplus
}
#endif

#endif /* CCU_V1_VASM_H */
