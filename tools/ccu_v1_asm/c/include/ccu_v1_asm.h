/**
 * High-performance asm text <-> CcuV1Instr conversion.
 * Parser scans in-place; formatter writes into caller buffer.
 */
#ifndef CCU_V1_ASM_H
#define CCU_V1_ASM_H

#include "ccu_v1_isa.h"

#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    CcuV1Instr *items;
    size_t count;
    size_t capacity;
} CcuV1Program;

void ccu_v1_program_init(CcuV1Program *p);
void ccu_v1_program_free(CcuV1Program *p);
int ccu_v1_program_reserve(CcuV1Program *p, size_t n);

/* Parse assembly text into program. Returns 0 on success, -1 on error (errmsg filled). */
int ccu_v1_assemble_text(const char *text, size_t text_len, CcuV1Program *out, char *errmsg, size_t errmsg_sz);

/* Format one instruction (canonical operand order). Returns bytes written (excl NUL) or -1. */
int ccu_v1_format_instr(const CcuV1Instr *instr, char *buf, size_t buf_sz);

/* Disassemble program into FILE. */
int ccu_v1_disassemble_program(const CcuV1Program *prog, FILE *out);

/* Encode/decode binary blob (must be multiple of 32). */
int ccu_v1_program_from_binary(const uint8_t *data, size_t len, CcuV1Program *out, char *errmsg, size_t errmsg_sz);
int ccu_v1_program_to_binary(const CcuV1Program *prog, uint8_t **out_data, size_t *out_len);

/* Semantic compare: 0 if equal, -1 if differ. */
int ccu_v1_program_semantic_eq(const CcuV1Program *a, const CcuV1Program *b);

/* Finalize header after filling payload via union fields. */
void ccu_v1_instr_set_opcode(CcuV1Instr *instr, const CcuV1OpcodeDesc *desc);

#ifdef __cplusplus
}
#endif

#endif /* CCU_V1_ASM_H */
