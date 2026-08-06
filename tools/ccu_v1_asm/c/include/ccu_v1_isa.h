/**
 * CCU V1 ISA — packed layouts matching hcomm CcuInstr (32B, LE, #pragma pack(1)).
 */
#ifndef CCU_V1_ISA_H
#define CCU_V1_ISA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CCU_V1_INSTR_SIZE   32
#define CCU_V1_PAYLOAD_SIZE 30
#define CCU_V1_MS_MAX       8

#define CCU_V1_LOAD_TYPE   0x0
#define CCU_V1_CTRL_TYPE   0x1
#define CCU_V1_TRANS_TYPE  0x2
#define CCU_V1_REDUCE_TYPE 0x3

#pragma pack(push, 1)

typedef union {
    struct {
        uint16_t code : 11;
        uint16_t type : 4;
        uint16_t reserved : 1;
    };
    uint16_t raw;
} CcuV1Header;

/* ---- LOAD ---- */
typedef struct {
    uint16_t gsa;
    uint16_t sqe;
    uint16_t reserved[13];
} CcuV1LoadSqeToGsa;

typedef struct {
    uint16_t xn;
    uint16_t sqe;
    uint16_t reserved[13];
} CcuV1LoadSqeToXn;

typedef struct {
    uint16_t gsa;
    uint64_t imm;
    uint16_t reserved[10];
} CcuV1LoadImdToGsa;

typedef struct {
    uint16_t xn;
    uint64_t imm;
    uint16_t sec;
    uint16_t reserved[9];
} CcuV1LoadImdToXn;

typedef struct {
    uint16_t gsad;
    uint16_t gsam;
    uint16_t xn;
    uint16_t reserved[12];
} CcuV1LoadGsaXn;

typedef struct {
    uint16_t gsad;
    uint16_t gsam;
    uint16_t gsan;
    uint16_t reserved[12];
} CcuV1LoadGsaGsa;

typedef struct {
    uint16_t xd;
    uint16_t xm;
    uint16_t xn;
    uint16_t reserved[12];
} CcuV1LoadXx;

/* ---- CTRL ---- */
typedef struct {
    uint16_t start;
    uint16_t end;
    uint16_t xn;
    uint16_t reserved[12];
} CcuV1Loop;

typedef struct {
    uint16_t start_loop;
    uint16_t xn;
    uint16_t xm;
    uint16_t hiperf : 1;
    uint16_t reserved1 : 15;
    uint16_t reserved[11];
} CcuV1LoopGroup;

typedef struct {
    uint16_t clear : 1;
    uint16_t reserved1 : 15;
    uint16_t set_id;
    uint16_t set_mask;
    uint16_t wait_id;
    uint16_t wait_mask;
    uint16_t reserved[10];
} CcuV1SetCke;

typedef struct {
    uint16_t clear : 1;
    uint16_t reserved1 : 15;
    uint16_t clear_id;
    uint16_t clear_mask;
    uint16_t wait_id;
    uint16_t wait_mask;
    uint16_t reserved[10];
} CcuV1ClearCke;

typedef struct {
    uint16_t dst_xn;
    uint16_t cond_xn;
    uint64_t expect;
    uint16_t reserved[9];
} CcuV1Jmp;

/* ---- TRANS common tails ---- */
typedef struct {
    uint16_t ms;
    uint16_t gsa;
    uint16_t xn;
    uint16_t len_xn;
    uint16_t ch;
    uint16_t reserved1[5];
    uint16_t clear : 1;
    uint16_t len_en : 1;
    uint16_t reserved : 14;
    uint16_t set_id;
    uint16_t set_mask;
    uint16_t wait_id;
    uint16_t wait_mask;
} CcuV1TransMemToMs; /* LocMem->LocMS and RmtMem->LocMS */

typedef struct {
    uint16_t gsa;
    uint16_t xn;
    uint16_t ms;
    uint16_t len_xn;
    uint16_t ch;
    uint16_t reserved1[5];
    uint16_t clear : 1;
    uint16_t len_en : 1;
    uint16_t reserved : 14;
    uint16_t set_id;
    uint16_t set_mask;
    uint16_t wait_id;
    uint16_t wait_mask;
} CcuV1TransMsToMem; /* LocMS->LocMem / LocMS->RmtMem / RmtMS->LocMem */

typedef struct {
    uint16_t dst_ms;
    uint16_t src_ms;
    uint16_t len_xn;
    uint16_t ch;
    uint16_t reserved1[6];
    uint16_t clear : 1;
    uint16_t len_en : 1;
    uint16_t reserved : 14;
    uint16_t set_id;
    uint16_t set_mask;
    uint16_t wait_id;
    uint16_t wait_mask;
} CcuV1TransLocMsToLocMs;

typedef struct {
    uint16_t loc_ms;
    uint16_t rmt_ms;
    uint16_t len_xn;
    uint16_t ch;
    uint16_t reserved1[6];
    uint16_t clear : 1;
    uint16_t len_en : 1;
    uint16_t reserved : 14;
    uint16_t set_id;
    uint16_t set_mask;
    uint16_t wait_id;
    uint16_t wait_mask;
} CcuV1TransRmtMsToLocMs;

typedef struct {
    uint16_t rmt_ms;
    uint16_t loc_ms;
    uint16_t len_xn;
    uint16_t ch;
    uint16_t rmt_set_id;
    uint16_t rmt_set_mask;
    uint16_t reserved1[4];
    uint16_t clear : 1;
    uint16_t len_en : 1;
    uint16_t reserved : 14;
    uint16_t set_id;
    uint16_t set_mask;
    uint16_t wait_id;
    uint16_t wait_mask;
} CcuV1TransLocMsToRmtMs;

typedef struct {
    uint16_t loc_gsa;
    uint16_t loc_xn;
    uint16_t rmt_gsa;
    uint16_t rmt_xn;
    uint16_t len_xn;
    uint16_t ch;
    uint16_t udf : 8;
    uint16_t reduce_dtype : 4;
    uint16_t reduce_op : 4;
    uint16_t reserved[3];
    uint16_t clear : 1;
    uint16_t len_en : 1;
    uint16_t reduce_en : 1;
    uint16_t reserved1 : 13;
    uint16_t set_id;
    uint16_t set_mask;
    uint16_t wait_id;
    uint16_t wait_mask;
} CcuV1TransRmtMemToLocMem;

typedef struct {
    uint16_t rmt_gsa;
    uint16_t rmt_xn;
    uint16_t loc_gsa;
    uint16_t loc_xn;
    uint16_t len_xn;
    uint16_t ch;
    uint16_t udf : 8;
    uint16_t reduce_dtype : 4;
    uint16_t reduce_op : 4;
    uint16_t reserved[3];
    uint16_t clear : 1;
    uint16_t len_en : 1;
    uint16_t reduce_en : 1;
    uint16_t reserved1 : 13;
    uint16_t set_id;
    uint16_t set_mask;
    uint16_t wait_id;
    uint16_t wait_mask;
} CcuV1TransLocMemToRmtMem;

typedef struct {
    uint16_t dst_gsa;
    uint16_t dst_xn;
    uint16_t src_gsa;
    uint16_t src_xn;
    uint16_t len_xn;
    uint16_t ch;
    uint16_t reserved1[4];
    uint16_t clear : 1;
    uint16_t len_en : 1;
    uint16_t reserved : 14;
    uint16_t set_id;
    uint16_t set_mask;
    uint16_t wait_id;
    uint16_t wait_mask;
} CcuV1TransLocMemToLocMem;

typedef struct {
    uint16_t rmt_cke;
    uint16_t loc_cke;
    uint16_t loc_mask;
    uint16_t ch;
    uint16_t reserved[6];
    uint16_t clear : 1;
    uint16_t reserved1 : 15;
    uint16_t set_id;
    uint16_t set_mask;
    uint16_t wait_id;
    uint16_t wait_mask;
} CcuV1SyncCke;

typedef struct {
    uint16_t rmt_gsa;
    uint16_t loc_gsa;
    uint16_t reserved2;
    uint16_t ch;
    uint16_t rmt_set_id;
    uint16_t rmt_set_mask;
    uint16_t reserved[4];
    uint16_t clear : 1;
    uint16_t reserved1 : 15;
    uint16_t set_id;
    uint16_t set_mask;
    uint16_t wait_id;
    uint16_t wait_mask;
} CcuV1SyncGsa;

typedef struct {
    uint16_t rmt_xn;
    uint16_t loc_xn;
    uint16_t reserved2;
    uint16_t ch;
    uint16_t rmt_set_id;
    uint16_t rmt_set_mask;
    uint16_t reserved[4];
    uint16_t clear : 1;
    uint16_t reserved1 : 15;
    uint16_t set_id;
    uint16_t set_mask;
    uint16_t wait_id;
    uint16_t wait_mask;
} CcuV1SyncXn;

typedef struct {
    uint16_t ms[CCU_V1_MS_MAX];
    uint16_t len_xn;
    uint16_t reserved;
    uint16_t clear : 1;
    uint16_t count : 3;
    uint16_t cast : 2;
    uint16_t reserved1 : 5;
    uint16_t dtype : 5;
    uint16_t set_id;
    uint16_t set_mask;
    uint16_t wait_id;
    uint16_t wait_mask;
} CcuV1Add;

typedef struct {
    uint16_t ms[CCU_V1_MS_MAX];
    uint16_t len_xn;
    uint16_t reserved;
    uint16_t clear : 1;
    uint16_t count : 3;
    uint16_t reserved1 : 7;
    uint16_t dtype : 5;
    uint16_t set_id;
    uint16_t set_mask;
    uint16_t wait_id;
    uint16_t wait_mask;
} CcuV1MaxMin;

typedef struct {
    CcuV1Header header;
    union {
        uint8_t raw[CCU_V1_PAYLOAD_SIZE];
        CcuV1LoadSqeToGsa load_sqe_gsa;
        CcuV1LoadSqeToXn load_sqe_xn;
        CcuV1LoadImdToGsa load_imd_gsa;
        CcuV1LoadImdToXn load_imd_xn;
        CcuV1LoadGsaXn load_gsa_xn;
        CcuV1LoadGsaGsa load_gsa_gsa;
        CcuV1LoadXx load_xx;
        CcuV1Loop loop;
        CcuV1LoopGroup loop_group;
        CcuV1SetCke set_cke;
        CcuV1ClearCke clear_cke;
        CcuV1Jmp jmp;
        CcuV1TransMemToMs trans_mem_to_ms;
        CcuV1TransMsToMem trans_ms_to_mem;
        CcuV1TransLocMsToLocMs trans_loc_ms_loc_ms;
        CcuV1TransRmtMsToLocMs trans_rmt_ms_loc_ms;
        CcuV1TransLocMsToRmtMs trans_loc_ms_rmt_ms;
        CcuV1TransRmtMemToLocMem trans_rmt_mem_loc_mem;
        CcuV1TransLocMemToRmtMem trans_loc_mem_rmt_mem;
        CcuV1TransLocMemToLocMem trans_loc_mem_loc_mem;
        CcuV1SyncCke sync_cke;
        CcuV1SyncGsa sync_gsa;
        CcuV1SyncXn sync_xn;
        CcuV1Add add;
        CcuV1MaxMin maxmin;
    };
} CcuV1Instr;

#pragma pack(pop)

_Static_assert(sizeof(CcuV1Instr) == CCU_V1_INSTR_SIZE, "CcuV1Instr must be 32 bytes");
_Static_assert(sizeof(CcuV1Header) == 2, "header must be 2 bytes");
_Static_assert(sizeof(CcuV1LoadImdToGsa) == CCU_V1_PAYLOAD_SIZE, "LoadImdToGsa payload");
_Static_assert(sizeof(CcuV1Add) == CCU_V1_PAYLOAD_SIZE, "Add payload");
_Static_assert(sizeof(CcuV1TransLocMemToRmtMem) == CCU_V1_PAYLOAD_SIZE, "LocMemToRmtMem payload");

typedef enum {
    CCU_V1_OP_LOAD_SQEARGS_TO_GSA = 0,
    CCU_V1_OP_LOAD_SQEARGS_TO_XN,
    CCU_V1_OP_LOAD_IMD_TO_GSA,
    CCU_V1_OP_LOAD_IMD_TO_XN,
    CCU_V1_OP_LOAD_GSA_XN,
    CCU_V1_OP_LOAD_GSA_GSA,
    CCU_V1_OP_LOAD_XX,
    CCU_V1_OP_LOOP,
    CCU_V1_OP_LOOP_GROUP,
    CCU_V1_OP_SET_CKE,
    CCU_V1_OP_CLEAR_CKE,
    CCU_V1_OP_JMP,
    CCU_V1_OP_TRANS_LOC_MEM_TO_LOC_MS,
    CCU_V1_OP_TRANS_RMT_MEM_TO_LOC_MS,
    CCU_V1_OP_TRANS_LOC_MS_TO_LOC_MEM,
    CCU_V1_OP_TRANS_LOC_MS_TO_RMT_MEM,
    CCU_V1_OP_TRANS_RMT_MS_TO_LOC_MEM,
    CCU_V1_OP_TRANS_LOC_MS_TO_LOC_MS,
    CCU_V1_OP_TRANS_RMT_MS_TO_LOC_MS,
    CCU_V1_OP_TRANS_LOC_MS_TO_RMT_MS,
    CCU_V1_OP_TRANS_RMT_MEM_TO_LOC_MEM,
    CCU_V1_OP_TRANS_LOC_MEM_TO_RMT_MEM,
    CCU_V1_OP_TRANS_LOC_MEM_TO_LOC_MEM,
    CCU_V1_OP_SYNC_CKE,
    CCU_V1_OP_SYNC_GSA,
    CCU_V1_OP_SYNC_XN,
    CCU_V1_OP_ADD,
    CCU_V1_OP_MAX,
    CCU_V1_OP_MIN,
    CCU_V1_OP_COUNT
} CcuV1OpcodeId;

typedef struct {
    const char *mnemonic;
    uint8_t type;
    uint16_t code;
    CcuV1OpcodeId id;
    const char *const *operands; /* positional operand names, NULL-terminated */
    int nop;                     /* operand count */
} CcuV1OpcodeDesc;

const CcuV1OpcodeDesc *ccu_v1_opcodes(void);
size_t ccu_v1_opcode_count(void);
const CcuV1OpcodeDesc *ccu_v1_lookup_mnemonic(const char *mnem);
const CcuV1OpcodeDesc *ccu_v1_lookup_opcode(uint8_t type, uint16_t code);

static inline uint16_t ccu_v1_make_header(uint8_t type, uint16_t code)
{
    return (uint16_t)(((uint16_t)type << 11) | (code & 0x7FFu));
}

static inline void ccu_v1_split_header(uint16_t raw, uint8_t *type, uint16_t *code)
{
    *code = (uint16_t)(raw & 0x7FFu);
    *type = (uint8_t)((raw >> 11) & 0xFu);
}

#ifdef __cplusplus
}
#endif

#endif /* CCU_V1_ISA_H */
