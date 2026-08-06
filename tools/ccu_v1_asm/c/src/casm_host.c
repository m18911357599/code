/**
 * Host for C-style CCU programs (e.g. loop_main.c).
 *
 * Flow:
 *   1. create context
 *   2. begin (install context)  — before calling _entry
 *   3. call user _entry()       — each instruction fills binary into context
 *   4. end
 *   5. write binary directly to file
 */
#include "ccu_v1_casm_api.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char **argv)
{
    const char *out_path = NULL;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) {
            out_path = argv[++i];
        } else {
            fprintf(stderr, "Usage: %s -o <out.bin>\n", argv[0]);
            return 2;
        }
    }
    if (!out_path) {
        fprintf(stderr, "Usage: %s -o <out.bin>\n", argv[0]);
        return 2;
    }

    CcuV1CasmCtx ctx;
    ccu_v1_casm_init(&ctx);

    /* create context → begin → call _entry (instr write ctx->inst) → end */
    ccu_v1_casm_run(&ctx, _entry);

    /* write packed instruction bytes directly to file */
    if (ccu_v1_casm_write_file(&ctx, out_path) != 0) {
        fprintf(stderr, "error: cannot write %s\n", out_path);
        ccu_v1_casm_free(&ctx);
        return 2;
    }

    printf("casm_host: %zu instructions -> %s (%zu bytes)\n", ctx.program.count, out_path,
           ctx.program.count * (size_t)CCU_V1_INSTR_SIZE);
    ccu_v1_casm_free(&ctx);
    return 0;
}
