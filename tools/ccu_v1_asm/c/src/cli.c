/**
 * ccu_v1_asm — high-performance CCU V1 assembler / disassembler CLI.
 *
 * Usage:
 *   ccu_v1_asm assemble   <in.s> -o <out.bin>
 *   ccu_v1_asm disassemble <in.bin> -o <out.s>
 *   ccu_v1_asm verify     <in.s> [-w workdir]
 */
#include "ccu_v1_asm.h"
#include "ccu_v1_vasm.h"
#include "ccu_v1_casm.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

static int read_file(const char *path, uint8_t **data, size_t *len)
{
#ifdef __linux__
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return -1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return -1;
    }
    size_t sz = (size_t)st.st_size;
    if (sz == 0) {
        close(fd);
        *data = (uint8_t *)malloc(1);
        *len = 0;
        return *data ? 0 : -1;
    }
    void *p = mmap(NULL, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
        return -1;
    }
    uint8_t *buf = (uint8_t *)malloc(sz);
    if (!buf) {
        munmap(p, sz);
        return -1;
    }
    memcpy(buf, p, sz);
    munmap(p, sz);
    *data = buf;
    *len = sz;
    return 0;
#else
    FILE *f = fopen(path, "rb");
    if (!f) {
        return -1;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return -1;
    }
    long sz = ftell(f);
    if (sz < 0) {
        fclose(f);
        return -1;
    }
    rewind(f);
    uint8_t *buf = (uint8_t *)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        return -1;
    }
    if (sz > 0 && fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return -1;
    }
    fclose(f);
    *data = buf;
    *len = (size_t)sz;
    return 0;
#endif
}

static int write_file(const char *path, const uint8_t *data, size_t len)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        return -1;
    }
    if (len && fwrite(data, 1, len, f) != len) {
        fclose(f);
        return -1;
    }
    fclose(f);
    return 0;
}

static int mkdir_p(const char *path)
{
    struct stat st;
    if (stat(path, &st) == 0) {
        return S_ISDIR(st.st_mode) ? 0 : -1;
    }
    if (mkdir(path, 0755) == 0) {
        return 0;
    }
    return -1;
}

static char *path_join(const char *a, const char *b, char *out, size_t out_sz)
{
    snprintf(out, out_sz, "%s/%s", a, b);
    return out;
}

static const char *basename_stem(const char *path, char *stem, size_t stem_sz)
{
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    snprintf(stem, stem_sz, "%s", base);
    char *dot = strrchr(stem, '.');
    if (dot) {
        *dot = '\0';
    }
    return stem;
}

static int cmd_assemble(const char *in_path, const char *out_path)
{
    uint8_t *text = NULL;
    size_t text_len = 0;
    if (read_file(in_path, &text, &text_len) != 0) {
        fprintf(stderr, "error: cannot read %s: %s\n", in_path, strerror(errno));
        return 2;
    }
    CcuV1Program prog;
    char errmsg[256];
    if (ccu_v1_assemble_text((const char *)text, text_len, &prog, errmsg, sizeof(errmsg)) != 0) {
        fprintf(stderr, "error: %s\n", errmsg);
        free(text);
        return 2;
    }
    free(text);
    uint8_t *bin = NULL;
    size_t bin_len = 0;
    if (ccu_v1_program_to_binary(&prog, &bin, &bin_len) != 0) {
        fprintf(stderr, "error: encode failed\n");
        ccu_v1_program_free(&prog);
        return 2;
    }
    if (write_file(out_path, bin, bin_len) != 0) {
        fprintf(stderr, "error: cannot write %s: %s\n", out_path, strerror(errno));
        free(bin);
        ccu_v1_program_free(&prog);
        return 2;
    }
    printf("assembled %zu instructions -> %s (%zu bytes)\n", prog.count, out_path, bin_len);
    free(bin);
    ccu_v1_program_free(&prog);
    return 0;
}

static int cmd_disassemble(const char *in_path, const char *out_path)
{
    uint8_t *bin = NULL;
    size_t bin_len = 0;
    if (read_file(in_path, &bin, &bin_len) != 0) {
        fprintf(stderr, "error: cannot read %s: %s\n", in_path, strerror(errno));
        return 2;
    }
    CcuV1Program prog;
    char errmsg[256];
    if (ccu_v1_program_from_binary(bin, bin_len, &prog, errmsg, sizeof(errmsg)) != 0) {
        fprintf(stderr, "error: %s\n", errmsg);
        free(bin);
        return 2;
    }
    free(bin);
    FILE *out = fopen(out_path, "wb");
    if (!out) {
        fprintf(stderr, "error: cannot write %s: %s\n", out_path, strerror(errno));
        ccu_v1_program_free(&prog);
        return 2;
    }
    if (ccu_v1_disassemble_program(&prog, out) != 0) {
        fprintf(stderr, "error: disassemble failed\n");
        fclose(out);
        ccu_v1_program_free(&prog);
        return 2;
    }
    fclose(out);
    printf("disassembled %zu instructions -> %s\n", prog.count, out_path);
    ccu_v1_program_free(&prog);
    return 0;
}

static int cmd_verify(const char *in_path, const char *work_dir)
{
    char work[512];
    if (work_dir) {
        snprintf(work, sizeof(work), "%s", work_dir);
    } else {
        char dir[512];
        const char *slash = strrchr(in_path, '/');
        if (slash) {
            size_t n = (size_t)(slash - in_path);
            if (n >= sizeof(dir)) {
                n = sizeof(dir) - 1;
            }
            memcpy(dir, in_path, n);
            dir[n] = '\0';
            snprintf(work, sizeof(work), "%s/.ccu_v1_verify", dir);
        } else {
            snprintf(work, sizeof(work), ".ccu_v1_verify");
        }
    }
    if (mkdir_p(work) != 0) {
        fprintf(stderr, "error: cannot create work dir %s\n", work);
        return 2;
    }

    char stem[256];
    basename_stem(in_path, stem, sizeof(stem));
    char bin_path[640], dis_path[640], rebin_path[640];
    path_join(work, stem, bin_path, sizeof(bin_path));
    strncat(bin_path, ".bin", sizeof(bin_path) - strlen(bin_path) - 1);
    path_join(work, stem, dis_path, sizeof(dis_path));
    strncat(dis_path, ".dis.s", sizeof(dis_path) - strlen(dis_path) - 1);
    path_join(work, stem, rebin_path, sizeof(rebin_path));
    strncat(rebin_path, ".re.bin", sizeof(rebin_path) - strlen(rebin_path) - 1);

    uint8_t *text = NULL;
    size_t text_len = 0;
    if (read_file(in_path, &text, &text_len) != 0) {
        fprintf(stderr, "error: cannot read %s\n", in_path);
        return 2;
    }

    CcuV1Program src;
    char errmsg[256];
    if (ccu_v1_assemble_text((const char *)text, text_len, &src, errmsg, sizeof(errmsg)) != 0) {
        fprintf(stderr, "error: assemble source: %s\n", errmsg);
        free(text);
        return 2;
    }
    free(text);

    uint8_t *bin = NULL;
    size_t bin_len = 0;
    if (ccu_v1_program_to_binary(&src, &bin, &bin_len) != 0 || write_file(bin_path, bin, bin_len) != 0) {
        fprintf(stderr, "error: write binary failed\n");
        free(bin);
        ccu_v1_program_free(&src);
        return 2;
    }

    CcuV1Program from_bin;
    if (ccu_v1_program_from_binary(bin, bin_len, &from_bin, errmsg, sizeof(errmsg)) != 0) {
        fprintf(stderr, "error: decode binary: %s\n", errmsg);
        free(bin);
        ccu_v1_program_free(&src);
        return 2;
    }

    FILE *df = fopen(dis_path, "wb");
    if (!df || ccu_v1_disassemble_program(&from_bin, df) != 0) {
        fprintf(stderr, "error: write disasm failed\n");
        if (df) {
            fclose(df);
        }
        free(bin);
        ccu_v1_program_free(&src);
        ccu_v1_program_free(&from_bin);
        return 2;
    }
    fclose(df);

    /* Re-parse disassembly and compare semantics (byte-identical packed instrs). */
    uint8_t *dis_text = NULL;
    size_t dis_len = 0;
    if (read_file(dis_path, &dis_text, &dis_len) != 0) {
        fprintf(stderr, "error: cannot re-read disasm\n");
        free(bin);
        ccu_v1_program_free(&src);
        ccu_v1_program_free(&from_bin);
        return 2;
    }
    CcuV1Program from_dis;
    if (ccu_v1_assemble_text((const char *)dis_text, dis_len, &from_dis, errmsg, sizeof(errmsg)) != 0) {
        fprintf(stderr, "error: assemble disasm: %s\n", errmsg);
        free(dis_text);
        free(bin);
        ccu_v1_program_free(&src);
        ccu_v1_program_free(&from_bin);
        return 2;
    }
    free(dis_text);

    uint8_t *rebin = NULL;
    size_t rebin_len = 0;
    if (ccu_v1_program_to_binary(&from_dis, &rebin, &rebin_len) != 0 ||
        write_file(rebin_path, rebin, rebin_len) != 0) {
        fprintf(stderr, "error: write rebin failed\n");
        free(bin);
        free(rebin);
        ccu_v1_program_free(&src);
        ccu_v1_program_free(&from_bin);
        ccu_v1_program_free(&from_dis);
        return 2;
    }

    int ok = 1;
    if (ccu_v1_program_semantic_eq(&src, &from_dis) != 0) {
        fprintf(stderr, "FAIL: semantic mismatch source vs disasm\n");
        ok = 0;
        for (size_t i = 0; i < src.count && i < from_dis.count; ++i) {
            if (memcmp(&src.items[i], &from_dis.items[i], CCU_V1_INSTR_SIZE) != 0) {
                char a[1024], b[1024];
                ccu_v1_format_instr(&src.items[i], a, sizeof(a));
                ccu_v1_format_instr(&from_dis.items[i], b, sizeof(b));
                fprintf(stderr, "  instr[%zu]\n    source: %s\n    disasm: %s\n", i, a, b);
            }
        }
    }
    if (bin_len != rebin_len || memcmp(bin, rebin, bin_len) != 0) {
        fprintf(stderr, "FAIL: binary mismatch assemble(source) vs assemble(disasm)\n");
        ok = 0;
    }

    if (ok) {
        printf("OK: %zu instructions\n", src.count);
        printf("  source     : %s\n", in_path);
        printf("  binary     : %s (%zu bytes)\n", bin_path, bin_len);
        printf("  disasm     : %s\n", dis_path);
        printf("  semantic   : source == disasm\n");
        printf("  binary     : assemble(source) == assemble(disasm)\n");
    }

    free(bin);
    free(rebin);
    ccu_v1_program_free(&src);
    ccu_v1_program_free(&from_bin);
    ccu_v1_program_free(&from_dis);
    return ok ? 0 : 1;
}

static int cmd_vasm(const char *in_path, const char *out_bin, const char *out_meta, const char *out_lowered)
{
    uint8_t *text = NULL;
    size_t text_len = 0;
    if (read_file(in_path, &text, &text_len) != 0) {
        fprintf(stderr, "error: cannot read %s: %s\n", in_path, strerror(errno));
        return 2;
    }
    CcuV1VasmResult vr;
    char errmsg[256];
    if (ccu_v1_vasm_assemble((const char *)text, text_len, NULL, &vr, errmsg, sizeof(errmsg)) != 0) {
        fprintf(stderr, "error: %s\n", errmsg);
        free(text);
        return 2;
    }
    free(text);

    uint8_t *bin = NULL;
    size_t bin_len = 0;
    if (ccu_v1_program_to_binary(&vr.program, &bin, &bin_len) != 0 || write_file(out_bin, bin, bin_len) != 0) {
        fprintf(stderr, "error: cannot write %s\n", out_bin);
        free(bin);
        ccu_v1_vasm_result_free(&vr);
        return 2;
    }
    free(bin);

    if (out_meta) {
        FILE *mf = fopen(out_meta, "wb");
        if (!mf || ccu_v1_vasm_write_metainfo(&vr, mf) != 0) {
            fprintf(stderr, "error: cannot write metainfo %s\n", out_meta);
            if (mf) {
                fclose(mf);
            }
            ccu_v1_vasm_result_free(&vr);
            return 2;
        }
        fclose(mf);
    }
    if (out_lowered && vr.lowered_asm) {
        if (write_file(out_lowered, (const uint8_t *)vr.lowered_asm, strlen(vr.lowered_asm)) != 0) {
            fprintf(stderr, "error: cannot write lowered asm %s\n", out_lowered);
            ccu_v1_vasm_result_free(&vr);
            return 2;
        }
    }

    printf("vasm: %zu instructions, %zu variables -> %s\n", vr.instr_count, vr.var_count, out_bin);
    if (out_meta) {
        printf("  metainfo  : %s\n", out_meta);
    }
    for (int t = 0; t < CCU_V1_RES_COUNT; ++t) {
        if (vr.used_peak[t]) {
            printf("  %-4s peak: %u / %u\n", ccu_v1_res_type_name((CcuV1ResType)t), (unsigned)vr.used_peak[t],
                   (unsigned)vr.cfg.limits[t]);
        }
    }
    ccu_v1_vasm_result_free(&vr);
    return 0;
}

static int cmd_verify_vasm(const char *in_path, const char *work_dir)
{
    char work[512];
    if (work_dir) {
        snprintf(work, sizeof(work), "%s", work_dir);
    } else {
        char dir[512];
        const char *slash = strrchr(in_path, '/');
        if (slash) {
            size_t n = (size_t)(slash - in_path);
            if (n >= sizeof(dir)) {
                n = sizeof(dir) - 1;
            }
            memcpy(dir, in_path, n);
            dir[n] = '\0';
            snprintf(work, sizeof(work), "%s/.ccu_v1_vasm_verify", dir);
        } else {
            snprintf(work, sizeof(work), ".ccu_v1_vasm_verify");
        }
    }
    if (mkdir_p(work) != 0) {
        fprintf(stderr, "error: cannot create %s\n", work);
        return 2;
    }

    char stem[256];
    basename_stem(in_path, stem, sizeof(stem));
    char bin_path[640], meta_path[640], low_path[640], rebin_path[640];
    path_join(work, stem, bin_path, sizeof(bin_path));
    strncat(bin_path, ".bin", sizeof(bin_path) - strlen(bin_path) - 1);
    path_join(work, stem, meta_path, sizeof(meta_path));
    strncat(meta_path, ".meta.json", sizeof(meta_path) - strlen(meta_path) - 1);
    path_join(work, stem, low_path, sizeof(low_path));
    strncat(low_path, ".lowered.s", sizeof(low_path) - strlen(low_path) - 1);
    path_join(work, stem, rebin_path, sizeof(rebin_path));
    strncat(rebin_path, ".re.bin", sizeof(rebin_path) - strlen(rebin_path) - 1);

    int rc = cmd_vasm(in_path, bin_path, meta_path, low_path);
    if (rc != 0) {
        return rc;
    }
    /* lowered numeric asm should assemble identically */
    if (cmd_assemble(low_path, rebin_path) != 0) {
        return 2;
    }
    uint8_t *a = NULL, *b = NULL;
    size_t al = 0, bl = 0;
    if (read_file(bin_path, &a, &al) != 0 || read_file(rebin_path, &b, &bl) != 0) {
        fprintf(stderr, "error: cannot read verify binaries\n");
        free(a);
        free(b);
        return 2;
    }
    int ok = (al == bl && memcmp(a, b, al) == 0);
    free(a);
    free(b);
    if (!ok) {
        fprintf(stderr, "FAIL: vasm binary != assemble(lowered)\n");
        return 1;
    }
    printf("OK vasm verify: binary == assemble(lowered), metainfo=%s\n", meta_path);
    return 0;
}

static int cmd_casm(const char *in_path, const char *out_path, const char *lowered_path)
{
    uint8_t *text = NULL;
    size_t text_len = 0;
    if (read_file(in_path, &text, &text_len) != 0) {
        fprintf(stderr, "error: cannot read %s: %s\n", in_path, strerror(errno));
        return 2;
    }
    CcuV1CasmCtx ctx;
    ccu_v1_casm_init(&ctx);
    if (ccu_v1_casm_compile((const char *)text, text_len, &ctx) != 0) {
        fprintf(stderr, "error: %s\n", ctx.errmsg);
        free(text);
        ccu_v1_casm_free(&ctx);
        return 2;
    }
    free(text);

    if (lowered_path) {
        FILE *lf = fopen(lowered_path, "wb");
        if (!lf || ccu_v1_disassemble_program(&ctx.program, lf) != 0) {
            fprintf(stderr, "error: cannot write lowered %s\n", lowered_path);
            if (lf) {
                fclose(lf);
            }
            ccu_v1_casm_free(&ctx);
            return 2;
        }
        fclose(lf);
    }

    uint8_t *bin = NULL;
    size_t bin_len = 0;
    if (ccu_v1_program_to_binary(&ctx.program, &bin, &bin_len) != 0) {
        fprintf(stderr, "error: encode failed\n");
        ccu_v1_casm_free(&ctx);
        return 2;
    }
    if (write_file(out_path, bin, bin_len) != 0) {
        fprintf(stderr, "error: cannot write %s: %s\n", out_path, strerror(errno));
        free(bin);
        ccu_v1_casm_free(&ctx);
        return 2;
    }
    printf("casm: %zu instructions -> %s (%zu bytes)\n", ctx.program.count, out_path, bin_len);
    free(bin);
    ccu_v1_casm_free(&ctx);
    return 0;
}

static int cmd_verify_casm(const char *in_path, const char *work_dir)
{
    char work[512];
    if (work_dir) {
        snprintf(work, sizeof(work), "%s", work_dir);
    } else {
        char dir[512];
        const char *slash = strrchr(in_path, '/');
        if (slash) {
            size_t n = (size_t)(slash - in_path);
            if (n >= sizeof(dir)) {
                n = sizeof(dir) - 1;
            }
            memcpy(dir, in_path, n);
            dir[n] = '\0';
            snprintf(work, sizeof(work), "%s/.ccu_v1_casm_verify", dir);
        } else {
            snprintf(work, sizeof(work), ".ccu_v1_casm_verify");
        }
    }
    if (mkdir_p(work) != 0) {
        fprintf(stderr, "error: cannot create work dir %s\n", work);
        return 2;
    }

    char stem[256];
    basename_stem(in_path, stem, sizeof(stem));
    char bin_path[640], low_path[640], rebin_path[640];
    path_join(work, stem, bin_path, sizeof(bin_path));
    strncat(bin_path, ".bin", sizeof(bin_path) - strlen(bin_path) - 1);
    path_join(work, stem, low_path, sizeof(low_path));
    strncat(low_path, ".lowered.s", sizeof(low_path) - strlen(low_path) - 1);
    path_join(work, stem, rebin_path, sizeof(rebin_path));
    strncat(rebin_path, ".re.bin", sizeof(rebin_path) - strlen(rebin_path) - 1);

    if (cmd_casm(in_path, bin_path, low_path) != 0) {
        return 2;
    }
    if (cmd_assemble(low_path, rebin_path) != 0) {
        return 2;
    }
    uint8_t *a = NULL, *b = NULL;
    size_t al = 0, bl = 0;
    if (read_file(bin_path, &a, &al) != 0 || read_file(rebin_path, &b, &bl) != 0) {
        fprintf(stderr, "error: cannot read verify binaries\n");
        free(a);
        free(b);
        return 2;
    }
    int ok = (al == bl && memcmp(a, b, al) == 0);
    free(a);
    free(b);
    if (!ok) {
        fprintf(stderr, "FAIL: casm binary != assemble(lowered)\n");
        return 1;
    }
    printf("OK casm verify: binary == assemble(lowered)\n");
    return 0;
}

static void usage(const char *argv0)
{
    fprintf(stderr,
            "Usage:\n"
            "  %s assemble    <in.s> -o <out.bin>\n"
            "  %s disassemble <in.bin> -o <out.s>\n"
            "  %s verify      <in.s> [-w workdir]\n"
            "  %s vasm        <in.s> -o <out.bin> [-m out.meta.json] [--lowered out.s]\n"
            "  %s verify-vasm <in.s> [-w workdir]\n"
            "  %s casm        <in.c> -o <out.bin> [--lowered out.s]\n"
            "  %s verify-casm <in.c> [-w workdir]\n"
            "Aliases: as, dis, assemble-c\n",
            argv0, argv0, argv0, argv0, argv0, argv0, argv0);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }
    const char *cmd = argv[1];
    if (strcmp(cmd, "assemble") == 0 || strcmp(cmd, "as") == 0) {
        const char *in = NULL;
        const char *out = NULL;
        for (int i = 2; i < argc; ++i) {
            if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
                if (i + 1 >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                out = argv[++i];
            } else if (!in) {
                in = argv[i];
            } else {
                usage(argv[0]);
                return 2;
            }
        }
        if (!in || !out) {
            usage(argv[0]);
            return 2;
        }
        return cmd_assemble(in, out);
    }
    if (strcmp(cmd, "disassemble") == 0 || strcmp(cmd, "dis") == 0) {
        const char *in = NULL;
        const char *out = NULL;
        for (int i = 2; i < argc; ++i) {
            if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
                if (i + 1 >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                out = argv[++i];
            } else if (!in) {
                in = argv[i];
            } else {
                usage(argv[0]);
                return 2;
            }
        }
        if (!in || !out) {
            usage(argv[0]);
            return 2;
        }
        return cmd_disassemble(in, out);
    }
    if (strcmp(cmd, "verify") == 0) {
        const char *in = NULL;
        const char *work = NULL;
        for (int i = 2; i < argc; ++i) {
            if (strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--work-dir") == 0) {
                if (i + 1 >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                work = argv[++i];
            } else if (!in) {
                in = argv[i];
            } else {
                usage(argv[0]);
                return 2;
            }
        }
        if (!in) {
            usage(argv[0]);
            return 2;
        }
        return cmd_verify(in, work);
    }
    if (strcmp(cmd, "vasm") == 0 || strcmp(cmd, "assemble-var") == 0) {
        const char *in = NULL;
        const char *out = NULL;
        const char *meta = NULL;
        const char *lowered = NULL;
        for (int i = 2; i < argc; ++i) {
            if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
                if (i + 1 >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                out = argv[++i];
            } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--metainfo") == 0) {
                if (i + 1 >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                meta = argv[++i];
            } else if (strcmp(argv[i], "--lowered") == 0) {
                if (i + 1 >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                lowered = argv[++i];
            } else if (!in) {
                in = argv[i];
            } else {
                usage(argv[0]);
                return 2;
            }
        }
        if (!in || !out) {
            usage(argv[0]);
            return 2;
        }
        return cmd_vasm(in, out, meta, lowered);
    }
    if (strcmp(cmd, "verify-vasm") == 0) {
        const char *in = NULL;
        const char *work = NULL;
        for (int i = 2; i < argc; ++i) {
            if (strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--work-dir") == 0) {
                if (i + 1 >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                work = argv[++i];
            } else if (!in) {
                in = argv[i];
            } else {
                usage(argv[0]);
                return 2;
            }
        }
        if (!in) {
            usage(argv[0]);
            return 2;
        }
        return cmd_verify_vasm(in, work);
    }
    if (strcmp(cmd, "casm") == 0 || strcmp(cmd, "assemble-c") == 0) {
        const char *in = NULL;
        const char *out = NULL;
        const char *lowered = NULL;
        for (int i = 2; i < argc; ++i) {
            if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
                if (i + 1 >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                out = argv[++i];
            } else if (strcmp(argv[i], "--lowered") == 0) {
                if (i + 1 >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                lowered = argv[++i];
            } else if (!in) {
                in = argv[i];
            } else {
                usage(argv[0]);
                return 2;
            }
        }
        if (!in || !out) {
            usage(argv[0]);
            return 2;
        }
        return cmd_casm(in, out, lowered);
    }
    if (strcmp(cmd, "verify-casm") == 0) {
        const char *in = NULL;
        const char *work = NULL;
        for (int i = 2; i < argc; ++i) {
            if (strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--work-dir") == 0) {
                if (i + 1 >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                work = argv[++i];
            } else if (!in) {
                in = argv[i];
            } else {
                usage(argv[0]);
                return 2;
            }
        }
        if (!in) {
            usage(argv[0]);
            return 2;
        }
        return cmd_verify_casm(in, work);
    }
    usage(argv[0]);
    return 2;
}
