# High-performance CCU V1 assembler / disassembler
CC      ?= gcc
CFLAGS  ?= -O3 -march=native -flto -Wall -Wextra -Werror -std=c11
CFLAGS  += -Iinclude
LDFLAGS ?= -flto

SRCS := src/isa.c src/asm.c src/cli.c
OBJS := $(SRCS:src/%.c=build/%.o)

.PHONY: all clean test install

all: build/ccu_v1_asm

build:
	mkdir -p build

build/%.o: src/%.c include/ccu_v1_isa.h include/ccu_v1_asm.h | build
	$(CC) $(CFLAGS) -c $< -o $@

build/ccu_v1_asm: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

test: build/ccu_v1_asm
	./build/ccu_v1_asm verify ../examples/all_opcodes.s -w build/verify_out
	./build/ccu_v1_asm as ../examples/all_opcodes.s -o build/all.bin
	./build/ccu_v1_asm dis build/all.bin -o build/all.dis.s
	./build/ccu_v1_asm as build/all.dis.s -o build/all.re.bin
	cmp build/all.bin build/all.re.bin
	@echo "ALL C TESTS PASSED"

clean:
	rm -rf build

install: build/ccu_v1_asm
	install -d $(DESTDIR)/usr/local/bin
	install -m 755 build/ccu_v1_asm $(DESTDIR)/usr/local/bin/ccu_v1_asm
