# C++ CCU V1 assembler (façade over C core sources in ../c)
CXX      ?= g++
CC       ?= gcc
CXXFLAGS ?= -O3 -march=native -Wall -Wextra -Werror -std=c++17
CXXFLAGS += -Iinclude -I../c/include
CFLAGS   ?= -O3 -march=native -Wall -Wextra -Werror -std=c11
CFLAGS   += -I../c/include

C_DIR  := ../c
C_SRCS := isa.c asm.c vasm.c
C_OBJS := $(C_SRCS:%.c=build/c_%.o)

SRCS := src/io.cpp src/vasm.cpp src/main.cpp
OBJS := $(SRCS:src/%.cpp=build/%.o)

.PHONY: all clean test

all: build/ccu_v1_asm_cpp

build:
	mkdir -p build

build/%.o: src/%.cpp include/ccu/v1/*.hpp | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile C core into this tree so we never mix clang-LTO bitcode with g++.
build/c_%.o: $(C_DIR)/src/%.c $(C_DIR)/include/ccu_v1_isa.h $(C_DIR)/include/ccu_v1_asm.h $(C_DIR)/include/ccu_v1_vasm.h | build
	$(CC) $(CFLAGS) -c $< -o $@

build/ccu_v1_asm_cpp: $(OBJS) $(C_OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $(C_OBJS) -o $@

test: build/ccu_v1_asm_cpp
	./build/ccu_v1_asm_cpp verify ../examples/all_opcodes.s -w build/verify_out
	./build/ccu_v1_asm_cpp as ../examples/all_opcodes.s -o build/all.bin
	./build/ccu_v1_asm_cpp dis build/all.bin -o build/all.dis.s
	./build/ccu_v1_asm_cpp as build/all.dis.s -o build/all.re.bin
	cmp build/all.bin build/all.re.bin
	./build/ccu_v1_asm_cpp verify-vasm ../examples/vars_reuse.s -w build/vasm_verify
	# Cross-check: C++ binary == C binary
	$(MAKE) -C $(C_DIR) build/ccu_v1_asm
	$(C_DIR)/build/ccu_v1_asm as ../examples/all_opcodes.s -o build/all_c.bin
	cmp build/all.bin build/all_c.bin
	@echo "ALL C++ TESTS PASSED (matches C encoding)"

clean:
	rm -rf build
