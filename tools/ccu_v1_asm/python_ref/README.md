# Python reference (optional)

Historical/reference implementation. Prefer the C tool:

```bash
cd ../c && make && ./build/ccu_v1_asm verify ../examples/all_opcodes.s
```

Run reference tests from `tools/`:

```bash
PYTHONPATH=ccu_v1_asm/python_ref python3 -m unittest discover -s ccu_v1_asm/python_ref/tests -v
```
