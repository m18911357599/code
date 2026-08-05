# code docs

Documentation for the `code` repository.

## Design documents

- [AscendC Basic API：基于 `kernel_operator_xx` 的 Feature 提取与 npu_arch 简化设计](design/npu_arch_kernel_operator_feature_simplify.md)

## Local preview

This site is built with [MkDocs](https://www.mkdocs.org/) and the
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/mkdocs serve -a 0.0.0.0:8000
```

Then open http://localhost:8000.
