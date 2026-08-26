#!/usr/bin/env python3
"""Sanity checks for ATen → Pattern mapping used by the profiler."""

from pattern_map import classify


def test_classify() -> None:
    cases = {
        "conv2d": "09",
        "mkldnn_convolution": "09",
        "conv_transpose2d": "09",
        "linear": "10",
        "matmul": "10",
        "addmm": "10",
        "bmm": "10",
        "lstm": "10",
        "relu_": "03",
        "gelu": "03",
        "hardswish": "03",
        "softmax": "12.1",
        "layer_norm": "11",
        "batch_norm": "11",
        "_native_batch_norm_legit_no_training": "11",
        "_native_multi_head_attention": "12.2",
        "scaled_dot_product_attention": "12.2",
        "embedding": "13",
        "index": "13",
        "index_put_": "13",
        "topk": "25",
        "nms": "25",
        "_unique2": "25",
        "cat": "15",
        "view": "14",
        "transpose": "14",
        "triu": "07",
        "masked_fill": "07",
        "add_": "01",
        "where": "02",
        "max_pool2d": "08",
        "upsample_bilinear2d": "23",
    }
    failed = []
    for name, expect in cases.items():
        pid, _, _ = classify(name)
        if pid != expect:
            failed.append((name, expect, pid))
    assert not failed, failed


if __name__ == "__main__":
    test_classify()
    print("pattern_map classify: ok")
