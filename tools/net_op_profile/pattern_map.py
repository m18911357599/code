"""Map ATen / profiler op names to Pattern IDs in docs/design/pytorch_op_feature_compete.md."""

from __future__ import annotations

# Longest-prefix / keyword rules. First match wins.
# Pattern IDs follow pytorch_op_feature_compete.md (01–29).
_RULES: list[tuple[str, str, str]] = [
    # Cube
    ("convolution", "09", "Cube"),
    ("conv2d", "09", "Cube"),
    ("conv1d", "09", "Cube"),
    ("conv3d", "09", "Cube"),
    ("conv_transpose", "09", "Cube"),
    ("mkldnn_convolution", "09", "Cube"),
    ("thnn_conv", "09", "Cube"),
    ("slow_conv", "09", "Cube"),
    ("im2col", "09", "Cube"),
    ("col2im", "09", "Cube"),
    ("addmm", "10", "Cube"),
    ("addmv", "10", "Cube"),
    ("mm", "10", "Cube"),
    ("bmm", "10", "Cube"),
    ("baddbmm", "10", "Cube"),
    ("addbmm", "10", "Cube"),
    ("matmul", "10", "Cube"),
    ("linear", "10", "Cube"),
    ("einsum", "10", "Cube"),
    ("tensordot", "10", "Cube"),
    ("dot", "10", "Cube"),
    ("scaled_dot_product", "12.2", "Cube"),
    ("flash_attention", "12.2", "Cube"),
    ("efficient_attention", "12.2", "Cube"),
    ("_native_multi_head_attention", "12.2", "Cube"),
    ("multi_head_attention", "12.2", "Cube"),
    ("lstm", "10", "Cube"),
    ("gru", "10", "Cube"),
    # Softmax (unfused)
    ("log_softmax", "12.1", "Vec"),
    ("softmax", "12.1", "Vec"),
    ("_softmax", "12.1", "Vec"),
    # Norm
    ("layer_norm", "11", "Vec"),
    ("batch_norm", "11", "Vec"),
    ("group_norm", "11", "Vec"),
    ("instance_norm", "11", "Vec"),
    ("rms_norm", "11", "Vec"),
    ("native_layer_norm", "11", "Vec"),
    ("native_batch_norm", "11", "Vec"),
    ("native_group_norm", "11", "Vec"),
    ("_native_batch_norm", "11", "Vec"),
    ("normalize", "11", "Vec"),
    # Activation
    ("gelu", "03", "Vec"),
    ("silu", "03", "Vec"),
    ("swish", "03", "Vec"),
    ("relu", "03", "Vec"),
    ("leaky_relu", "03", "Vec"),
    ("hardtanh", "03", "Vec"),
    ("hardswish", "03", "Vec"),
    ("hardsigmoid", "03", "Vec"),
    ("hardshrink", "03", "Vec"),
    ("sigmoid", "03", "Vec"),
    ("tanh", "03", "Vec"),
    ("elu", "03", "Vec"),
    ("celu", "03", "Vec"),
    ("selu", "03", "Vec"),
    ("prelu", "03", "Vec"),
    ("threshold", "03", "Vec"),
    ("glu", "03", "Vec"),
    ("mish", "03", "Vec"),
    # Pool / upsample
    ("max_pool", "08", "Vec"),
    ("avg_pool", "08", "Vec"),
    ("adaptive_avg_pool", "08", "Vec"),
    ("adaptive_max_pool", "08", "Vec"),
    ("lp_pool", "08", "Vec"),
    ("upsample", "23", "Vec"),
    ("interpolate", "23", "Vec"),
    ("grid_sampler", "23", "Vec"),
    ("affine_grid", "23", "Vec"),
    # Pad
    ("constant_pad", "07", "Vec"),
    ("reflection_pad", "07", "Vec"),
    ("replication_pad", "07", "Vec"),
    ("circular_pad", "07", "Vec"),
    ("pad", "07", "Vec"),
    ("zero_", "07", "Vec"),
    ("fill", "07", "Vec"),
    ("triu", "07", "Vec"),
    ("tril", "07", "Vec"),
    # Index / embedding
    ("embedding", "13", "Vec"),
    ("gather", "13", "Vec"),
    ("scatter", "13", "Vec"),
    ("index_select", "13", "Vec"),
    ("index_put", "13", "Vec"),
    ("index_add", "13", "Vec"),
    ("index_copy", "13", "Vec"),
    ("nonzero", "26", "Vec"),
    ("masked_select", "26", "Vec"),
    ("take", "13", "Vec"),
    ("one_hot", "13", "Vec"),
    ("bucketize", "13", "Vec"),
    ("searchsorted", "13", "Vec"),
    # Sort / nms helpers
    ("nms", "25", "Vec"),
    ("topk", "25", "Vec"),
    ("sort", "25", "Vec"),
    ("argsort", "25", "Vec"),
    ("kthvalue", "25", "Vec"),
    ("unique", "25", "Vec"),
    ("_unique2", "25", "Vec"),
    ("index", "13", "Vec"),
    # Cat / split
    ("cat", "15", "Vec"),
    ("concat", "15", "Vec"),
    ("stack", "15", "Vec"),
    ("split", "15", "Vec"),
    ("chunk", "15", "Vec"),
    ("unbind", "15", "Vec"),
    ("tensor_split", "15", "Vec"),
    # Copy / cast
    ("clone", "16", "Vec"),
    ("copy", "16", "Vec"),
    ("_to_copy", "16", "Vec"),
    ("to", "16", "Vec"),
    # Factory
    ("empty", "17", "Vec"),
    ("zeros", "17", "Vec"),
    ("ones", "17", "Vec"),
    ("full", "17", "Vec"),
    ("arange", "17", "Vec"),
    ("new_empty", "17", "Vec"),
    ("new_zeros", "17", "Vec"),
    ("new_ones", "17", "Vec"),
    ("scalar_tensor", "17", "Vec"),
    # Dropout
    ("dropout", "18", "Vec"),
    # Scan
    ("cumsum", "06", "Vec"),
    ("cumprod", "06", "Vec"),
    ("cummax", "06", "Vec"),
    ("cummin", "06", "Vec"),
    # Reduce
    ("logsumexp", "05", "Vec"),
    ("sum", "05", "Vec"),
    ("mean", "05", "Vec"),
    ("prod", "05", "Vec"),
    ("argmax", "05", "Vec"),
    ("argmin", "05", "Vec"),
    ("amax", "05", "Vec"),
    ("amin", "05", "Vec"),
    ("max", "05", "Vec"),
    ("min", "05", "Vec"),
    ("all", "05", "Vec"),
    ("any", "05", "Vec"),
    ("var", "05", "Vec"),
    ("std", "05", "Vec"),
    ("norm", "05", "Vec"),
    ("linalg_vector_norm", "05", "Vec"),
    # Compare / select
    ("where", "02", "Vec"),
    ("clamp", "02", "Vec"),
    ("maximum", "02", "Vec"),
    ("minimum", "02", "Vec"),
    ("eq", "02", "Vec"),
    ("ne", "02", "Vec"),
    ("lt", "02", "Vec"),
    ("le", "02", "Vec"),
    ("gt", "02", "Vec"),
    ("ge", "02", "Vec"),
    ("logical_", "02", "Vec"),
    ("bitwise_", "02", "Vec"),
    # Layout
    ("as_strided", "14", "Vec"),
    ("view", "14", "Vec"),
    ("reshape", "14", "Vec"),
    ("transpose", "14", "Vec"),
    ("permute", "14", "Vec"),
    ("squeeze", "14", "Vec"),
    ("unsqueeze", "14", "Vec"),
    ("flatten", "14", "Vec"),
    ("unflatten", "14", "Vec"),
    ("contiguous", "14", "Vec"),
    ("select", "14", "Vec"),
    ("narrow", "14", "Vec"),
    ("slice", "14", "Vec"),
    ("unfold", "14", "Vec"),
    ("t", "14", "Vec"),
    ("expand", "04", "Vec"),
    ("repeat", "04", "Vec"),
    ("broadcast", "04", "Vec"),
    ("detach", "16", "Vec"),
    ("alias", "16", "Vec"),
    # Pointwise arithmetic (keep last: short names)
    ("addcdiv", "01", "Vec"),
    ("addcmul", "01", "Vec"),
    ("lerp", "01", "Vec"),
    ("add", "01", "Vec"),
    ("sub", "01", "Vec"),
    ("rsub", "01", "Vec"),
    ("mul", "01", "Vec"),
    ("div", "01", "Vec"),
    ("true_divide", "01", "Vec"),
    ("floor_divide", "01", "Vec"),
    ("remainder", "01", "Vec"),
    ("pow", "01", "Vec"),
    ("sqrt", "01", "Vec"),
    ("rsqrt", "01", "Vec"),
    ("exp", "01", "Vec"),
    ("log", "01", "Vec"),
    ("neg", "01", "Vec"),
    ("abs", "01", "Vec"),
    ("reciprocal", "01", "Vec"),
    ("floor", "01", "Vec"),
    ("ceil", "01", "Vec"),
    ("round", "01", "Vec"),
    ("trunc", "01", "Vec"),
    ("erf", "27", "Vec"),
    ("lgamma", "27", "Vec"),
]


_PATTERN_NAME = {
    "01": "逐元素算术",
    "02": "比较/逻辑/选择",
    "03": "激活与非线性",
    "04": "广播与维扩展",
    "05": "规约",
    "06": "扫描与累积",
    "07": "填充",
    "08": "池化",
    "09": "卷积 (Cube)",
    "10": "矩阵乘 (Cube)",
    "11": "归一化",
    "12.1": "Softmax",
    "12.2": "SDPA/MHA (Cube)",
    "13": "索引/散射/Embedding",
    "14": "布局变换",
    "15": "拼接与分割",
    "16": "拷贝与类型转换",
    "17": "工厂与创建",
    "18": "随机与 Dropout",
    "23": "上采样与插值",
    "25": "排序/TopK",
    "26": "数据依赖动态输出",
    "27": "特殊函数",
    "28": "元信息/控制",
    "99": "其它",
}


def normalize_op_name(name: str) -> str:
    n = name.strip()
    for prefix in ("aten::", "aten.", "torchvision::", "torchvision.", "prepacked::"):
        if n.startswith(prefix):
            n = n[len(prefix) :]
    n = n.split(".")[0]
    if n.endswith("_"):
        stem = n[:-1]
        if stem:
            n = stem
    return n.lower()


def classify(name: str) -> tuple[str, str, str]:
    """Return (pattern_id, pattern_name, class Cube|Vec|Other)."""
    key = normalize_op_name(name)
    for needle, pid, klass in _RULES:
        if len(needle) <= 2:
            if key == needle:
                return pid, _PATTERN_NAME.get(pid, pid), klass
            continue
        if key == needle or key.startswith(needle) or key.endswith(needle):
            return pid, _PATTERN_NAME.get(pid, pid), klass
    return "99", _PATTERN_NAME["99"], "Other"


def pattern_title(pid: str) -> str:
    return _PATTERN_NAME.get(pid, pid)
