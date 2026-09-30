from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


FFN_SHAPES = {(768, 3072), (3072, 768)}
QUANTIZED_LAYERS = range(6, 12)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize the upper EmbeddingRWKV FFN layers while preserving sensitive paths"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = onnx.load(args.input, load_external_data=False)
    shapes = {initializer.name: tuple(initializer.dims) for initializer in model.graph.initializer}
    prefixes = tuple(f"/layers.{layer}/" for layer in QUANTIZED_LAYERS)
    nodes = [
        node.name
        for node in model.graph.node
        if node.op_type == "MatMul"
        and len(node.input) > 1
        and shapes.get(node.input[1]) in FFN_SHAPES
        and node.name.startswith(prefixes)
    ]
    if len(nodes) != 12:
        raise ValueError(f"Expected 12 upper-layer FFN MatMul nodes, found {len(nodes)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        args.input,
        args.output,
        weight_type=QuantType.QInt8,
        per_channel=True,
        nodes_to_quantize=nodes,
        extra_options={"MatMulConstBOnly": True},
    )
    print(f"Quantized {len(nodes)} MatMul nodes to {args.output}")


if __name__ == "__main__":
    main()
