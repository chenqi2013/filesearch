from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


FFN_SHAPES = {(768, 3072), (3072, 768)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize selected EmbeddingRWKV FFN and attention output projections"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffn-start", type=int, default=3, choices=range(13),
                        help="First INT8 FFN layer; production default is layer 3 (zero based)")
    parser.add_argument("--attention-output-start", type=int, default=6, choices=range(13),
                        help="First INT8 attention output layer; 12 disables it, default is 6")
    parser.add_argument("--reduce-range", action="store_true",
                        help="Use seven-bit weights to avoid AVX2 integer accumulation saturation")
    args = parser.parse_args()

    model = onnx.load(args.input, load_external_data=False)
    shapes = {initializer.name: tuple(initializer.dims) for initializer in model.graph.initializer}
    prefixes = tuple(f"/layers.{layer}/" for layer in range(args.ffn_start, 12))
    nodes = [
        node.name
        for node in model.graph.node
        if node.op_type == "MatMul"
        and len(node.input) > 1
        and shapes.get(node.input[1]) in FFN_SHAPES
        and node.name.startswith(prefixes)
    ]
    expected = 2 * (12 - args.ffn_start)
    if args.attention_output_start is not None:
        nodes.extend(node.name for node in model.graph.node
                     if node.op_type == "MatMul" and node.name in {
                         f"/layers.{layer}/MatMul_11"
                         for layer in range(args.attention_output_start, 12)})
        expected += 12 - args.attention_output_start
    if len(nodes) != expected or not nodes:
        raise ValueError(f"Expected {expected} MatMul nodes, found {len(nodes)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        args.input,
        args.output,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=args.reduce_range,
        nodes_to_quantize=nodes,
        extra_options={"MatMulConstBOnly": True},
    )
    print(f"Quantized {len(nodes)} MatMul nodes to {args.output}")


if __name__ == "__main__":
    main()
