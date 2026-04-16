import onnx
from onnx import helper, TensorProto

TARGETS = {
    "/pts_bbox_head/encoder/layers.0/attentions.0/GridSample",
    "/pts_bbox_head/encoder/layers.0/attentions.1/deformable_attention/GridSample",
    "/pts_bbox_head/encoder/layers.1/attentions.0/GridSample",
    "/pts_bbox_head/encoder/layers.1/attentions.1/deformable_attention/GridSample",
    "/pts_bbox_head/encoder/layers.2/attentions.0/GridSample",
    "/pts_bbox_head/encoder/layers.2/attentions.1/deformable_attention/GridSample",
}

src = "/workspace/quant_pipeline/QuantizedSSR/quantized_export/onnx/vad_detector_int8.onnx"
dst = "/workspace/quant_pipeline/QuantizedSSR/quantized_export/onnx/vad_detector_int8_custom_gs.onnx"

# --------------------------------------------------
# Rewrite graph:
#   GridSample(x: float, grid: double)
# becomes
#   Cast(grid -> float)
#   ai.onnx.contrib::GridSampleBilinearZerosAC0(x, casted_grid)
# --------------------------------------------------
m = onnx.load(src)

new_nodes = []
for node in m.graph.node:
    if node.name in TARGETS and node.op_type == "GridSample":
        x_in = node.input[0]
        grid_in = node.input[1]
        y_out = node.output[0]

        cast_grid_out = grid_in + "_cast_float_for_custom_gs"

        cast_node = helper.make_node(
            "Cast",
            inputs=[grid_in],
            outputs=[cast_grid_out],
            name=node.name + "_CastGridToFloat",
            to=TensorProto.FLOAT,
        )

        custom_node = helper.make_node(
            "GridSampleBilinearZerosAC0",
            inputs=[x_in, cast_grid_out],
            outputs=[y_out],
            name=node.name,
            domain="ai.onnx.contrib",
        )

        new_nodes.append(cast_node)
        new_nodes.append(custom_node)
    else:
        new_nodes.append(node)

del m.graph.node[:]
m.graph.node.extend(new_nodes)

has_contrib = any(op.domain == "ai.onnx.contrib" for op in m.opset_import)
if not has_contrib:
    opset = m.opset_import.add()
    opset.domain = "ai.onnx.contrib"
    opset.version = 1

onnx.save(m, dst)
print("saved:", dst)