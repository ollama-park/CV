import onnx
from onnx import helper, TensorProto

model = onnx.load("runs/detect/engine_bay_runs/yolo11_engine/weights/best.onnx")
graph = model.graph

orig_output_name = graph.output[0].name
print(f"Original output: {orig_output_name}")

def make_slice(input_name, output_name, starts, ends, axes):
    starts_init = helper.make_tensor(output_name + "_starts", TensorProto.INT64, [len(starts)], starts)
    ends_init   = helper.make_tensor(output_name + "_ends",   TensorProto.INT64, [len(ends)],   ends)
    axes_init   = helper.make_tensor(output_name + "_axes",   TensorProto.INT64, [len(axes)],   axes)
    node = helper.make_node("Slice",
        inputs=[input_name, output_name + "_starts", output_name + "_ends", output_name + "_axes"],
        outputs=[output_name]
    )
    return node, [starts_init, ends_init, axes_init]

# Slice boxes [:, :, 0:4]
box_node,   box_inits   = make_slice(orig_output_name, "boxes_raw",  [0], [4], [2])
# Slice scores [:, :, 4:5]
score_node, score_inits = make_slice(orig_output_name, "scores_raw", [4], [5], [2])
# Slice classIDs [:, :, 5:6]
cls_node,   cls_inits   = make_slice(orig_output_name, "cls_raw",    [5], [6], [2])

# opset 12: Squeeze axes are an ATTRIBUTE, not an input
# boxes  [1, 300, 4] -> [300, 4]
squeeze_boxes  = helper.make_node("Squeeze", inputs=["boxes_raw"],  outputs=["boxes"],     axes=[0])
# scores [1, 300, 1] -> [300]
squeeze_scores = helper.make_node("Squeeze", inputs=["scores_raw"], outputs=["scores"],    axes=[0, 2])
# cls    [1, 300, 1] -> [300]
squeeze_cls    = helper.make_node("Squeeze", inputs=["cls_raw"],    outputs=["cls_float"], axes=[0, 2])

# Cast classIDs float -> int32
cast_cls = helper.make_node("Cast", inputs=["cls_float"], outputs=["class_ids"], to=TensorProto.INT32)

# Add nodes
for node in [box_node, score_node, cls_node, squeeze_boxes, squeeze_scores, squeeze_cls, cast_cls]:
    graph.node.append(node)

# Add slice initializers only
for init in box_inits + score_inits + cls_inits:
    graph.initializer.append(init)

# Replace outputs
del graph.output[:]
graph.output.extend([
    helper.make_tensor_value_info("boxes",     TensorProto.FLOAT, [300, 4]),
    helper.make_tensor_value_info("class_ids", TensorProto.INT32,  [300]),
    helper.make_tensor_value_info("scores",    TensorProto.FLOAT, [300]),
])

onnx.checker.check_model(model)
out_path = "runs/detect/engine_bay_runs/yolo11_engine/weights/best_3out.onnx"
onnx.save(model, out_path)
print(f"✅ Saved 3-output model to {out_path}")