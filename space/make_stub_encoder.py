#!/usr/bin/env python3
"""Write a tiny placeholder encoder ONNX (zeros of shape [batch, seq, d_model]).

transformers.js loads a seq2seq model as encoder_model + decoder_model_merged from one
folder. The split-decoder build keeps the Ido decoder in onnx_ido/ next to this stub,
so the browser can load it as a second model without downloading the real encoder
twice; the app then points that model's encoder session at the real one.
Usage: python space/make_stub_encoder.py OUT.onnx [d_model=1024]
"""
import sys

import onnx
from onnx import TensorProto, helper

out, d = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1024
nodes = [
    helper.make_node("Shape", ["input_ids"], ["bs"]),
    helper.make_node("Constant", [], ["d"], value=helper.make_tensor("d", TensorProto.INT64, [1], [d])),
    helper.make_node("Concat", ["bs", "d"], ["shape"], axis=0),
    helper.make_node("ConstantOfShape", ["shape"], ["last_hidden_state"],
                     value=helper.make_tensor("z", TensorProto.FLOAT, [1], [0.0])),
]
g = helper.make_graph(nodes, "stub_encoder",
                      [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"]),
                       helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "seq"])],
                      [helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["batch", "seq", d])])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 8
onnx.checker.check_model(m)
onnx.save(m, out)
print("wrote", out)
