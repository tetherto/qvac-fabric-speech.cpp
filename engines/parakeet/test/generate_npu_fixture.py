from pathlib import Path

import onnx
from onnx import TensorProto, helper

graph = helper.make_graph(
    [helper.make_node('Identity', ['audio_signal'], ['outputs']),
     helper.make_node('Identity', ['length'], ['encoded_lengths']),
     helper.make_node('Identity', ['audio_signal'], ['debug'])],
    'parakeet-onnx-encoder-contract',
    [helper.make_tensor_value_info('length', TensorProto.INT64, [1]),
     helper.make_tensor_value_info('audio_signal', TensorProto.FLOAT, [1, 2, 4])],
    [helper.make_tensor_value_info('debug', TensorProto.FLOAT, [1, 2, 4]),
     helper.make_tensor_value_info('encoded_lengths', TensorProto.INT64, [1]),
     helper.make_tensor_value_info('outputs', TensorProto.FLOAT, [1, 2, 4])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=8)
onnx.checker.check_model(model)
onnx.save(model, Path(__file__).with_name('npu-encoder-identity.onnx'))
