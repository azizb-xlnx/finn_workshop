import onnx
import numpy as np

import onnx.helper as helper
import onnx.numpy_helper as numpy_helper

# Define the input tensor
input_tensor = helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [None, None])

# Define the output tensor
output_tensor = helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [None, None])

# Define the cached tensors
cos_values = np.random.rand(32768, 64).astype(np.float32)  # Random values
sin_values = np.random.rand(32768, 64).astype(np.float32)  # Random values

cos_tensor = numpy_helper.from_array(cos_values, name='cos')
sin_tensor = numpy_helper.from_array(sin_values, name='sin')

# Define the custom RoPE node
rope_node = helper.make_node(
    'RotaryEmbedding',  # Custom node name
    ['input', 'cos', 'sin'],  # Inputs
    ['output'],  # Outputs
    name='CustomRoPE'
)

# Create the graph
graph = helper.make_graph(
    [rope_node],  # Nodes
    'RoPEGraph',  # Graph name
    [input_tensor],  # Inputs
    [output_tensor],  # Outputs
    initializer=[
        helper.make_tensor('cos', onnx.TensorProto.FLOAT, cos_values.shape, cos_values),
        helper.make_tensor('sin', onnx.TensorProto.FLOAT, sin_values.shape, sin_values)
    ]  # Initializers
)

# Create the model
model = helper.make_model(graph, producer_name='custom_rope_model')

# Save the model to a file
onnx.save(model, 'rope_node.onnx')

print("ONNX model with custom RoPE node has been created and saved as 'rope_node.onnx'")