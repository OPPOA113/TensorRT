
# onnx 创建模型

依赖库

```Python
import onnx
from onnx import helper
from onnx import TensorProto

# valueinfoProto 值信息类型，如输入输出: 
# helper.make_tensor_value_info
input_tensor = helper.make_tensor_value_info(
            str(layer_name), TensorProto.FLOAT, [batch_size, channels, height, width]
        )
output_tensor = helper.make_tensor_value_info(tensor_name, TensorProto.FLOAT, output_dims)

# 创建常规节点类型: 
# helper.make_node
conv_node = helper.make_node(
            "Conv",
            inputs=inputs,
            outputs=[layer_name],
            kernel_shape=kernel_shape,
            strides=strides,
            auto_pad="SAME_LOWER",
            dilations=dilations,
            name=layer_name,
        )
batchnorm_node = helper.make_node(
                "BatchNormalization",
                inputs=inputs,
                outputs=[layer_name_bn],
                epsilon=self.epsilon_bn,
                momentum=self.momentum_bn,
                name=layer_name_bn,
            )
lrelu_node = helper.make_node(
                "LeakyRelu", inputs=inputs, outputs=[layer_name_lrelu], name=layer_name_lrelu, alpha=self.alpha_lrelu
            )

# 创建节点后，创建graph: 
# helper.make_graph
self.graph_def = helper.make_graph(
            nodes=self._nodes, name="YOLOv3-608", inputs=inputs, outputs=outputs, initializer=initializer
        )

# 然后再创建model：
# helper.make_model
model_def = helper.make_model(self.graph_def, producer_name="NVIDIA TensorRT sample")

# check模型的正确性：
# onnx.checker.check_model
onnx.checker.check_model(yolov3_model_def)

# 导出模型：
# onnx.save
onnx.save(yolov3_model_def, output_file_path)
```