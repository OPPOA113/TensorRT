#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import json

import wget
import onnx
import onnx_graphsurgeon as gs

MODEL_URL = "https://github.com/onnx/models/raw/e77240a62df68ed13e3138a5812553a552b857bb/text/machine_comprehension/bidirectional_attention_flow/model/bidaf-9.onnx"

WORKING_DIR = os.environ.get("TRT_WORKING_DIR") or os.path.dirname(os.path.realpath(__file__))
MODEL_DIR  = os.path.join(WORKING_DIR, "models")
RAW_MODEL_PATH = os.path.join(MODEL_DIR, "bidaf-9.onnx")
TRT_MODEL_PATH = os.path.join(MODEL_DIR, "bidaf-9-trt.onnx")

def _do_graph_surgery(raw_model_path, trt_model_path):
    graph = gs.import_onnx(onnx.load(raw_model_path))

    # Replace unsupported Hardmax with our CustomHardmax op
    # 用自定义的op 'CustomHardmax' 替换trt不支持的'Hardmax' op
    # 这里添加的op类型为CustomHardmax，在onnx的graph中，仅仅表示一个符号，不能用onnx做推理。
    # 如果需要onnx做推理，则需要再onnx-runtime实现该op的运算规则。
    # 在tensorrt中解析到该op类型时，需要将CustomHardmax 类型实现为一个op plugin插件！！
    for node in graph.nodes:
        if node.op == 'Hardmax':
            node.op = 'CustomHardmax'
            hardmax_node = node

    # The original onnx model also uses another unsupported op called "Compress".
    # "Compress" returns values from the first tensor for all indices which evaluate to
    # True in the second tensor. In our case the second Tensor is the output of Hardmax,
    # so exactly one index will evaluate to true because the value at it will be 1, and
    # all other values will be 0. We can achieve the same result as "Compress" by taking the
    # dot product of our value tensor and the Hardmax output.
    #
    # So, we will replace the subgraph Compress(Transpose_29, Cast(Reshape(Hardmax)))
    # with the subgraph Einsum(Transpose_29, Hardmax) where the equation in Einsum takes the dot product.
    # 用'Einsum' op, 替换 两个'Transpose_29, Hardmax' 两个op
    node_by_name = {node.name : node for node in graph.nodes}
    transpose_node = node_by_name['Transpose_29']
    compress_node  = node_by_name['Compress_31']

    einsum_node = gs.Node(
        'Einsum',
        'Dot_of_Hardmax_and_Transpose',
        attrs={'equation': 'ij,ij->i'}, # "Dot product" of 2d tensors
        inputs=[hardmax_node.outputs[0], transpose_node.outputs[0]],
        outputs=[compress_node.outputs[0]]
    )
    graph.nodes.append(einsum_node) # 新加节点直接入nodes 列表

    # Separate the old subgraph which will be deleted with graph.cleanup()
    # 通过clean删除已经分离出来的子图
    #  A    B
    #   \  /
    #     C
    #     |
    #     D
    # A: transpose_node, B：hardmax_node， D: compress_node
    #  通过一下三个clean, 则会保留A之前，B之前，D之后的op，而删除C节点。
    hardmax_node.o().inputs.clear()
    transpose_node.o().inputs.clear()
    compress_node.outputs.clear()

    # Also remove the CategoryMapper nodes which convert strings to integers as the first step in the model.
    # 删除字符串到整数的映射op
    # We need to convert the following structure:
    #
    #      Input as                        Converted to
    #   String tokens                     Integer tokens
    #  ---------------->[CategoryMapper]------------------>[Rest of Model]
    #
    # into the following:
    #
    #      Input as
    #   Integer tokens
    #  ------------------>[Rest of Model]
    #
    # Later we will feed the model the integer tokens directly.
    # Note: list conversion is necessary because we modify graph.nodes in the for loop.
    category_mapper_nodes = [node for node in graph.nodes if node.op == 'CategoryMapper']
    for node in category_mapper_nodes:
        # Remove CategoryMapper node from onnx graph
        # # 删除节点
        graph.nodes.remove(node)

        # Also remove references its inputs in the graph's inputs
        for input_tensor in node.inputs:
            graph.inputs.remove(input_tensor)

        # The graph's new inputs are the Integer tokens output by CategoryMapper
        # 添加输入
        graph.inputs += node.outputs

        # Save String->Int map 保存word to index的映射，保存到文件
        with open(node.name + ".json", "w") as fp:
            json.dump(node.attrs, fp)

    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), trt_model_path)


def make_trt_compatible_onnx_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(RAW_MODEL_PATH):
        wget.download(MODEL_URL, out=RAW_MODEL_PATH)
        print("\nDownloaded BiDAF model from Onnx Model Zoo")
    print("Performing graph surgery on Onnx Model Zoo BiDAF model")
    _do_graph_surgery(RAW_MODEL_PATH, TRT_MODEL_PATH)
    print("Graph Surgery complete!")


def main():
    if os.path.exists(TRT_MODEL_PATH):
        print("TRT-compatible onnx model already exists!")
    else:
        print("TRT-compatible onnx model not found, generating...")
        make_trt_compatible_onnx_model()


if __name__ == "__main__":
    main()
