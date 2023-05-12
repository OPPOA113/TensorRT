/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TRT_HARDMAX_PLUGIN_H
#define TRT_HARDMAX_PLUGIN_H

#include "NvInferPlugin.h"
#include <cublas_v2.h>
#include <string>
#include <vector>

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2 and IPluginCreator classes.
// For requirements for overriden functions, check TensorRT API docs.

// 插件功能实现类
class HardmaxPlugin final : public nvinfer1::IPluginV2DynamicExt
{
public:
    // 构造函数
    HardmaxPlugin() = delete;
    HardmaxPlugin(int32_t axis);                                    // 序列化时的构造函数
    HardmaxPlugin(void const* serialData, size_t serialLength);     // 反序列化时的构造函数
    ~HardmaxPlugin() override;

    template <typename TDataType>
    TDataType const* pointer_const_cast(void const* const p);

    template <typename TDataType>
    TDataType* pointer_cast(void* p);

    int32_t getNbOutputs() const noexcept override;

    // 重要的成员函数.1: 根据输入确定输出维度。可返回表达式的维度。
    // DynamicExt plugins returns DimsExprs class instead of Dims
    nvinfer1::DimsExprs getOutputDimensions(int32_t index, nvinfer1::DimsExprs const* inputs, int32_t nbInputDims,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override; // determine output dims based on input info

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    // 重要的成员函数.4:    该plugin功能实现的接口，功能实现的cuda或cpu代码放入此。
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;
    // 重要的成员函数.2:    判断pos索引的输入 /输出数据是否符合指定的format格式和type数据类型。
    bool supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    char const* getPluginType() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    void destroy() noexcept override;

    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    void attachToContext(
        cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) noexcept override;

    void detachFromContext() noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;
    // 重要的成员函数.3: 判断输入和输出类型、数量是否正确
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

private:
    std::string mNamespace;

    // Number of elements in the axis along which hardmax is performed.
    int32_t mAxisSize{0};

    // Product of dimensions before and after mAxis.
    // mAxis轴，前后轴维度的乘积
    // For example, if the input dimensions are [3, 4, 5, 6, 7] and mAxis = 2,
    // then mDimProductOuter = 12 and mDimProductInner = 42.
    int32_t mDimProductOuter{1};
    int32_t mDimProductInner{1};

    cublasHandle_t mCublas;

    // Attributes
    // Axis along which to perform hardmax.
    // Can be negative initially, but once configurePlugin() is called it will
    // be converted to a positive axis.
    int32_t mAxis{-1};
};

// 插件Factory类，用于注册插件
class HardmaxPluginCreator : public nvinfer1::IPluginCreator
{
public:
    HardmaxPluginCreator();

    ~HardmaxPluginCreator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    // 通过PluginFieldCollection将plugin需要的权重和参数，并调用插件类的第一个构造函数创建plugin
    nvinfer1::IPluginV2DynamicExt* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2DynamicExt* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

private:
    // mFC op 字段集。与onnx对齐。字段名、数据、类型、 
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

#endif // TRT_HARDMAX_PLUGIN_H
