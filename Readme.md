

# 编译说明

tensorrt 包含开源和闭源部分。
github上能下载到的为开源部分，从[官网](https://developer.nvidia.cn/nvidia-tensorrt-download)下载的tar包为闭源部分。
开源包编译需要用到闭源包的某些库
从源码编译TensorRT：
1. cuda
2. cudnn ， 下载的include和lib文件放到cuda路径
3. 配置闭源包的变量路径：
    - export TRT_LIBPATH=`pwd`/TensorRT-8.6.1.6
4. 编译开源部分，mkdir build & cd build, 使用编译命令为：cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH/lib -DTRT_OUT_DIR=`pwd`/out
    - 注意编译cmd的-DTRT_LIB_DIR=$TRT_LIBPATH/**lib**带有lib
    - make -j$(nproc)

    - cuda arch的warning，可以在`project(TensorRT`行之前加以下设置。
    ```bash
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75)
        endif()
    ```

    - 注意开源trt的版本，要与submodule的onnx版本对应，不然会报类似`error: ‘OnnxParserFlag’ has not been declared`的错误
