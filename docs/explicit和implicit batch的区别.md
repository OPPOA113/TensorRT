
# explicit batch 和 implicit batch

TensorRT 支持使用两种方式来指定网络的 Layout，即：explicit batch 和 implicit batch。所谓 显式 和 隐式 的差别就在于 Batch 这一维，即 显式 ==> NCHW，隐式 ==> CHW。在比较老版本的 TensorRT 中一般就用 implicit batch 隐式batch，而现在新的开始慢慢用 explicit batch 进行替代。这是为啥呢？这主要是因为，随着算法的不断发展，网络中新出现了很多新算子、新结构，有些时候需要 操控 batch 这个维度，这个时候如果使用 隐式batch 来开发，显然是不可行的。

隐式batch 模式不能胜任的场景主要包括：

● Reducing across the batch dimension ==> batch 维度上的规约操作，如 [N, C, H, W] -> [1, C, H, W]；

● Reshaping the batch dimension ==> batch 维度上的变化，如 [N, C, H, W] -> [M, C, H, W]；

● Transposing the batch dimension with another dimension ==> 涉及 batch 的维度变换操作，如 [N, C, H, W] -> [C, H, W, N]；

总的来说，就是涉及到 batch 的操作，explicit batch 就无能为力了，而显而易见的是，隐式batch 的开发难度会低一些，因为开发自始至终，你只需要关注 CHW。

**在context中，根据输入的是显式还是隐式，有不同推理api:**

context执行推理的几个api


|api|方式|status|batch方式|
|:---:|:---:|:---:|:---:|
|execute_async|异步|-|implicit（CHW）
|execute_async_v2|异步|DEPRECATED|explicit（NCHW）|
|execute_async_v3|异步|-|-|
|execute|同步|DEPRECATED|explicit（NCHW）|
|execute_v2|同步|-|explicit（NCHW）|

