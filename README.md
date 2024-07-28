# Playground Candle

I wanted to explore [Candle](https://github.com/huggingface/candle), this repo holds my toy code that does that.

## Mnist

The `manual.rs` file implements a linear network and training, based on [this article][mnist_stepbystep], this implements the layers, forward and backward propagation as well as the gradient descent computation manually using the `Tensor` object and its operations.

The `network.rs` uses more of the features provided by `Candle` to create the network and train it using `SGD` or `AdamW`.

For the linear configuration from `manual.rs` this runs one forward pass to compare the output to the one from the `manual.rs` implementation to confirm that the manual implementation is exactly what is happening in Candle's linear layers.

The convolution network parameters are taken from [this article][convolution_mnist] and achieves 98.5% accuracy against the validation set.

[mnist_stepbystep]: https://medium.com/@koushikkushal95/mnist-hand-written-digit-classification-using-neural-network-from-scratch-54da85712a06
[convolution_mnist]: https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/

## FCN

[Paper](https://arxiv.org/pdf/1411.4038)

Not sure if this is working correctly, training is problematic because of [this issue](https://github.com/huggingface/candle/issues/1241), with some changes in the candle's `backprop.rs` the vram usage is bearable (see [comment](https://github.com/huggingface/candle/issues/1241#issuecomment-2254229442)), but it seems that we also accumulate memory over time as we train, so manually restarting training from a checkpoint seems necessary.

Definitely needs preloaded vgg16 weights; `--validation-batch-limit 10 --save-val-mask --learning-rate 0.001 --minibatch-size 3`, more than 3 per minibatch results in out of vram.

Did run into an issue with stride=2 backpropagation, fixed in [this PR](https://github.com/huggingface/candle/pull/2337).

## Notes on Burn

Compile with
```
LIBTORCH_CXX11_ABI=0 CC=gcc-13 CXX=g++-13 LIBTORCH=<venv_with_cuda_pytorch>/site-packages/torch LD_LIBRARY_PATH=<venv_with_cuda_pytorch>/site-packages/torch/lib cargo r --release --example mnist  --features tch-gpu
```


## Misc
License is MIT OR Apache-2.0.

- https://github.com/huggingface/candle/pull/2337
