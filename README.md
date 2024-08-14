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

Only resnet50 backbone works.

[Paper](https://arxiv.org/pdf/1411.4038)

Training is problematic because of [this issue](https://github.com/huggingface/candle/issues/1241), with some changes in the candle's `backprop.rs` the vram usage is bearable (see [comment](https://github.com/huggingface/candle/issues/1241#issuecomment-2254229442)), but it seems that we also accumulate memory over time as we train, so manually restarting training from a checkpoint seems necessary.

Did run into an issue with stride=2 backpropagation, fixed in [this PR](https://github.com/huggingface/candle/pull/2337).


Inference is working with the resnet backbone, with `fcn_resnet50_coco-1167a1af` created from the pytorch fcn_resnet50 weights:
```
cargo r --release -- /.../VOCdevkit/VOC2012 infer --load /.../converted_from_hub/fcn_resnet50_coco-1167a1af.safetensors --load-limit 100 --upscale
```

Training leaks memory but overfitting a single image seems to work:
```
cargo r --release -- /.../VOCdevkit/VOC2012 fit --validation-batch-limit 10 --save-val-mask --learning-rate 1e-2 --minibatch-size 15  --save-train-mask --create-post-train-mask  --train-on-first-n 1 --reduction mean --load /.../converted_from_hub/fcn_resnet50_coco-1167a1af.safetensors
```

Inference works relatively well, see the images in `./doc/` or below, depicting inference & desired target.

<img src="./doc/val_013_2010_002251_pred.png" width="20% "/> <img src="./doc/val_013_2010_002251_target.png" width="20% "/> <img src="./doc/val_018_2008_002358_pred.png" width="20% "/> <img src="./doc/val_018_2008_002358_target.png" width="20% "/>

## Notes on Burn

Compile with
```
LIBTORCH_CXX11_ABI=0 CC=gcc-13 CXX=g++-13 LIBTORCH=<venv_with_cuda_pytorch>/site-packages/torch LD_LIBRARY_PATH=<venv_with_cuda_pytorch>/site-packages/torch/lib cargo r --release --example mnist  --features tch-gpu
```


## Misc
License is MIT OR Apache-2.0.

- https://github.com/huggingface/candle/pull/2337
