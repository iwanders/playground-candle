# Playground Candle

I wanted to explore [Candle](https://github.com/huggingface/candle), this repo holds my toy code that does that.

The `manual.rs` file implements a linear network and training, based on [this article][mnist_stepbystep], this implements the layers, forward and backward propagation as well as the gradient descent computation manually using the `Tensor` object and its operations.

The `network.rs` uses more of the features provided by `Candle` to create the network and train it using `SGD` or `AdamW`.

For the linear configuration from `manual.rs` this runs one forward pass to compare the output to the one from the `manual.rs` implementation to confirm that the manual implementation is exactly what is happening in Candle's linear layers.

The convolution network parameters are taken from [this article][convolution_mnist] and achieves 98.5% accuracy against the never-seen-during-training test set.

[mnist_stepbystep]: https://medium.com/@koushikkushal95/mnist-hand-written-digit-classification-using-neural-network-from-scratch-54da85712a06
[convolution_mnist]: https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/