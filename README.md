# Playground Candle

I wanted to explore [Candle](https://github.com/huggingface/candle), this repo holds my toy
code that does that.

The `manual.rs` file implements a linear network and tracining, based on [this article][mnist_stepbystep], this implements the layers and computation manually.

The `network.rs` uses more of the features provided by `Candle` to create the network and train it using `SGD` or `AdamW`.

[mnist_stepbystep]: https://medium.com/@koushikkushal95/mnist-hand-written-digit-classification-using-neural-network-from-scratch-54da85712a06
