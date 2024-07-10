use crate::candle_util::prelude::*;
use crate::candle_util::SequentialT;
use candle_core::bail;
use candle_core::{DType, Device, Module, Result, Tensor, Var, D};
use candle_nn::{VarBuilder, VarMap, Activation, Dropout};
use crate::candle_util::MaxPoolLayer;
/*
use candle_core::IndexOp;
use candle_nn::ops::{log_softmax, softmax};
use candle_nn::Optimizer;
use candle_nn::{Activation, Dropout};

use rand::prelude::*;
use rand_xorshift::XorShiftRng;
*/

/* 
Fully Convolutional Networks for Semantic Segmentation
 https://arxiv.org/pdf/1411.4038
    Is a good read, and perhaps a better intermediate step than directly implementing Yolo.

    https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn32s/train.prototxt
    https://ethereon.github.io/netscope/#/editor

    Those parameters aren't really stated in the paper?
*/

/*
    https://pytorch.org/vision/main/generated/torchvision.transforms.PILToTensor.html
        Converts a PIL Image (H x W x C) to a Tensor of shape (C x H x W).
*/

pub struct FCN32s {
    network: SequentialT,
    device: Device,
}



impl FCN32s {
    pub fn new(vs: VarBuilder, device: &Device) -> Result<Self> {
        let mut network = SequentialT::new();

        // Block 1
        network.add(candle_nn::conv2d(3, 64, 3, Default::default(), vs.pp(format!("b1_c0")))?);
        network.add(Activation::Relu);
        network.add(candle_nn::conv2d(64, 64, 3, Default::default(), vs.pp(format!("b1_c1")))?);
        network.add(Activation::Relu);
        network.add(MaxPoolLayer::new(2)?);

        // Block 2
        network.add(candle_nn::conv2d(64, 128, 3, Default::default(), vs.pp(format!("b2_c0")))?);
        network.add(Activation::Relu);
        network.add(candle_nn::conv2d(128, 128, 3, Default::default(), vs.pp(format!("b2_c1")))?);
        network.add(Activation::Relu);
        network.add(MaxPoolLayer::new(2)?);
        
        // Block 3
        network.add(candle_nn::conv2d(128, 256, 3, Default::default(), vs.pp(format!("b3_c0")))?);
        network.add(Activation::Relu);
        network.add(candle_nn::conv2d(256, 256, 3, Default::default(), vs.pp(format!("b3_c1")))?);
        network.add(Activation::Relu);
        network.add(candle_nn::conv2d(256, 256, 3, Default::default(), vs.pp(format!("b3_c1")))?);
        network.add(Activation::Relu);
        network.add(MaxPoolLayer::new(2)?);
        
        // Block 4
        network.add(candle_nn::conv2d(256, 512, 3, Default::default(), vs.pp(format!("b4_c1")))?);
        network.add(Activation::Relu);
        network.add(candle_nn::conv2d(512, 512, 3, Default::default(), vs.pp(format!("b4_c2")))?);
        network.add(Activation::Relu);
        network.add(candle_nn::conv2d(512, 512, 3, Default::default(), vs.pp(format!("b4_c2")))?);
        network.add(Activation::Relu);
        network.add(MaxPoolLayer::new(2)?);

        // Block 5
        network.add(candle_nn::conv2d(512, 512, 3, Default::default(), vs.pp(format!("b5_c1")))?);
        network.add(Activation::Relu);
        network.add(candle_nn::conv2d(512, 512, 3, Default::default(), vs.pp(format!("b5_c2")))?);
        network.add(Activation::Relu);
        network.add(candle_nn::conv2d(512, 512, 3, Default::default(), vs.pp(format!("b5_c2")))?);
        network.add(Activation::Relu);
        network.add(MaxPoolLayer::new(2)?);

        // Block 6
        network.add(candle_nn::conv2d(512, 4096, 7, Default::default(), vs.pp(format!("b6_c1")))?);
        network.add(Activation::Relu);
        network.add(Dropout::new(0.5));
        
        // Block 7
        network.add(candle_nn::conv2d(4096, 4096, 1, Default::default(), vs.pp(format!("b7_c1")))?);
        network.add(Activation::Relu);
        network.add(Dropout::new(0.5));

        Ok(Self {
            network,
            device: device.clone(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.network.forward(x)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::candle_util::approx_equal;

    #[test]
    fn instantiate() -> Result<()> {
        let device = Device::Cpu;
        // let device = Device::new_cuda(0)?;

        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let network = FCN32s::new(vs, &device);
        let network = if let Err(e) = network {
            eprintln!("{}", e);
            // handle the error properly here
            assert!(false);
            unreachable!();
        } else {
            network.ok().unwrap()
        };



        // Create a dummy image.
        // Image is 448x448
        // 0.5 gray
        let gray = Tensor::full(0.5f32, (3, 448, 448), &device)?;

        // Make a batch of two of these.
        let batch = Tensor::stack(&[&gray, &gray], 0)?;
        
        // Pass that into the network..
        let r = network.forward(&batch);

        // Do this here to get nice error message without newlines.
        let r = if let Err(e) = r {
            eprintln!("{}", e);
            // handle the error properly here
            assert!(false);
            unreachable!();
        } else {
            r.ok().unwrap()
        };
        let _r = r;


        Ok(())
    }
}
