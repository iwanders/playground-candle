use crate::candle_util::prelude::*;
use crate::candle_util::SequentialT;
use candle_core::bail;
use candle_core::{DType, Device, Module, Result, Tensor, Var, D};
use candle_nn::{VarBuilder, VarMap, Activation};
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
 https://arxiv.org/pdf/1506.02640

    They also mention Fast Yolo with only 9 convolution layers instead of 24, that may be easier to
    train, but it not yet clear to me which layers from figure 3 are used for that flavour from the
    paper.

    Fig 3;
        Conv. Layer
            What does the '7x7x64-s-2' 64s2 part mean?
            7x7 kernel makes sense
            64 output channels?
            s2 stride, probably
            Fig 3 is the 24 convolutional layers

        What does the
            'we train the convolutional layers at half resolution; 224x244 and double for detection' 
        Mean? The first input layer is 448, so input needs to be 448
*/

/*
    https://pytorch.org/vision/main/generated/torchvision.transforms.PILToTensor.html
        Converts a PIL Image (H x W x C) to a Tensor of shape (C x H x W).
*/

pub struct YoloV1 {
    network: SequentialT,
    device: Device,
}

const INPUT_IMAGE_SIZE: usize = 448;

impl YoloV1 {
    pub fn new(vs: VarBuilder, device: &Device) -> Result<Self> {
        let mut network = SequentialT::new();

        let s2 = candle_nn::conv::Conv2dConfig {
            padding: 0,
            stride: 2,
            dilation: 1,
            groups: 1,
        };

        // Block 1: convolution layer.
        network.add(candle_nn::conv2d(3, 64, 7, s2, vs.pp(format!("b1_c0")))?);
        network.add(Activation::LeakyRelu(0.1)); // confirmed with test_leaky_relu.
        network.add(MaxPoolLayer::new(2)?);
        
        // Block 2: Convolution layer
        network.add(candle_nn::conv2d(64, 192, 3, Default::default(), vs.pp(format!("b2_c0")))?);
        network.add(Activation::LeakyRelu(0.1));
        network.add(MaxPoolLayer::new(2)?);

        // Block 3: Convolution layer
        network.add(candle_nn::conv2d(192, 128, 1, Default::default(), vs.pp(format!("b3_c0")))?);
        network.add(Activation::LeakyRelu(0.1));
        network.add(candle_nn::conv2d(128, 256, 3, Default::default(), vs.pp(format!("b3_c1")))?);
        network.add(Activation::LeakyRelu(0.1));
        network.add(candle_nn::conv2d(256, 256, 1, Default::default(), vs.pp(format!("b3_c2")))?);
        network.add(Activation::LeakyRelu(0.1));
        network.add(candle_nn::conv2d(256, 512, 3, Default::default(), vs.pp(format!("b3_c3")))?);
        network.add(MaxPoolLayer::new(2)?);

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
    fn test_leaky_relu() -> Result<()> {
        // Eq 2, p3.
        //    f(x) = {   x, if x > 0
        //               0.1x, otherwise.
        // Candle implements as LeakyRelu as:
        //     let zeros = xs.zeros_like()?;
        //     xs.maximum(&zeros)? + xs.minimum(&zeros)? * negative_slope
        // Does that need a minus sign or not.
        let device = Device::Cpu;
        let v = |v: f32| Tensor::full(v, (), &device);

        let paper_relu = |s: f32, v: f32| if v > 0.0 { v } else {s * v};
        
        approx_equal!(paper_relu(0.1, 1.0), Activation::LeakyRelu(0.1).forward(&v(1.0)?)?.to_scalar::<f32>()?, 0.01);
        approx_equal!(paper_relu(0.1, -1.0), Activation::LeakyRelu(0.1).forward(&v(-1.0)?)?.to_scalar::<f32>()?, 0.01);
        approx_equal!(paper_relu(0.1, 10.0), Activation::LeakyRelu(0.1).forward(&v(10.0)?)?.to_scalar::<f32>()?, 0.01);
        approx_equal!(paper_relu(0.1, -10.0), Activation::LeakyRelu(0.1).forward(&v(-10.0)?)?.to_scalar::<f32>()?, 0.01);

        Ok(())
        
    }
    #[test]
    fn instantiate() -> Result<()> {
        let device = Device::Cpu;
        // let device = Device::new_cuda(0)?;

        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let network = YoloV1::new(vs, &device);
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
