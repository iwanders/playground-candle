
use candle_core::bail;
use crate::candle_util::prelude::*;
use crate::candle_util::SequentialT;
use candle_core::{DType, Device, Module, Result, Tensor, Var, D};
use candle_nn::{VarBuilder, VarMap};
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
    
*/

pub struct YoloV1 {
    network: SequentialT,
    device: Device,
}

const INPUT_IMAGE_SIZE: usize = 448;

impl YoloV1 {
    pub fn new(vs: VarBuilder, device: &Device) -> Result<Self> {
        let mut network = SequentialT::new();

        Ok(Self {
            network,
            device: device.clone()
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn instantiate() {
        let device = Device::Cpu;
        // let device = Device::new_cuda(0)?;

        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let network = YoloV1::new(vs, &device);
        assert!(network.is_ok());
    }
}
