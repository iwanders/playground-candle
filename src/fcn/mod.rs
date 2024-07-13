use crate::candle_util::prelude::*;
use crate::candle_util::MaxPoolLayer;
use crate::candle_util::SequentialT;
use candle_core::bail;
use candle_core::{DType, Device, Module, Result, Tensor, Var, D};
use candle_nn::{Activation, ConvTranspose2dConfig, Dropout, VarBuilder, VarMap};
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

/// VGG Network, only the convolution section
///
/// https://arxiv.org/pdf/1409.1556
pub struct VGG16 {
    network: SequentialT,
    device: Device,
}

impl VGG16 {
    pub fn from_path<P>(path: P, device: &Device) -> Result<Self>
    where
        P: AsRef<std::path::Path> + Copy,
    {
        // https://github.com/huggingface/candle/pull/869

        // let m = crate::candle_util::load_from_safetensors(path, &device)?;
        // for (k, v) in m.iter() {
        // println!("{k:?} ");
        // }

        let vs = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
        let vgg16 = VGG16::new(vs, device)?;
        Ok(vgg16)
    }

    pub fn new(vs: VarBuilder, device: &Device) -> Result<Self> {
        let mut network = SequentialT::new();

        // The convolution stride is fixed to 1 pixel; the spatial padding of conv. layer input is
        // such that the spatial resolution is preserved after convolution, i.e. the padding is 1
        // pixel for 3 × 3 conv. layers.
        let padding_one = candle_nn::conv::Conv2dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
        };

        let vs = vs.pp("features");

        // Block 1
        network.add(candle_nn::conv2d(3, 64, 3, padding_one, vs.pp("0"))?); // 0
        network.add(Activation::Relu); // 1
        network.add(candle_nn::conv2d(64, 64, 3, padding_one, vs.pp("2"))?); // 2
        network.add(Activation::Relu); // 3
        network.add(MaxPoolLayer::new(2)?); // 4

        // Block 2
        network.add(candle_nn::conv2d(64, 128, 3, padding_one, vs.pp("5"))?); // 5
        network.add(Activation::Relu); // 6
        network.add(candle_nn::conv2d(128, 128, 3, padding_one, vs.pp("7"))?); // 7
        network.add(Activation::Relu); // 8
        network.add(MaxPoolLayer::new(2)?); // 9

        // Block 3
        network.add(candle_nn::conv2d(128, 256, 3, padding_one, vs.pp("10"))?); // 10
        network.add(Activation::Relu); // 11
        network.add(candle_nn::conv2d(256, 256, 3, padding_one, vs.pp("12"))?); // 12
        network.add(Activation::Relu); // 13
        network.add(candle_nn::conv2d(256, 256, 3, padding_one, vs.pp("14"))?); // 14
        network.add(Activation::Relu); // 15
        network.add(MaxPoolLayer::new(2)?); // 16

        // Block 4
        network.add(candle_nn::conv2d(256, 512, 3, padding_one, vs.pp("17"))?); //17
        network.add(Activation::Relu); // 18
        network.add(candle_nn::conv2d(512, 512, 3, padding_one, vs.pp("19"))?); // 19
        network.add(Activation::Relu); // 20
        network.add(candle_nn::conv2d(512, 512, 3, padding_one, vs.pp("21"))?); // 21
        network.add(Activation::Relu); // 22
        network.add(MaxPoolLayer::new(2)?); // 23

        // Block 5
        network.add(candle_nn::conv2d(512, 512, 3, padding_one, vs.pp("24"))?); // 24
        network.add(Activation::Relu); // 25
        network.add(candle_nn::conv2d(512, 512, 3, padding_one, vs.pp("26"))?); // 26
        network.add(Activation::Relu); // 27
        network.add(candle_nn::conv2d(512, 512, 3, padding_one, vs.pp("28"))?); // 28
        network.add(Activation::Relu);
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

const PASCAL_VOC_CLASSES: usize = 21;

pub struct FCN32s {
    vgg16: VGG16,
    network: SequentialT,
    device: Device,
}

impl FCN32s {
    pub fn new(vgg16: VGG16, vs: VarBuilder, device: &Device) -> Result<Self> {
        let mut network = SequentialT::new();

        // After https://raw.githubusercontent.com/shelhamer/fcn.berkeleyvision.org/master/voc-fcn32s/train.prototxt
        // into https://ethereon.github.io/netscope/#/editor

        // Tack on the head.
        let deconv_config = candle_nn::conv::ConvTranspose2dConfig {
            padding: 1,
            output_padding: 1,
            stride: 2,
            dilation: 1,
        };
        let norm_config = candle_nn::batch_norm::BatchNormConfig::default();


        network.add(Activation::Relu);

        // deconv1
        network.add(candle_nn::conv::conv_transpose2d(512, 512, 3, deconv_config,vs.pp("deconv1"))?);
        network.add(candle_nn::batch_norm::batch_norm(512, norm_config, vs.pp("deconv1_norm"))?);

        // deconv2
        network.add(candle_nn::conv::conv_transpose2d(512, 256, 3, deconv_config,vs.pp("deconv2"))?);
        network.add(candle_nn::batch_norm::batch_norm(256, norm_config, vs.pp("deconv2_norm"))?);

        // deconv3
        network.add(candle_nn::conv::conv_transpose2d(256, 128, 3, deconv_config,vs.pp("deconv3"))?);
        network.add(candle_nn::batch_norm::batch_norm(128, norm_config, vs.pp("deconv3_norm"))?);

        // deconv4
        network.add(candle_nn::conv::conv_transpose2d(128, 64, 3, deconv_config,vs.pp("deconv4"))?);
        network.add(candle_nn::batch_norm::batch_norm(64, norm_config, vs.pp("deconv4_norm"))?);

        // deconv5
        network.add(candle_nn::conv::conv_transpose2d(64, 32, 3, deconv_config,vs.pp("deconv5"))?);
        network.add(candle_nn::batch_norm::batch_norm(32, norm_config, vs.pp("deconv5_norm"))?);


        network.add(candle_nn::conv2d(
            32,
            PASCAL_VOC_CLASSES,
            1,
            Default::default(),
            vs.pp(format!("classifier")),
        )?);

        Ok(Self {
            vgg16,
            network,
            device: device.clone(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let z = self.vgg16.forward(x)?;
        self.network.forward(&z)
    }
}

#[cfg(test)]
mod test {

    use super::*;

    use crate::candle_util::{approx_equal, error_unwrap};

    #[test]
    fn test_vgg_load() -> Result<()> {
        let device = Device::Cpu;

        let vgg_dir = std::env::var("VGG_MODEL");
        let vgg = if let Ok(model_file) = vgg_dir {
            let vgg = VGG16::from_path(&model_file, &device);
            error_unwrap!(vgg)
        } else {
            let varmap = VarMap::new();
            let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            let vgg = VGG16::new(vs, &device);
            error_unwrap!(vgg)
        };

        // Create a dummy image.
        // Image is 224x224, 3 channels,  make it 0.5 gray
        let gray = Tensor::full(0.5f32, (3, 224, 224), &device)?;

        // Make a batch of two of these.
        let batch = Tensor::stack(&[&gray, &gray], 0)?;

        // Pass that into the network..
        let r = vgg.forward(&batch);

        // Unwrap it
        let r = error_unwrap!(r);
        eprintln!("r shape: {:?}", r.shape());
        assert_eq!(r.shape().dims4()?, (2, 512, 7, 7));

        Ok(())
    }
    #[test]
    fn test_fcn_instantiate() -> Result<()> {
        let device = Device::Cpu;
        // let device = Device::new_cuda(0)?;

        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let vgg16 = VGG16::new(vs, &device);

        let vgg16 = error_unwrap!(vgg16);

        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let network = FCN32s::new(vgg16, vs, &device);

        let network = error_unwrap!(network);

        // Create a dummy image.
        // Image is 448x448
        // 0.5 gray
        let gray = Tensor::full(0.5f32, (3, 224, 224), &device)?;

        // Make a batch of two of these.
        let batch = Tensor::stack(&[&gray, &gray], 0)?;

        // Pass that into the network..
        let r = network.forward(&batch);

        // Do this here to get nice error message without newlines.
        let r = error_unwrap!(r);
        let _r = r;
        eprintln!("r shape: {:?}", _r.shape());

        Ok(())
    }
}