use crate::candle_util::SequentialT;
use crate::candle_util::*;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Activation, ModuleT, VarBuilder};

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

        // let vs = vs.pp("features");

        let padding_100 = candle_nn::conv::Conv2dConfig {
            padding: 100,
            stride: 1,
            dilation: 1,
            groups: 1,
        };

        // Block 1
        network.add(candle_nn::conv2d(3, 64, 3, padding_100, vs.pp("conv1_1"))?); // 0
        network.add(Activation::Relu); // 1
        network.add(candle_nn::conv2d(64, 64, 3, padding_one, vs.pp("conv1_2"))?); // 2
        network.add(Activation::Relu); // 3
        network.add(MaxPoolLayer::new(2)?); // 4

        // Block 2
        network.add(candle_nn::conv2d(
            64,
            128,
            3,
            padding_one,
            vs.pp("conv2_1"),
        )?); // 5
        network.add(Activation::Relu); // 6
        network.add(candle_nn::conv2d(
            128,
            128,
            3,
            padding_one,
            vs.pp("conv2_2"),
        )?); // 7
        network.add(Activation::Relu); // 8
        network.add(MaxPoolLayer::new(2)?); // 9

        // Block 3
        network.add(candle_nn::conv2d(
            128,
            256,
            3,
            padding_one,
            vs.pp("conv3_1"),
        )?); // 10
        network.add(Activation::Relu); // 11
        network.add(candle_nn::conv2d(
            256,
            256,
            3,
            padding_one,
            vs.pp("conv3_2"),
        )?); // 12
        network.add(Activation::Relu); // 13
        network.add(candle_nn::conv2d(
            256,
            256,
            3,
            padding_one,
            vs.pp("conv3_3"),
        )?); // 14
        network.add(Activation::Relu); // 15
        network.add(MaxPoolLayer::new(2)?); // 16

        // Block 4
        network.add(candle_nn::conv2d(
            256,
            512,
            3,
            padding_one,
            vs.pp("conv4_1"),
        )?); //17
        network.add(Activation::Relu); // 18
        network.add(candle_nn::conv2d(
            512,
            512,
            3,
            padding_one,
            vs.pp("conv4_2"),
        )?); // 19
        network.add(Activation::Relu); // 20
        network.add(candle_nn::conv2d(
            512,
            512,
            3,
            padding_one,
            vs.pp("conv4_3"),
        )?); // 21
        network.add(Activation::Relu); // 22
        network.add(MaxPoolLayer::new(2)?); // 23

        // Block 5
        network.add(candle_nn::conv2d(
            512,
            512,
            3,
            padding_one,
            vs.pp("conv5_1"),
        )?); // 24
        network.add(Activation::Relu); // 25
        network.add(candle_nn::conv2d(
            512,
            512,
            3,
            padding_one,
            vs.pp("conv5_2"),
        )?); // 26
        network.add(Activation::Relu); // 27
        network.add(candle_nn::conv2d(
            512,
            512,
            3,
            padding_one,
            vs.pp("conv5_3"),
        )?); // 28
        network.add(Activation::Relu);
        network.add(MaxPoolLayer::new(2)?);

        Ok(Self {
            network,
            device: device.clone(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.to_device(&self.device)?;
        self.network.forward(&x)
    }
}

impl ModuleT for VGG16 {
    fn forward_t(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let x = x.to_device(&self.device)?;
        self.network.forward_t(&x, train)
    }
}

#[cfg(test)]
mod test {

    use super::*;

    use crate::error_unwrap;
    use candle_core::Device;
    use candle_nn::{VarBuilder, VarMap};

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
        assert_eq!(r.shape().dims4()?, (2, 512, 13, 13));

        Ok(())
    }
}
