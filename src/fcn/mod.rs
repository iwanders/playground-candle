// use crate::candle_util::prelude::*;
use crate::candle_util::MaxPoolLayer;
use crate::candle_util::SequentialT;
// use candle_core::bail;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Activation, ConvTranspose2dConfig, VarBuilder, VarMap};

use rand::prelude::*;
use rand_xorshift::XorShiftRng;
/*
use candle_core::IndexOp;
use candle_nn::ops::{log_softmax, softmax};
use candle_nn::Optimizer;
use candle_nn::{Activation, Dropout};

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
        let x = x.to_device(&self.device)?;
        self.network.forward(&x)
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
        let deconv_config = ConvTranspose2dConfig {
            padding: 1,
            output_padding: 1,
            stride: 2,
            dilation: 1,
        };
        let norm_config = candle_nn::batch_norm::BatchNormConfig::default();

        network.add(Activation::Relu);

        // deconv1
        network.add(candle_nn::conv::conv_transpose2d(
            512,
            512,
            3,
            deconv_config,
            vs.pp("deconv1"),
        )?);
        network.add(candle_nn::batch_norm::batch_norm(
            512,
            norm_config,
            vs.pp("deconv1_norm"),
        )?);

        // deconv2
        network.add(candle_nn::conv::conv_transpose2d(
            512,
            256,
            3,
            deconv_config,
            vs.pp("deconv2"),
        )?);
        network.add(candle_nn::batch_norm::batch_norm(
            256,
            norm_config,
            vs.pp("deconv2_norm"),
        )?);

        // deconv3
        network.add(candle_nn::conv::conv_transpose2d(
            256,
            128,
            3,
            deconv_config,
            vs.pp("deconv3"),
        )?);
        network.add(candle_nn::batch_norm::batch_norm(
            128,
            norm_config,
            vs.pp("deconv3_norm"),
        )?);

        // deconv4
        network.add(candle_nn::conv::conv_transpose2d(
            128,
            64,
            3,
            deconv_config,
            vs.pp("deconv4"),
        )?);
        network.add(candle_nn::batch_norm::batch_norm(
            64,
            norm_config,
            vs.pp("deconv4_norm"),
        )?);

        // deconv5
        network.add(candle_nn::conv::conv_transpose2d(
            64,
            32,
            3,
            deconv_config,
            vs.pp("deconv5"),
        )?);
        network.add(candle_nn::batch_norm::batch_norm(
            32,
            norm_config,
            vs.pp("deconv5_norm"),
        )?);

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
        let x = x.to_device(&self.device)?;
        let z = self.vgg16.forward(&x)?;
        self.network.forward(&z)
    }
}


fn lines_from_file(filename: impl AsRef<std::path::Path>) -> std::result::Result<std::collections::HashSet<String>, anyhow::Error> {
    use std::io::BufRead;
    let file = std::fs::File::open(filename)?;
    let buf = std::io::BufReader::new(file);
    let mut ids = vec![];
    for l in buf.lines() {
        let l = l?;
        let s = l.split_whitespace().map(|z| z.to_owned()).collect::<Vec<_>>();
        if s.len() == 1 {
            ids.push(s[0].clone());
        } else if s[1] == "1" {
            ids.push(s[0].clone());
        }
        
    }
    Ok(ids.drain(..).collect())
}


pub fn gather_ids(
    path: &std::path::Path,
    classes: &[&str],
) -> std::result::Result<(std::collections::HashSet<String>, std::collections::HashSet<String>), anyhow::Error> {
    let mut seg_dir = path.to_owned().join("ImageSets/Segmentation");

    let train_ids = lines_from_file(&seg_dir.join("train.txt"))?;
    let val_ids = lines_from_file(&seg_dir.join("val.txt"))?;
    println!("val_ids: {val_ids:?}");

    let mut main_dir = path.to_owned().join("ImageSets/Main");

    let mut collected_train: std::collections::HashSet<String> = Default::default();
    let mut collected_val: std::collections::HashSet<String> = Default::default();
    for c in classes {
        let class_ids = lines_from_file(&main_dir.join(format!("{c}_trainval.txt")))?;
        for c in class_ids {
            if train_ids.contains(&c) {
                collected_train.insert(c);
            } else  if val_ids.contains(&c) {
                collected_val.insert(c);
            }
        }
    }

    Ok((collected_train, collected_val))
}

pub struct SampleTensor {
    pub sample: voc_dataset::Sample,
    pub image: Tensor,
    pub mask: Tensor,
}

impl SampleTensor {
    pub fn load(
        sample: voc_dataset::Sample,
        device: &Device,
    ) -> std::result::Result<SampleTensor, anyhow::Error> {
        println!("sample: {sample:?}");
        let image = Tensor::full(0.5f32, (3, 224, 224), device)?;
        let mask = image.clone();
        Ok(Self {
            sample,
            image,
            mask,
        })
    }
}

pub fn fit(
    fcn: &FCN32s,
    path: &std::path::Path,
    device: &Device,
) -> std::result::Result<(), anyhow::Error> {

    // Okay, that leaves 2913 images that have a segmentation mask.
    // That's 2913 (images) * 224 (w) * 224 (h) * 3 (channels) * 4 (float) = 1 902 028 800
    // 1.9 GB, that fits in RAM and VRAM, so lets convert all the images to tensors.

    /*
    const MINIBATCH_SIZE: usize = 20; // from the paper, p6.
    let batch_count = v.len() / MINIBATCH_SIZE;

    let s = SampleTensor::load(v[0].clone(), device)?;

    let mut rng = XorShiftRng::seed_from_u64(1);
    let mut shuffled_indices: Vec<usize> = (0..batch_count).collect();

    println!("Samples segmented: {}", v.len());
    */
    todo!()
}
use anyhow::{Context};
pub fn main() -> std::result::Result<(), anyhow::Error> {
    let args = std::env::args().collect::<Vec<String>>();

    let voc_dir = std::path::PathBuf::from(&args[1]);
    let (train, val) = gather_ids(&voc_dir, &["person", "cat", "bicycle", "bird"])?;
    println!("Train {}, val: {}", train.len(), val.len());

    let samples = voc_dataset::load(&voc_dir)?;

    let mut samples_train = vec![];
    let mut samples_val = vec![];
    println!("Samples start: {}", samples.len());
    for s in samples {
        if let Some(segmentation) = s.annotation.segmented {
            if segmentation {
                let name = s.image_path.file_stem().map(|z|z.to_str().map(|z|z.to_owned())).flatten().with_context(|| "failed to convert path")?;
                if train.contains(&name) {
                    samples_train.push(s);
                } else if val.contains(&name) {
                    samples_val.push(s);
                }
            }
        }
    }
    println!("train: {}, val: {}", samples_train.len(), samples_val.len());

    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let vgg16 = VGG16::new(vs, &device)?;

    // let vgg16 = error_unwrap!(vgg16);

    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let network = FCN32s::new(vgg16, vs, &device)?;

    fit(&network, &voc_dir, &device)?;
    Ok(())
}

#[cfg(test)]
mod test {

    use super::*;

    use crate::candle_util::{approx_equal, error_unwrap};
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
        // Image is 224x224, 3 channels,  make it 0.5 gray
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
