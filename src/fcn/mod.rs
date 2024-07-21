
use crate::candle_util::MaxPoolLayer;
use crate::candle_util::SequentialT;
// use candle_core::bail;
use crate::candle_util::*;
use candle_core::{DType, Device, IndexOp, Result, Tensor};
// use candle_nn::ops::log_softmax;
use candle_nn::{Activation, ConvTranspose2dConfig, ModuleT, Optimizer, VarBuilder, VarMap};
use rayon::prelude::*;

use rand::prelude::*;
use rand_xorshift::XorShiftRng;


use clap::{Args, Parser, Subcommand};
use anyhow::Context;
/*
use candle_core::IndexOp;
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

impl ModuleT for VGG16 {
    fn forward_t(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let x = x.to_device(&self.device)?;
        self.network.forward_t(&x, train)
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

impl ModuleT for FCN32s {
    fn forward_t(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let x = x.to_device(&self.device)?;
        let z = self.vgg16.forward_t(&x, train)?;
        self.network.forward_t(&z, train)
    }
}

fn lines_from_file(
    filename: impl AsRef<std::path::Path>,
) -> std::result::Result<std::collections::HashSet<String>, anyhow::Error> {
    use std::io::BufRead;
    let file = std::fs::File::open(filename)?;
    let buf = std::io::BufReader::new(file);
    let mut ids = vec![];
    for l in buf.lines() {
        let l = l?;
        let s = l
            .split_whitespace()
            .map(|z| z.to_owned())
            .collect::<Vec<_>>();
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
) -> std::result::Result<
    (
        std::collections::HashSet<String>,
        std::collections::HashSet<String>,
    ),
    anyhow::Error,
> {
    let seg_dir = path.to_owned().join("ImageSets/Segmentation");

    let train_ids = lines_from_file(&seg_dir.join("train.txt"))?;
    let val_ids = lines_from_file(&seg_dir.join("val.txt"))?;
    // println!("val_ids: {val_ids:?}");

    let main_dir = path.to_owned().join("ImageSets/Main");

    let mut collected_train: std::collections::HashSet<String> = Default::default();
    let mut collected_val: std::collections::HashSet<String> = Default::default();
    for c in classes {
        let class_ids = lines_from_file(&main_dir.join(format!("{c}_trainval.txt")))?;
        for c in class_ids {
            if train_ids.contains(&c) {
                collected_train.insert(c);
            } else if val_ids.contains(&c) {
                collected_val.insert(c);
            }
        }
    }

    Ok((collected_train, collected_val))
}

pub struct SampleTensor {
    pub sample: voc_dataset::Sample,
    pub image: Tensor,
    pub segmentation: Tensor,
    pub segmentation_one_hot: Tensor,
    pub name: String,
}

pub fn rgbf32_to_image(v: &Tensor) -> anyhow::Result<image::Rgb32FImage> {
    // image is 28x28, input tensor is 1x784.
    let w = v.dims()[1] as u32;
    let h = v.dims()[2] as u32;
    let img = image::Rgb32FImage::from_vec(w, h, v.flatten_all()?.to_vec1()?)
        .with_context(|| "buffer to small")?;
    Ok(img)
}
pub fn img_tensor_to_png(v: &Tensor, path: &str) -> std::result::Result<(), anyhow::Error> {
    let back_img = rgbf32_to_image(v)?;
    let _r = image::DynamicImage::ImageRgb32F(back_img)
        .to_rgb8()
        .save(path)?;
    Ok(())
}

const CLASSESS: [&'static str; 21] = [
    "NONE",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
];
use image::Rgb;
const COLORS: [Rgb<u8>; 21] = [
    Rgb([0, 0, 0]),       // "NONE",
    Rgb([128, 0, 0]),     // "aeroplane",
    Rgb([0, 128, 0]),     // "bicycle",
    Rgb([128, 128, 0]),   // "bird",
    Rgb([0, 0, 128]),     // "boat",
    Rgb([128, 0, 128]),   // "bottle",
    Rgb([0, 128, 128]),   // "bus",
    Rgb([128, 128, 128]), // "car",
    Rgb([64, 0, 0]),      // "cat",
    Rgb([192, 0, 0]),     // "chair",
    Rgb([64, 128, 0]),    // "cow",
    Rgb([192, 128, 0]),   // "diningtable",
    Rgb([64, 0, 128]),    // "dog",
    Rgb([192, 0, 128]),   // "horse",
    Rgb([64, 128, 128]),  // "motorbike",
    Rgb([192, 128, 128]), // "person",
    Rgb([0, 64, 0]),      // "pottedplant",
    Rgb([128, 64, 0]),    // "sheep",
    Rgb([0, 192, 0]),     // "sofa",
    Rgb([128, 192, 0]),   // "train",
    Rgb([0, 64, 128]),    // "tvmonitor"
];
const BORDER: Rgb<u8> = Rgb([224, 224, 192]);

pub fn mask_to_tensor(img: &image::RgbImage, device: &Device) -> anyhow::Result<Tensor> {
    // let mut x = Tensor::full(0u32, (1, 224, 224), device)?;
    let mut index_vec = vec![];
    let mut previous = (Rgb([0, 0, 0]), 0);
    for y in 0..img.height() {
        for x in 0..img.width() {
            let p = img.get_pixel(x, y);
            if previous.0 == *p {
                index_vec.push(previous.1);
            } else {
                if BORDER == *p {
                    previous = (*p, 0 as u32);
                    index_vec.push(0);
                    continue;
                }
                if let Some(index) = COLORS.iter().position(|z| z == p) {
                    previous = (*p, index as u32);
                    index_vec.push(index as u32);
                } else {
                    anyhow::bail!("could not find color {p:?}")
                }
            }
        }
    }
    Ok(Tensor::from_vec(index_vec, (1, 224, 224), device)?)
}
pub fn tensor_to_mask(x: &Tensor) -> anyhow::Result<image::RgbImage> {
    let w = x.dims()[1] as u32;
    let h = x.dims()[2] as u32;
    let mut pixels = Vec::with_capacity(3 * w as usize * h as usize);
    for idx in x.flatten_all()?.to_vec1::<u32>()?.iter() {
        let c = COLORS
            .get(*idx as usize)
            .with_context(|| format!("no color for {idx:?}"))?;
        pixels.push(c.0[0]);
        pixels.push(c.0[1]);
        pixels.push(c.0[2]);
    }
    let img = image::RgbImage::from_vec(w, h, pixels).with_context(|| "buffer to small")?;
    Ok(img)
}

pub fn batch_tensor_to_mask(index: usize, x: &Tensor) -> anyhow::Result<image::RgbImage> {
    let z = x.i(index)?;
    tensor_to_mask(&z)
}

impl SampleTensor {
    pub fn load(
        sample: voc_dataset::Sample,
        device: &Device,
    ) -> std::result::Result<SampleTensor, anyhow::Error> {
        // println!("sample: {sample:?}");

        let img = ImageReader::open(&sample.image_path)?.decode()?;
        let img = img
            .resize_exact(224, 224, image::imageops::FilterType::Lanczos3)
            .to_rgb32f();
        let image = Tensor::from_vec(img.into_vec(), (3, 224, 224), device)?;
        // println!("s: {:?}", image.shape());
        // img_tensor_to_png(&image, "/tmp/foo.png")?;

        let segmentation_path = sample.image_path.clone();
        let segmentation_path = segmentation_path.parent().unwrap().parent().unwrap();
        let segmentation_path = segmentation_path.join("SegmentationClass");
        let segmentation_path = segmentation_path
            .join(sample.image_path.file_stem().unwrap())
            .with_extension("png");
        // println!("segmentaiton path: {segmentation_path:?}");
        use image::io::Reader as ImageReader;
        let img = ImageReader::open(&segmentation_path)?.decode()?;
        let img = img.resize_exact(224, 224, image::imageops::FilterType::Nearest);
        // let img = img.to_rgb32f();
        let segmentation = mask_to_tensor(&img.to_rgb8(), device)?;
        if false {
            let back_to_img = tensor_to_mask(&segmentation)?;
            back_to_img.save("/tmp/mask.png")?;
        }
        // let image = Tensor::from_vec(img.into_vec(), (1, 224, 224), device)?;

        let segmentation_one_hot = c_u32_one_hot(&segmentation, CLASSESS.len())?;
        // Next, read the mask from somewhere.
        // let image = Tensor::full(0.5f32, (3, 224, 224), device)?;
        // let mask = image.clone();

        let img_id = sample.image_path.file_stem().with_context(|| "no file stem")?;
        let name = img_id.to_str().with_context(|| "failed to convert to str")?.to_owned();

        Ok(Self {
            sample,
            image,
            name,
            segmentation,
            segmentation_one_hot,
        })
    }
}
#[derive(PartialEq, Eq, Debug, Copy, Clone)]
pub enum Segmentation{
    OneHot,
    Class
}
pub fn collect_minibatch_input_output(input: &[SampleTensor], batch_indices: &[usize], device: &Device, seg: Segmentation) -> anyhow::Result<(Tensor, Tensor)> {
    let train_input = batch_indices
        .iter()
        .map(|i| &input[*i].image)
        .collect::<Vec<_>>();
    let train_input_tensor = Tensor::stack(&train_input, 0)?;
    let train_output = batch_indices
        .iter()
        .map(|i| if seg == Segmentation::OneHot {
            &input[*i].segmentation_one_hot
        } else {
            &input[*i].segmentation
        })
        .collect::<Vec<_>>();
    let train_output_tensor = Tensor::stack(&train_output, 0)?;

    let train_input_tensor = train_input_tensor.to_device(device)?;
    let train_output_tensor = train_output_tensor.to_device(device)?;
    Ok((train_input_tensor, train_output_tensor))
}


pub fn fit(
    varmap: &VarMap,
    fcn: &FCN32s,
    device: &Device,
    sample_train: &Vec<SampleTensor>,
    sample_val: &Vec<SampleTensor>,
    settings: &FitSettings,
) -> std::result::Result<(), anyhow::Error> {
    // Okay, that leaves 2913 images that have a segmentation mask.
    // That's 2913 (images) * 224 (w) * 224 (h) * 3 (channels) * 4 (float) = 1 902 028 800
    // 1.9 GB, that fits in RAM and VRAM, so lets convert all the images to tensors.

    const MINIBATCH_SIZE: usize = 10; // 20 from the paper, p6.
    let batch_count = sample_train.len() / MINIBATCH_SIZE;

    let mut rng = XorShiftRng::seed_from_u64(1);
    let mut shuffled_indices: Vec<usize> = (0..sample_train.len()).collect();

    println!("Fitting with {settings:#?}");

    // sgd doesn't support momentum, but it would be 0.9
    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), settings.learning_rate)?;
    for epoch in 1..settings.max_epochs.unwrap_or(usize::MAX) {
        if epoch.rem_euclid(settings.save_interval) == 0 {
            // Save the checkpoint.
            let mut output_path = settings.save_path.clone();
            output_path.push(format!("fcn_{epoch}_lr{lr}.safetensors", lr=settings.learning_rate));
            varmap.save(&output_path)?;
        }

        shuffled_indices.shuffle(&mut rng);

        let mut sum_loss = 0.0f32;
        // Train with batches
        for (bi, batch_indices) in shuffled_indices.chunks(MINIBATCH_SIZE).enumerate() {
            let (train_input_tensor, train_output_tensor) = collect_minibatch_input_output(&sample_train, &batch_indices, device, Segmentation::OneHot)?;
            let logits = fcn.forward_t(&train_input_tensor, true)?;

            // Dump an image that was trained on.
            {
                // https://github.com/huggingface/candle/blob/2489a606fe1a66519da37e4237907926c1ee48a7/candle-examples/examples/vgg/main.rs#L58-L60
                // Should this be softmax?
                let sigm = candle_nn::ops::sigmoid(&logits)?;
                let zzz = sigm.argmax_keepdim(1)?; // get maximum in the class dimension
                let img = batch_tensor_to_mask(0, &zzz)?;
                let img_id = &sample_train[batch_indices[0]].name;
                img.save(format!("/tmp/train_{epoch}_{img_id}.png"))?;
            }

            let batch_loss = binary_cross_entropy_logits_loss(&logits, &train_output_tensor)?;
            // println!("Batch logits shape: {logits:?}");
            // println!("Batch output truth: {train_output_tensor:?}");
            // println!("Going into backwards step");
            sgd.backward_step(&batch_loss)?;
            let batch_loss_f32 = batch_loss.sum_all()?.to_scalar::<f32>()?;
            sum_loss += batch_loss_f32;
            println!(
                "      bi: {bi: >2?} / {}: {batch_loss_f32}",
                shuffled_indices.len() / MINIBATCH_SIZE
            );
        }
        let avg_loss = sum_loss / (batch_count as f32);


        // Validate, also in batches to avoid vram limits.
        let mut correct_pixels = 0;
        let mut pixel_count = 0;
        let sample_val_indices = (0..sample_val.len()).collect::<Vec<usize>>();
        for (bi, batch_indices) in sample_val_indices.chunks(MINIBATCH_SIZE).enumerate() {
            let (val_input_tensor, val_output_tensor) = collect_minibatch_input_output(&sample_val, &batch_indices, device, Segmentation::Class)?;
            let logits_val = fcn.forward_t(&val_input_tensor, false)?;
            let sigm = candle_nn::ops::sigmoid(&logits_val)?;
            let classified_pixels = sigm.argmax_keepdim(1)?; // get maximum in the class dimension
            {
                let img = batch_tensor_to_mask(0, &classified_pixels)?;
                let img_id = &sample_val[batch_indices[0]].name;
                img.save(format!("/tmp/val_{epoch:0>5}_{img_id}.png"))?;
            }
            // classified pixels is now (B, 1, 224, 224) u32
            // Validation tensor is also (B, 1, 224, 224) u32, so we can equal them.
            let eq = classified_pixels.eq(&val_output_tensor)?;
            let eq = eq.to_dtype(DType::U32)?;  // eq returns u8, so change that back to u32.
            // Accuracy is number of correct pixels.
            let correct = eq.sum_all()?;
            correct_pixels += correct.to_scalar::<u32>()?;
            // Right now we're counting black / no label as well... in the paper that's not what
            // they do, they ignore no label.
            pixel_count += eq.elem_count();

            if bi >= settings.validation_batch_limit.unwrap_or(usize::MAX) {
                break
            }
        }

        let test_accuracy = (correct_pixels as f64) / (pixel_count as f64);
        
        println!(
            "{epoch:4} train loss {:8.5} test acc: {:5.2}%",
            avg_loss,
            100.0 * test_accuracy
        );
    }

    println!("Reached end of train... currently this is sad.");
    Ok(())
}

pub fn create_data(
    voc_dir: &std::path::Path,
    categories: &[&str],
) -> std::result::Result<(Vec<SampleTensor>, Vec<SampleTensor>), anyhow::Error> {
    let device_storage = Device::Cpu;

    let (train, val) = gather_ids(&voc_dir, categories)?;
    println!("Train {}, val: {}", train.len(), val.len());

    let samples = voc_dataset::load(&voc_dir)?;

    let mut samples_train = vec![];
    let mut samples_val = vec![];
    println!("Samples start: {}", samples.len());
    for s in samples {
        if let Some(segmentation) = s.annotation.segmented {
            if segmentation {
                let name = s
                    .image_path
                    .file_stem()
                    .map(|z| z.to_str().map(|z| z.to_owned()))
                    .flatten()
                    .with_context(|| "failed to convert path")?;
                if train.contains(&name) {
                    samples_train.push(s);
                } else if val.contains(&name) {
                    samples_val.push(s);
                }
            }
        }
    }

    println!(
        "Samples train: {}, samples val: {}",
        samples_train.len(),
        samples_val.len()
    );

    println!("Loading train");
    let tensor_samples_train_results = samples_train
        .par_iter()
        .map(|s| SampleTensor::load(s.clone(), &device_storage))
        .collect::<Vec<_>>();
    let mut tensor_samples_train = vec![];
    for s in tensor_samples_train_results {
        tensor_samples_train.push(s?);
    }

    println!("Loading val");
    let tensor_samples_val_results = samples_val
        .par_iter()
        .map(|s| SampleTensor::load(s.clone(), &device_storage))
        .collect::<Vec<_>>();
    let mut tensor_samples_val = vec![];
    for s in tensor_samples_val_results {
        tensor_samples_val.push(s?);
    }

    Ok((tensor_samples_train, tensor_samples_val))
}




#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    data_path: std::path::PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Args, Debug, Clone)]
pub struct FitSettings {

    #[arg(long)]
    #[arg(default_value="/tmp/")]
    /// The directory into which to store checkpoints during training
    save_path: std::path::PathBuf,

    #[arg(long)]
    #[arg(default_value="1")]
    /// Save the checkpiont every save_interval epochs during training.
    save_interval: usize,

    #[arg(short, long)]
    #[arg(default_value="1e-4")]
    learning_rate: f64,

    #[arg(long)]
    /// There's a lot of validation data, if set this limits it to n batches.
    validation_batch_limit: Option<usize>,

    #[arg(long)]
    /// Limit the number of epochs, default unlimited
    max_epochs: Option<usize>,
}

#[derive(Subcommand)]
enum Commands {
    Fit(FitSettings),
}


pub fn main() -> std::result::Result<(), anyhow::Error> {
    let device = Device::new_cuda(0)?;

    println!("Building network");
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let vgg16 = VGG16::new(vs, &device)?;

    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let network = FCN32s::new(vgg16, vs, &device)?;

    let cli = Cli::parse();

    match &cli.command {
        Commands::Fit(s) => {
            let (tensor_samples_train, tensor_samples_val) =
                create_data(&cli.data_path, &["person", "cat", "bicycle", "bird"])?;

            println!("Starting fit");
            fit(
                &varmap,
                &network,
                &device,
                &tensor_samples_train,
                &tensor_samples_val,
                &s,
            )?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod test {

    use super::*;

    use crate::{approx_equal, error_unwrap};
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
