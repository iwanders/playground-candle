// use crate::candle_util::prelude::*;
use crate::candle_util::MaxPoolLayer;
use crate::candle_util::SequentialT;
// use candle_core::bail;
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::ops::log_softmax;
use candle_nn::{Activation, ConvTranspose2dConfig, ModuleT, Optimizer, VarBuilder, VarMap};

use rayon::prelude::*;

use rand::prelude::*;
use rand_xorshift::XorShiftRng;
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
    let mut seg_dir = path.to_owned().join("ImageSets/Segmentation");

    let train_ids = lines_from_file(&seg_dir.join("train.txt"))?;
    let val_ids = lines_from_file(&seg_dir.join("val.txt"))?;
    // println!("val_ids: {val_ids:?}");

    let mut main_dir = path.to_owned().join("ImageSets/Main");

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
    let r = image::DynamicImage::ImageRgb32F(back_img)
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
    let z = x.i(0)?;
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
        let back_to_img = tensor_to_mask(&segmentation)?;
        // back_to_img.save("/tmp/mask.png")?;
        // let image = Tensor::from_vec(img.into_vec(), (1, 224, 224), device)?;

        // Next, read the mask from somewhere.
        // let image = Tensor::full(0.5f32, (3, 224, 224), device)?;
        let mask = image.clone();
        Ok(Self {
            sample,
            image,
            segmentation,
        })
    }
}

pub fn binary_cross_entropy(truths: &Tensor, predicted: &Tensor) -> Result<Tensor> {
    // https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    // That's used in one implementation.
    // Another uses
    // y_comp = (truths==predicted)
    // nvalid = (truths != 255).sum()
    // bce = reduce_sum(-y_comp * log(predicted + 1e-7)) / nvalid
    //
    // Reading the BCELoss docs;
    // Oh, that help does explain why it's clamped, which is that + 1e-7 in the 'one'
    // implementation.
    //
    // Can we just drop half of the equation which means that mean(yn * log(xn)) remains?

    // Truths are always u32s, they're (batch, 1, 224, 224), of data type u32
    // Predicted is f32: (batch, 21, 224, 224).

    // So we need to explode the truths to (batch, 21, 224, 224) at data type f32.

    if truths.dtype() != DType::U32 {
        candle_core::bail!("truths has wrong type, got: {:?}", truths.dtype());
    }
    if predicted.dtype() != DType::F32 {
        candle_core::bail!("predicted has wrong type, got: {:?}", predicted.dtype());
    }

    let device = truths.device();

    // Conver the truths to a f32 thing.
    // Make a tensor that's full of ones.
    let ones_truths = Tensor::full(1.0f32, predicted.shape(), device)?.force_contiguous()?;
    let zeros_truth = Tensor::full(0.0f32, predicted.shape(), device)?.force_contiguous()?;
    // println!("predicted: {predicted:?}");
    // println!("truths: {truths:?}");
    let scatter_truth = truths.repeat((1, predicted.dims()[1], 1, 1))?;
    let truth_f32s = zeros_truth.scatter_add(&scatter_truth, &ones_truths, 1)?;



    // Truths are already lacking the mask... but was that smart? :/
    // Number of valid pixels is anything that in the truths doesn't have zero.
    // Make a zero vector.
    let zero_truths = Tensor::full(0u32, truths.shape(), device)?;
    // ne; out comes same shape, but u8, so convert back to u32.
    let not_zero = truths.ne(&zero_truths)?.to_dtype(DType::U32)?;
    let not_zero_count = not_zero.sum_all()?.to_scalar::<u32>()?;
    /*

    // Next, calculate the number of correct scalars.
    let correct_pixels = truths.eq(predicted)?;
    let correct_scalars = correct_pixels.to_dtype(DType::F32)?;
    // println!("correct_pixels: {correct_pixels:?}");

    let minus_one: Tensor = Tensor::new(-1.0f32, device)?;
    let floor: Tensor = Tensor::new(1e-7f32, device)?;
    let left_part = minus_one.broadcast_mul(&correct_scalars)?;
    // this feels wrong? Different classes have different weight? Maybe that doesn't matter as we
    // multiply on the left with the class?
    let predicted = predicted.to_dtype(DType::F32)?;
    let combined = (left_part * (predicted.broadcast_add(&floor))?.log())?;

    let combined_sum = combined.sum_all()?.to_scalar::<f32>()?;
    let loss = combined_sum / (not_zero_count as f32);

    // let correct_count = correct_u32s.sum_all()?.to_scalar::<u32>()?;

    // println!("combined: {combined:?}");
    // println!("not_zero_count: {not_zero_count:?}");
    // println!("loss: {loss:?}");
    // todo!()
    Tensor::new(loss, device)
    */
    // let delta = (predicted - truth_f32s)?;
    // let log_thing = 
    let minus_one: Tensor = Tensor::new(-1.0f32, device)?;
    let floor: Tensor = Tensor::new(1e-7f32, device)?;
    let left = minus_one.broadcast_mul(&truth_f32s)?;
    let combined = (left * (predicted.broadcast_add(&floor))?.log()?)?;
    Ok(combined)
}

pub fn fit(
    varmap: &VarMap,
    fcn: &FCN32s,
    path: &std::path::Path,
    device: &Device,
    sample_train: &Vec<SampleTensor>,
    sample_val: &Vec<SampleTensor>,
) -> std::result::Result<(), anyhow::Error> {
    // Okay, that leaves 2913 images that have a segmentation mask.
    // That's 2913 (images) * 224 (w) * 224 (h) * 3 (channels) * 4 (float) = 1 902 028 800
    // 1.9 GB, that fits in RAM and VRAM, so lets convert all the images to tensors.

    const MINIBATCH_SIZE: usize = 10; // from the paper, p6.
    let batch_count = sample_train.len() / MINIBATCH_SIZE;

    let mut rng = XorShiftRng::seed_from_u64(1);
    let mut shuffled_indices: Vec<usize> = (0..sample_train.len()).collect();

    // Collapse all sample_vals to a single tensor.
    let val_input = sample_val.iter().map(|z| &z.image).collect::<Vec<_>>();
    let val_input_tensor = Tensor::stack(&val_input, 0)?;
    let val_output = sample_val
        .iter()
        .map(|z| &z.segmentation)
        .collect::<Vec<_>>();
    let val_output_tensor = Tensor::stack(&val_output, 0)?;

    let learning_rate = 1e-4;
    // sgd doesn't support momentum, but it would be 0.9
    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), learning_rate)?;
    for epoch in 1..10 {
        shuffled_indices.shuffle(&mut rng);

        let mut sum_loss = 0.0f32;
        // create batches
        for (bi, batch_indices) in shuffled_indices.chunks(MINIBATCH_SIZE).enumerate() {
            let train_input = batch_indices
                .iter()
                .map(|i| &sample_train[*i].image)
                .collect::<Vec<_>>();
            let train_input_tensor = Tensor::stack(&train_input, 0)?;
            let train_output = batch_indices
                .iter()
                .map(|i| &sample_train[*i].segmentation)
                .collect::<Vec<_>>();
            let train_output_tensor = Tensor::stack(&train_output, 0)?;

            let train_input_tensor = train_input_tensor.to_device(device)?;
            let train_output_tensor = train_output_tensor.to_device(device)?;

            let logits = fcn.forward_t(&train_input_tensor, true)?;
            // let y_hat = logits.argmax_keepdim(1)?; // get maximum in the class dimension
            // println!("y_hat shape: {:?} t: {:?}", logits.shape(), logits.dtype());
            {
                // let img = batch_tensor_to_mask(0, &logits)?;
                // img.save(format!("/tmp/y_hat_{epoch}_{bi}.png"))?;
            }

            let batch_loss = binary_cross_entropy(&train_output_tensor, &logits)?;
            println!("Batch logits shape: {logits:?}");
            println!("Batch output truth: {train_output_tensor:?}");
            println!("Going into backwards step");
            sgd.backward_step(&batch_loss)?;
            let batch_loss_f32 = batch_loss.to_scalar::<f32>()?;
            sum_loss += batch_loss.to_scalar::<f32>()?;
            println!(
                "bi: {bi:?} / {}: {batch_indices:?}: {batch_loss_f32}",
                shuffled_indices.len() / MINIBATCH_SIZE
            );
        }
        let avg_loss = sum_loss / ((shuffled_indices.len() / MINIBATCH_SIZE) as f32);

        let test_accuracy = 0.0;
        println!(
            "{epoch:4} train loss {:8.5} test acc: {:5.2}%",
            avg_loss,
            100.0 * test_accuracy
        );
    }

    println!("Reached end of train... currently this is sad.");
    Ok(())
}
use anyhow::Context;
pub fn main() -> std::result::Result<(), anyhow::Error> {
    let device_storage = Device::Cpu;
    let device = Device::new_cuda(0)?;

    let args = std::env::args().collect::<Vec<String>>();

    let voc_dir = std::path::PathBuf::from(&args[1]);
    // let (train, val) = gather_ids(&voc_dir, &["person", "cat", "bicycle", "bird"])?;
    let (train, val) = gather_ids(&voc_dir, &["person", "cat", "bicycle", "bird"])?;
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

    println!("Building network");
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let vgg16 = VGG16::new(vs, &device)?;

    // let vgg16 = error_unwrap!(vgg16);

    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let network = FCN32s::new(vgg16, vs, &device)?;

    println!("Starting fit");
    fit(
        &varmap,
        &network,
        &voc_dir,
        &device,
        &tensor_samples_train,
        &tensor_samples_val,
    )?;
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

    #[test]
    fn test_crossentropy() -> Result<()> {
        let device = Device::Cpu;
        /*
            0 1 0
            2 2 2
            0 0 0
        */

        let truth = Tensor::from_slice(&[0u32, 1, 0, 2, 2, 2, 0, 0, 0], (1, 3, 3), &device)?;
        // let truth_batch = Tensor::stack(&[&gray, &gray], 0)?;
        let predicted = Tensor::from_slice(&[0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (1, 3, 3), &device)?;
        let loss = error_unwrap!(binary_cross_entropy(&truth, &predicted));
        let loss_a = loss.to_scalar::<f32>()?;
        println!("loss_a: {loss_a:?}");

        let predicted = Tensor::from_slice(&[0.0f32, 1.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0], (1, 3, 3), &device)?;
        let loss = error_unwrap!(binary_cross_entropy(&truth, &predicted));
        let loss_b = loss.to_scalar::<f32>()?;
        println!("loss_b: {loss_b:?}");
        assert!(loss_b < loss_a);
        Ok(())
    }
}
