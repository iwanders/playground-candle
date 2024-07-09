use super::util;
// use candle_nn::sequential::seq;
// use candle_core::IndexOp;
use candle_core::bail;
use candle_core::IndexOp;
use candle_core::{DType, Device, Module, Result, Tensor, Var, D};
use candle_nn::ops::{log_softmax, softmax};
use candle_nn::Optimizer;
use candle_nn::{Activation, Dropout};
use candle_nn::{VarBuilder, VarMap};

use crate::candle_util::prelude::*;
use crate::candle_util::{SequentialT, MaxPoolLayer};
use rand::prelude::*;
use rand_xorshift::XorShiftRng;

pub struct SoftmaxLayer {
    pub dim: usize,
}
impl SoftmaxLayer {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}
impl Module for SoftmaxLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        softmax(&xs, self.dim)
    }
}

pub struct ToImageLayer {}
impl Module for ToImageLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_sz, _img_dim) = xs.dims2()?;
        xs.reshape((b_sz, 1, 28, 28))
    }
}

pub struct FlattenLayer {
    pub dim: usize,
}
impl Module for FlattenLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.flatten_from(self.dim)
    }
}

pub struct SequentialNetwork {
    network: SequentialT,
    device: Device,
}

impl SequentialNetwork {
    fn create_network(vs: VarBuilder, config: &Config) -> Result<SequentialT> {
        let mut network = SequentialT::new();

        let mut sizes = config.linear_layers.clone();

        if config.convolution_layers.is_empty() {
            if !config.convolution_layers.is_empty() {
                bail!("convolution layers is not empty in linear config");
            }
            sizes.insert(0, 28 * 28);
            for l in 1..sizes.len() {
                let layer = candle_nn::linear(sizes[l - 1], sizes[l], vs.pp(format!("fc{l}")))?;
                network.add(layer);
                // Add sigmoid on all but the last layer.
                if l != (sizes.len() - 1) {
                    network.add(Activation::Sigmoid);
                }
            }
            // network.add(SoftmaxLayer::new(1));
        } else {
            if config.linear_layers.len() != 2 {
                bail!("linear layers must be two long in convolution config");
            }
            // https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
            network.add(ToImageLayer {});
            for l in 0..config.convolution_layers.len() {
                let (in_channels, out_channels, kernel) = config.convolution_layers[l];
                network.add(candle_nn::conv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    Default::default(),
                    vs.pp(format!("c{l}")),
                )?);
                network.add(Activation::Relu);
                network.add(MaxPoolLayer::new(2)?);
            }
            network.add(FlattenLayer { dim: 1 });
            network.add(candle_nn::linear(
                config.linear_layers[0],
                config.linear_layers[1],
                vs.pp(format!("conv_fc0")),
            )?);
            network.add(Activation::Relu);
            network.add(Dropout::new(0.5));
            network.add(candle_nn::linear(
                config.linear_layers[1],
                10,
                vs.pp(format!("conv_fc1")),
            )?);
        }

        Ok(network)
    }

    pub fn load_default(vm: &VarMap, sizes: &[usize], device: &Device) -> Result<()> {
        let mut sizes = sizes.to_vec();
        if sizes[0] != 28 * 28 {
            sizes.insert(0, 28 * 28);
        }
        let mut rng = XorShiftRng::seed_from_u64(1);
        for l in 1..sizes.len() {
            let w_size = sizes[l] * sizes[l - 1];
            let mut v: Vec<f32> = Vec::with_capacity(w_size);
            for _ in 0..w_size {
                v.push(rng.sample(rand_distr::StandardNormal));
            }
            let w = Tensor::from_vec(v, (sizes[l], sizes[l - 1]), device)?;
            let d = (sizes[l - 1] as f32).sqrt();
            let w = w.div(&Tensor::full(d, (sizes[l], sizes[l - 1]), device)?)?;
            let b = Tensor::full(0f32, (sizes[l],), device)?;

            if l == 1 {
                // println!("w: {:?}", w.p());
            }
            let z = vm.data();
            let mut z = z.lock().unwrap();
            let name_w = format!("fc{l}.weight");
            let name_b = format!("fc{l}.bias");
            z.insert(name_w, Var::from_tensor(&w)?);
            z.insert(name_b, Var::from_tensor(&b)?);
        }
        Ok(())
    }

    pub fn new(vs: VarBuilder, config: &Config, device: &Device) -> Result<Self> {
        // let mut layers_sizes = layers_size.to_vec();

        // First layer is input_size long.
        // layers_sizes.insert(0, input_size);

        let network = Self::create_network(vs, config)?;

        Ok(SequentialNetwork {
            network,
            device: device.clone(),
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input = input.to_device(&self.device)?;
        let z = self.network.forward(&input)?;
        Ok(z)
    }

    pub fn forward_t(&self, input: &Tensor) -> Result<Tensor> {
        let z = self.network.forward_t(input, true)?;
        Ok(z)
    }

    // Determine the ratio of correct answers.
    fn predict(&self, x: &Tensor, y: &Tensor) -> anyhow::Result<f32> {
        let a = self.forward(&x)?;
        let y_hat = a.argmax(1)?;
        let y_real = y.to_dtype(DType::U32)?;
        let same = y_hat.eq(&y_real)?.to_dtype(DType::F32)?;
        let v = same.mean_all()?.to_scalar::<f32>()?;
        Ok(v)
    }

    fn classify(&self, x: &Tensor) -> anyhow::Result<Vec<u8>> {
        let x = x.to_device(&self.device)?;
        let a = self.forward(&x)?;
        let y_hat = a.argmax(1)?;
        let y_hat_vec = y_hat.to_vec1::<u32>()?;
        Ok(y_hat_vec.iter().map(|x| *x as u8).collect())
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum TrainingOptimizer {
    SGD,
    AdamW,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub convolution_layers: Vec<(usize, usize, usize)>,
    pub linear_layers: Vec<usize>,

    pub optimizer: TrainingOptimizer,
    pub learning_rate: f64,
    pub iterations: usize,
    pub batch_size: usize,
}

pub fn fit(
    x: &Tensor,
    y: &Tensor,
    x_test: &Tensor,
    y_test: &Tensor,
    config: Config,
) -> anyhow::Result<SequentialNetwork> {
    use candle_nn::loss;
    let device = Device::cuda_if_available(0)?;
    let x = x.to_device(&device)?;
    let y = y.to_device(&device)?;
    let x_test = x_test.to_device(&device)?;
    let y_test = y_test.to_device(&device)?;

    // let mut layers = layers.to_vec();
    let varmap = VarMap::new();
    // layers.insert(0, 28*28);
    // SequentialNetwork::load_default(&varmap, &layers, &device)?;
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = SequentialNetwork::new(vs.clone(), &config, &device)?;

    println!("Creating batches");
    let mut mini_batches = util::create_mini_batches(&x, &y, config.batch_size, &device)?;
    for (_, train_label) in mini_batches.iter_mut() {
        *train_label = train_label.argmax(1)?;
    }
    let batch_len = mini_batches.len();

    println!("Starting iterations");
    use rand::prelude::*;
    use rand_xorshift::XorShiftRng;
    let mut rng = XorShiftRng::seed_from_u64(1);
    let mut shuffled_indices: Vec<usize> = (0..batch_len).collect();

    if config.optimizer == TrainingOptimizer::SGD {
        let mut sgd = candle_nn::SGD::new(varmap.all_vars(), config.learning_rate)?;
        for epoch in 1..config.iterations {
            shuffled_indices.shuffle(&mut rng);

            let mut sum_loss = 0f32;
            for idx in shuffled_indices.iter() {
                let (train_img, train_label) = &mini_batches[*idx];
                let logits = model.forward_t(&train_img)?;
                let log_sm = log_softmax(&logits, D::Minus1)?;
                let loss = loss::nll(&log_sm, &train_label)?;
                sgd.backward_step(&loss)?;
                sum_loss += loss.to_vec0::<f32>()?;
            }
            let avg_loss = sum_loss / batch_len as f32;
            let test_accuracy = model.predict(&x_test, &y_test)?;
            println!(
                "{epoch:4} train loss {:8.5} test acc: {:5.2}%",
                avg_loss,
                100. * test_accuracy
            );
        }
    } else {
        let adamw_params = candle_nn::ParamsAdamW {
            lr: config.learning_rate as f64,
            ..Default::default()
        };
        let mut opt = candle_nn::AdamW::new(varmap.all_vars(), adamw_params)?;

        for epoch in 1..config.iterations {
            shuffled_indices.shuffle(&mut rng);

            let mut sum_loss = 0f32;
            for idx in shuffled_indices.iter() {
                let (train_img, train_label) = &mini_batches[*idx];
                let logits = model.forward_t(&train_img)?;
                let log_sm = log_softmax(&logits, D::Minus1)?;
                let loss = loss::nll(&log_sm, &train_label)?;
                opt.backward_step(&loss)?;
                sum_loss += loss.to_vec0::<f32>()?;
            }
            let avg_loss = sum_loss / batch_len as f32;

            let test_accuracy = model.predict(&x_test, &y_test)?;
            println!(
                "{epoch:4} train loss {:8.5} test acc: {:5.2}%",
                avg_loss,
                100. * test_accuracy
            );
        }
    }
    Ok(model)
}

pub type MainResult = anyhow::Result<()>;
pub fn main() -> MainResult {
    let args = std::env::args().collect::<Vec<String>>();
    let m = candle_datasets::vision::mnist::load_dir(&args[1])?;

    println!("train-images: {:?}", m.train_images.shape());
    println!("train-labels: {:?}", m.train_labels.shape());
    println!("test-images: {:?}", m.test_images.shape());
    println!("test-labels: {:?}", m.test_labels.shape());

    println!("m.train_labels[0]: {:?}", m.train_labels.get(0));
    let train_0 = m.train_images.get(0)?;
    println!("train_0: {:?}", train_0);
    let img_0 = util::mnist_image(&train_0)?;
    img_0.save("/tmp/image_0.png")?;

    let device = Device::Cpu;
    // let device = Device::new_cuda(0)?;
    {
        let manual_config = Config {
            convolution_layers: vec![],
            linear_layers: vec![10, 10],
            optimizer: TrainingOptimizer::SGD,
            learning_rate: 0.01,
            iterations: 20,
            batch_size: 64,
        };
        let varmap = VarMap::new();
        SequentialNetwork::load_default(&varmap, &manual_config.linear_layers, &device)?;
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let ann = SequentialNetwork::new(vs, &manual_config, &device)?;
        // let mut t = vec![];
        let r = ann.forward(&m.train_images.i((0..64, ..))?)?;
        // let r = ann.step_forward(&m.train_images.i((0..64, ..))?)?;
        let r = softmax(&r, 1)?;
        println!("r: {:?}", r.t()?.get(0)?.p());
        // println!("r: {:?}", r.t()?.get(0)?.to_vec1::<f32>()?);

        // Check against numbers from manual.
        let values = r.t()?.get(0)?.to_vec1::<f32>()?;
        assert!((values[0] - 0.10819631).abs() < 0.0001);
        assert!((values[1] - 0.116772465).abs() < 0.0001);
        assert!((values[2] - 0.12155545).abs() < 0.0001);
        assert!((values[3] - 0.11611187).abs() < 0.0001);
        assert!((values[63] - 0.10917442).abs() < 0.0001);
    }

    let _linear_config = Config {
        convolution_layers: vec![],
        linear_layers: vec![1000, 10],
        optimizer: TrainingOptimizer::SGD,
        learning_rate: 0.01,
        iterations: 20,
        batch_size: 64,
    };

    let _convolution_config = Config {
        // let (in_channels, out_channels, kernel) = config.convolution_layers[l];
        convolution_layers: vec![(1, 32, 3)],
        linear_layers: vec![5408, 1000],
        optimizer: TrainingOptimizer::AdamW,
        learning_rate: 0.001,
        iterations: 20,
        batch_size: 64,
    };

    // let config_used = _linear_config;
    let config_used = _convolution_config;
    println!("Fitting now using {config_used:?}");

    let model = fit(
        &m.train_images,
        &m.train_labels,
        &m.test_images,
        &m.test_labels,
        config_used,
    )?;

    let test_range = 0..100;
    let digits = model.classify(&m.test_images.i((test_range.clone(), ..))?)?;
    for (i, digit) in test_range.zip(digits.iter()) {
        let this_image = m.test_images.get(i)?;
        let this_image = util::mnist_image(&this_image)?;
        this_image.save(format!("/tmp/image_{i}_d_{digit}.png"))?;
    }

    Ok(())
}
