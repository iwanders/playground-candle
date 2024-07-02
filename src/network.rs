// use candle_nn::sequential::seq;
// use candle_core::IndexOp;
use candle_core::IndexOp;
use candle_core::{DType, Device, Module, Result, Tensor, Shape, Var, ModuleT};
use candle_nn::{linear, seq, Linear, Sequential, Activation};
use candle_nn::{VarBuilder, VarMap};
use candle_nn::ops::softmax;
use candle_nn::Optimizer;

use crate::candle_util::prelude::*;
use crate::util;
use rand::prelude::*;

pub struct SoftmaxLayer {
  pub dim: usize
}
impl SoftmaxLayer {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
        }
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

pub struct MaxPoolLayer {
  pub dim: usize
}
impl Module for MaxPoolLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.max_pool2d(self.dim)
    }
}



pub struct FlattenLayer {
  pub dim: usize
}
impl Module for FlattenLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.flatten_from(self.dim)
    }
}

pub struct SequentialNetwork {
    network: Sequential,
    device: Device,
}

pub struct DropoutLayer {
    dropout: candle_nn::Dropout,
    linear: Linear,
}
impl Module for DropoutLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.dropout.forward(&xs, false)?.apply(&self.linear)
    }
}



impl SequentialNetwork {
    fn create_network(vs: VarBuilder, config: &Config, device: &Device) -> Result<Sequential> {
        let mut network = seq();

        let mut sizes = config.linear_layers.clone();

        if (config.convolution_layers.is_empty()) {
            sizes.insert(0, 28 * 28);
        } else {
            sizes.insert(0, 1024);
            network = network.add(ToImageLayer{});
            for l in 0..config.convolution_layers.len() {
                let (in_channels, out_channels, kernel) = config.convolution_layers[l];
                network = network.add(candle_nn::conv2d(in_channels, out_channels, kernel, Default::default(), vs.pp(format!("c{l}")))?);
                network = network.add(MaxPoolLayer{dim: 2});
            }
            // network = network.add(candle_nn::conv2d(1, 32, 5, Default::default(), vs.pp("c1"))?);
            // network = network.add(MaxPoolLayer{dim: 2});
            // network = network.add(candle_nn::conv2d(32, 64, 5, Default::default(), vs.pp("c2"))?);
            // network = network.add(MaxPoolLayer{dim: 2});
            network = network.add(FlattenLayer{dim: 1});
        }

        for l in 1..sizes.len() {
            let layer = candle_nn::linear(sizes[l-1], sizes[l], vs.pp(format!("fc{l}")))?;
            if (l == 1) {
                // println!("w: {:?}", layer.weight().p());
            }
            network = network.add(layer);
            // Add sigmoid on all but the last layer.
            if l != (sizes.len() - 1) {
                network = network.add(Activation::Sigmoid);
            }
        }
        network = network.add(SoftmaxLayer::new(1));
        Ok(network)
    }

    pub fn load_default(vm: &VarMap, sizes: &[usize], device: &Device) -> Result<()> {
        use rand::prelude::*;
        use rand_xorshift::XorShiftRng;
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

            if (l == 1) {
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

    pub fn new(vs: VarBuilder, config: &Config, input_size: usize, device: &Device) -> Result<Self> {
        // let mut layers_sizes = layers_size.to_vec();

        // First layer is input_size long.
        // layers_sizes.insert(0, input_size);

        let network = Self::create_network(vs, config, device)?;

        Ok(SequentialNetwork { network, device: device.clone() })
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

    pub fn step_forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut xs = input.clone();
        for i in 0..self.network.len() {
            xs = self.network.layer(i).unwrap().forward(&xs)?;
            println!("xs{i}: {:?}", xs.p());
        }
        Ok(xs)
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
    use candle_core::D;
    use candle_nn::loss;
    use candle_nn::ops;
    let device = Device::cuda_if_available(0)?;
    let x = x.to_device(&device)?;
    let y = y.to_device(&device)?;
    let x_test = x_test.to_device(&device)?;
    let y_test = y_test.to_device(&device)?;



    let convert_outputs = |v: &Tensor| -> Result<Tensor> {
        let one = Tensor::full(1.0 as f32, (1, 1), &device)?;
        let len = v.dims1()?;
        let mut output = Tensor::full(0.0 as f32, (len, 10), &device)?;
        for i in 0..len {
            let d = y.get(i)?.to_scalar::<u8>()? as usize;
            output = output.slice_assign(&[i..=i, d..=d], &one)?;
        }
        Ok(output)
    };


    // let mut layers = layers.to_vec();
    let mut varmap = VarMap::new();
    // layers.insert(0, 28*28);
    // SequentialNetwork::load_default(&varmap, &layers, &device)?;
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = SequentialNetwork::new(vs.clone(), &config, 28*28, &device)?;

    let mut mini_batches = util::create_mini_batches(&x, &y, config.batch_size, &device)?;
    for (_, train_label) in mini_batches.iter_mut() {
        *train_label = train_label.argmax(1)?;
    }
    let batch_len = mini_batches.len();

    if config.optimizer == TrainingOptimizer::SGD {
        let mut sgd = candle_nn::SGD::new(varmap.all_vars(), config.learning_rate)?;
        for epoch in 1..config.iterations {

            let mut sum_loss = 0f32;
            for (train_img, train_label) in mini_batches.iter() {
                let logits = model.forward_t(&train_img)?;
                let log_sm = logits.log()?;  // softmax is done by the network, so only need log here.
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

        let y = y.to_dtype(DType::F32)?;
        let y_test = y_test.to_dtype(DType::F32)?;


        let adamw_params = candle_nn::ParamsAdamW {
            lr: config.learning_rate as f64,
            ..Default::default()
        };
        let mut opt = candle_nn::AdamW::new(varmap.all_vars(), adamw_params)?;


        for epoch in 1..config.iterations {
            let mut sum_loss = 0f32;
            for (train_img, train_label) in mini_batches.iter() {
                let logits = model.forward_t(&train_img)?;
                let log_sm = logits.log()?;  // softmax is done by the network, so only need log here.
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

    // let device = Device::Cpu;
    let device = Device::new_cuda(0)?;

    // let mut varmap = VarMap::new();
    // let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    // let mut ann = SequentialNetwork::new(vs, &[28*28, 10, 10], 28 * 28, &device)?;
    // let mut t = vec![];
    // let r = ann.forward(&m.train_images.i((0..64, ..))?)?;
    // let r = ann.step_forward(&m.train_images.i((0..64, ..))?)?;
    // println!("r: {:?}", r.t()?.get(0)?.to_vec1::<f32>()?);
    
    let linear_config = Config{
        convolution_layers: vec![],
        linear_layers: vec![10, 10],
        optimizer: TrainingOptimizer::SGD,
        learning_rate: 0.01,
        iterations: 20,
        batch_size: 64,
    };

    let convolution_config = Config {
        convolution_layers: vec![(1, 32, 5), (32, 64, 5)],
        linear_layers: vec![1024, 10],
        optimizer: TrainingOptimizer::AdamW,
        learning_rate: 0.01,
        iterations: 20,
        batch_size: 64,

    };


    let config_used = convolution_config;

    let model = fit(&m.train_images,
        &m.train_labels,
        &m.test_images,
        &m.test_labels,
        config_used
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
