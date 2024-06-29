// use candle_nn::sequential::seq;
// use candle_core::IndexOp;
use candle_core::IndexOp;
use candle_core::{DType, Device, Module, Result, Tensor};
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


pub struct LinearNetworkNetwork {
    network: Sequential,
    device: Device,
}

impl LinearNetworkNetwork {
    fn create_network(vs: VarBuilder, sizes: &[usize], device: &Device) -> Result<Sequential> {
        use rand::prelude::*;
        use rand_xorshift::XorShiftRng;
        let mut rng = XorShiftRng::seed_from_u64(1);

        // Can probabyl do some 'contains' tricks to ensure we seed with the exact same values.

        let mut network = seq();

        for l in 1..sizes.len() {
            if false {
                let w_size = sizes[l] * sizes[l - 1];
                let mut v: Vec<f32> = Vec::with_capacity(w_size);
                for _ in 0..w_size {
                    v.push(rng.sample(rand_distr::StandardNormal));
                }
                let w = Tensor::from_vec(v, (sizes[l], sizes[l - 1]), device)?;
                let d = (sizes[l - 1] as f32).sqrt();
                let w = w.div(&Tensor::full(d, (sizes[l], sizes[l - 1]), device)?)?;

                let b = Tensor::full(0f32, (1, sizes[l]), device)?;
                // println!("l{l} w: {:?}", w.p());
                // println!("l{l} b: {:?}", b.p());
                network = network.add(Linear::new(w, Some(b)));
            } else {
                let layer = candle_nn::linear(sizes[l-1], sizes[l], vs.pp(format!("fc{l}")))?;
                network = network.add(layer);
            }

            // Add sigmoid on all but the last layer.
            if l != (sizes.len() - 1) {
                network = network.add(Activation::Sigmoid);
            }
        }
        network = network.add(SoftmaxLayer::new(1));
        Ok(network)
    }

    pub fn new(vs: VarBuilder, layers_size: &[usize], input_size: usize, device: Device) -> Result<Self> {
        let mut layers_sizes = layers_size.to_vec();

        // let mut varmap = VarMap::new();
        // let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // First layer is input_size long.
        layers_sizes.insert(0, input_size);

        let network = Self::create_network(vs, &layers_sizes, &device)?;
        // println!("Network: {network:?}");

        Ok(LinearNetworkNetwork { network, device })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let z = self.network.forward(input)?;
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


}

pub fn fit(
    x: &Tensor,
    y: &Tensor,
    x_test: &Tensor,
    y_test: &Tensor,
    layers: &[usize],
    learning_rate: f32,
    iterations: usize,
    batch: usize,
) -> anyhow::Result<LinearNetworkNetwork> {
    use candle_core::D;
    use candle_nn::loss;
    use candle_nn::ops;
    let device = Device::cuda_if_available(0)?;
    let x = x.to_device(&device)?;
    let y = y.to_device(&device)?;
    let x_test = x_test.to_device(&device)?;
    let y_test = y_test.to_device(&device)?;


    let mini_batches = util::create_mini_batches(&x, &y, batch, &device)?;
    let batch_len = mini_batches.len();
    let mut batch_idxs = (0..batch_len).collect::<Vec<usize>>();

    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = LinearNetworkNetwork::new(vs.clone(), layers, 28*28, device)?;


    let adamw_params = candle_nn::ParamsAdamW {
        lr: learning_rate as f64,
        ..Default::default()
    };
    let mut opt = candle_nn::AdamW::new(varmap.all_vars(), adamw_params)?;


    for epoch in 1..iterations {
        

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

    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let mut ann = LinearNetworkNetwork::new(vs, &[10, 10], 28 * 28, device)?;

    // let mut t = vec![];
    let r = ann.forward(&m.train_images.i((0..64, ..))?)?;
    // let r = ann.step_forward(&m.train_images.i((0..64, ..))?)?;
    println!("r: {:?}", r.t()?.get(0)?.to_vec1::<f32>()?);
    
    let learning_rate = 0.1;
    let iterations = 3;
    let batch_size = 64;
    fit(&m.train_images,
        &m.train_labels,
        &m.test_images,
        &m.test_labels,
        &[10, 10],
        learning_rate,
        iterations,
        batch_size)?;

    Ok(())
}
