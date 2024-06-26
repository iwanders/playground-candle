// use candle_nn::sequential::seq;
// use candle_core::IndexOp;
use candle_nn::{VarMap, VarBuilder};
use candle_nn::{Sequential, seq, linear, Linear};
use candle_core::{DType, Device, Result, Tensor, Module}; 
use candle_core::IndexOp;

use crate::util;


pub struct LinearNetworkNetwork {
    network: Sequential,
    device: Device,
}

impl LinearNetworkNetwork {
    fn create_network(sizes: &[usize], device: &Device) -> Result<Sequential> {

        use rand::prelude::*;
        use rand_xorshift::XorShiftRng;
        let mut rng = XorShiftRng::seed_from_u64(1);


        // let varmap = VarMap::new();
        // let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        // let model = linear(2, 1, vb.pp("linear"))?;
        // , vb.pp(format!("layer{}", l - 1))

        let mut network = seq();

        for l in 1..sizes.len() {

            let w_size = sizes[l] * sizes[l - 1];
            let mut v: Vec<f32> = Vec::with_capacity(w_size);
            for _ in 0..w_size {
                v.push(rng.sample(rand_distr::StandardNormal));
            }
            let w = Tensor::from_vec(v, (sizes[l], sizes[l - 1]), device)?;
            let d = (sizes[l - 1] as f32).sqrt();
            let w = w.div(&Tensor::full(d, (sizes[l], sizes[l - 1]), device)?)?;

            let b = Tensor::full(0f32, (sizes[l], 1), device)?;

            network = network.add(Linear::new(w, Some(b)));
        }
        Ok(network)
    }

    pub fn new(layers_size: &[usize], input_size: usize, device: Device) -> Result<Self> {
        let mut layers_sizes = layers_size.to_vec();

        // First layer is input_size long.
        layers_sizes.insert(0, input_size);

        let network = Self::create_network(&layers_sizes, &device)?;
        // println!("Network: {network:?}");

        Ok(LinearNetworkNetwork {
            network,
            device,
        })
    }


    pub fn forward(
        &self,
        input: &Tensor,
    ) -> Result<Tensor> {
        self.network.forward(input)
    }
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

    let mut ann = LinearNetworkNetwork::new(&[10, 10], 28 * 28, device)?;

    // let mut t = vec![];
    let r = ann.forward(&m.train_images.i((0..64, ..))?)?;
    println!("r: {:?}", r.get(0)?.to_vec1::<f32>()?);
    // println!("t: {:?}", t);
    Ok(())
}

