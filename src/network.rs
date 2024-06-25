// use candle_nn::sequential::seq;
// use candle_core::IndexOp;
use candle_nn::{VarMap, VarBuilder};
use candle_nn::{Sequential, seq, linear, Linear};
use candle_core::{DType, Device, Result, Tensor, Module}; 


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

