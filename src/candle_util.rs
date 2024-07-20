// use candle_core::IndexOp;
use candle_core::{DType, ModuleT, Tensor, Device};

pub mod prelude {
    pub use super::PrintableTensorTrait;
}

pub trait PrintableTensorTrait {
    fn p(&self) -> PrintableTensor;
}

impl PrintableTensorTrait for Tensor {
    fn p(&self) -> PrintableTensor {
        PrintableTensor { tensor: self }
    }
}

pub struct PrintableTensor<'a> {
    tensor: &'a Tensor,
}

impl<'a> std::fmt::Debug for PrintableTensor<'a> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        // Assume tensors are 2D?
        write!(
            fmt,
            "Tensor: {:?} ({:?}): ",
            self.tensor.shape(),
            self.tensor.dtype()
        )?;
        let dim_count = self.tensor.dims().len();
        if dim_count == 0 {
            write!(fmt, "∅")?;
        } else if dim_count == 1 {
                if self.tensor.dtype() == DType::F32 {
                    write!(
                        fmt,
                        "{:?}",
                        self.tensor.to_vec1::<f32>().map_err(|_| std::fmt::Error)?
                    )?;
                }
                if self.tensor.dtype() == DType::U32 {
                    write!(
                        fmt,
                        "{:?}",
                        self.tensor.to_vec1::<u32>().map_err(|_| std::fmt::Error)?
                    )?;
                }
            
        } else if dim_count == 2 {
            let mut v = fmt.debug_list();
            // let mut b = &mut v;
            for y in 0..self.tensor.dim(0).map_err(|_| std::fmt::Error)? {
                let r = self.tensor.get(y).map_err(|_| std::fmt::Error)?;
                // write!(fmt, "{:?}", r.to_vec1::<f32>().map_err(|_| std::fmt::Error)?)?;
                // v.entry(&format!("{:?}", r.to_vec1::<f32>().map_err(|_| std::fmt::Error)?));
                if self.tensor.dtype() == DType::F32 {
                    v.entry(&format_args!(
                        "{:?}",
                        r.to_vec1::<f32>().map_err(|_| std::fmt::Error)?
                    ));
                }
                if self.tensor.dtype() == DType::U32 {
                    v.entry(&format_args!(
                        "{:?}",
                        r.to_vec1::<u32>().map_err(|_| std::fmt::Error)?
                    ));
                }
            }
            v.finish()?;
        } else {
            let mut v = fmt.debug_struct(&format!("d{dim_count}"));
            for z in 0..self.tensor.dim(0).map_err(|_| std::fmt::Error)? {
                let t = self.tensor.get(z).map_err(|_| std::fmt::Error)?;
                let p = t.p();
                v.field(&format!("{z}"), &format_args!("{:#?}", p));
            }
            v.finish()?;
        }
        Ok(())
    }
}

/// A sequential struct that holds ModuleT instead of Module such that it can be used for training.
pub struct SequentialT {
    layers: Vec<Box<dyn ModuleT>>,
}

impl SequentialT {
    pub fn new() -> Self {
        Self { layers: vec![] }
    }

    pub fn add<T: ModuleT + 'static>(&mut self, v: T) {
        self.layers.push(Box::new(v))
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn forward_t(&self, xs: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = (**layer).forward_t(&xs, train)?
        }
        Ok(xs)
    }
    pub fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = (**layer).forward_t(&xs, false)?
        }
        Ok(xs)
    }
}

pub struct MaxPoolLayer {
    dim: usize,
}
impl MaxPoolLayer {
    pub fn new(sz: usize) -> candle_core::Result<MaxPoolLayer> {
        Ok(Self { dim: sz })
    }
}
impl ModuleT for MaxPoolLayer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        let _ = train;
        xs.max_pool2d(self.dim)
    }
}

pub fn load_from_safetensors<P>(
    path: P,
    device: &candle_core::Device,
) -> candle_core::Result<std::collections::HashMap<String, Tensor>>
where
    P: AsRef<std::path::Path> + Copy,
{
    // use candle_core::safetensors::Load;
    use candle_core::safetensors::{Load, MmapedSafetensors};
    let tensors = unsafe { MmapedSafetensors::new(path)? };
    let tensors: std::collections::HashMap<_, _> = tensors.tensors().into_iter().collect();
    let mut res = std::collections::HashMap::<String, Tensor>::new();
    for (name, _tensor) in tensors.iter() {
        match tensors.get(name) {
            Some(tensor_view) => {
                let tensor = tensor_view.load(device)?;
                res.insert(name.clone(), tensor);
            }
            None => {}
        }
    }
    Ok(res)
}



#[macro_export]
macro_rules! approx_equal {
    ($a:expr, $b: expr, $max_error:expr) => {
        let delta = ($a - $b).abs();
        if delta > $max_error {
            panic!(
                "a: {a:?}, b: {b:?},  delta was {delta}, this exceeded allowed {max_error}.",
                a = $a,
                b = $b,
                max_error = $max_error
            );
        }
    };
}
macro_rules! approx_equal_slice {
    ($a:expr, $b: expr, $max_error:expr) => {
        for (i, (a_v, b_v)) in $a.iter().zip($b.iter()).enumerate() {
            let delta = (*a_v - *b_v).abs();
            if delta > $max_error {
                panic!(
                    "a: {a_v:?}, b: {b_v:?},  delta was {delta}, this exceeded allowed {max_error} at {i}",
                    max_error = $max_error
                );
            }
            
        }
    }
}

#[macro_export]
macro_rules! error_unwrap {
    ($a:expr) => {
        if let Err(e) = $a {
            eprintln!("{}", e);
            // handle the error properly here
            assert!(false);
            unreachable!();
        } else {
            $a.ok().unwrap()
        }
    };
}

// https://stackoverflow.com/a/31749071  export the macro local to this file into the module.
// pub(crate) use approx_equal;
// pub(crate) use error_unwrap;


pub fn binary_cross_entropy(input: &Tensor, target: &Tensor) -> candle_core::Result<Tensor> {
    if input.dtype() != DType::F32 {
        candle_core::bail!("input has wrong type, got: {:?}", input.dtype());
    }
    if target.dtype() != DType::F32 {
        candle_core::bail!("target has wrong type, got: {:?}", target.dtype());
    }

    let device = target.device();
    // -(target * (torch.clamp(input, 0) + eps).log() + (1. - target) * (1. - torch.clamp(input, 0) + eps).log())
    // left: target * (torch.clamp(input, 0) + eps).log() 
    // right (1. - target) * (1. - torch.clamp(input, 0) + eps).log()
    let eps = Tensor::full(1e-10f32, (), &device)?;
    let one = Tensor::full(1.0f32, (), &device)?;
    let input_clamp = input.maximum(0.0)?;
    let left = (target * (input_clamp.broadcast_add(&eps)?).log()?)?;
    let right = (one.broadcast_sub(&target) * ((one.broadcast_sub(&input_clamp)?.broadcast_add(&eps)?).log()?))?;
    let calcfinal = ((left + right)?).neg()?;
    Ok(calcfinal)
}


#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_candle_print() -> anyhow::Result<()> {
        use candle_core::{Device::Cpu, Tensor};
        use candle_nn::{Linear, Module};

        let w = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &Cpu)?;
        let layer = Linear::new(w.clone(), None); // Use no bias.
        let xs = Tensor::new(&[[10f32, 100.]], &Cpu)?;
        let ys = layer.forward(&xs)?;
        assert_eq!(ys.to_vec2::<f32>()?, &[[210.0, 430.0, 650.0]]);

        println!("{:?}", xs.p());
        println!("{:#?}", xs.p());
        println!("{:?}", w.p());
        println!("{:#?}", w.p());
        println!("{:#?}", w.t()?.p());
        let row = Tensor::new(&[10f32, 100.], &Cpu)?;
        println!("{:?}", row.p());
        println!("{:#?}", row.p());

        Ok(())
    }


    #[test]
    fn test_crossentropy() -> anyhow::Result<()> {
        let device = Device::Cpu;

        /*

            import torch
            from torch import tensor
            from torch import nn

            m = nn.Sigmoid()
            loss = nn.BCELoss()
            input = tensor([[[[ 2.8929, -1.0923],
                              [-0.4709, -0.1996]]]], requires_grad=True)
            target = tensor([[[[ 1.0, 0.5],
                              [1.0, 0.2]]]], requires_grad=True)
            output = loss(m(input), target)
            print(f"BCELoss           : {output}")
            print(f"BCELoss no reduce:", nn.BCELoss(reduction='none')(m(input), target))

            criterion = torch.nn.BCEWithLogitsLoss()
            z = criterion(input, target)
            print(f"BCELoss with logit: {z}")

            # s(x) = 1.0 / (1 + e^-x) = e^-x / (1 + e^x)
            loss = (torch.clamp(input, 0) - input * target  + torch.log(1 + torch.exp(-torch.abs(input))))
            print(f"loss: {loss}")
            loss = loss.mean()
            print(f"loss mean:          {loss}")

            # Loss without sigmoid built in;
            eps = 1e-6
            input = m(input)
            z=  -(target * (torch.clamp(input, 0) + eps).log() + (1. - target) * (1. - torch.clamp(input, 0) + eps).log())
            print(f"loss: {z}")


            BCELoss           : 0.6209125518798828

            BCELoss no reduce: tensor([[[[0.0539, 0.8354],
                                         [0.9561, 0.6382]]]], grad_fn=<BinaryCrossEntropyBackward0>)
            BCELoss with logit: 0.6209125518798828

            loss: tensor([[[[0.0539, 0.8354],
                            [0.9561, 0.6382]]]], grad_fn=<AddBackward0>)
            loss mean:          0.6209125518798828

            loss: tensor([[[[0.0539, 0.8354],
                            [0.9561, 0.6382]]]], grad_fn=<NegBackward0>)



            > https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
            > This loss combines a Sigmoid layer and the BCELoss in one single class.
            > This version is more numerically stable than using a plain Sigmoid followed by a
            > BCELoss as, by combining the operations into one layer, we take advantage of the
            > log-sum-exp trick for numerical stability.

        */

        let expected = Tensor::from_slice(&[0.0539, 0.854,
                                            0.9561, 0.6282f32], (1, 1, 2, 2), &device)?;


        let target = Tensor::from_slice(&[1.0, 0.5,
                                          1.0, 0.2f32], (1, 1, 2, 2), &device)?;

        #[rustfmt::skip]
        let input = Tensor::from_slice(&[2.8929, -1.0923,
                                        -0.4709, -0.1996f32], (1, 1, 2, 2), &device)?;
        let input_s = candle_nn::ops::sigmoid(&input)?;
        let loss = error_unwrap!(binary_cross_entropy(&input_s, &target));
        

        let expected_v = expected.flatten_all()?.to_vec1::<f32>()?;
        let loss_v = loss.flatten_all()?.to_vec1::<f32>()?;
        println!("expected_v: {expected_v:?}");
        println!("loss_v: {loss_v:?}");
        approx_equal_slice!(&expected_v, &loss_v, 0.02);

        // let loss_a = loss.to_scalar::<f32>()?;
        // println!("loss_a: {loss_a:?}");

        Ok(())
    }
}
