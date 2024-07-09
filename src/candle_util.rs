// use candle_core::IndexOp;
use candle_core::{DType, ModuleT, Tensor, Module};

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
        }
        if dim_count == 1 {
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
        }
        if dim_count == 2 {
            let mut v = fmt.debug_list();
            // let mut b = &mut v;
            for y in 0..self.tensor.dim(0).map_err(|_| std::fmt::Error)? {
                let r = self.tensor.get(y).map_err(|_| std::fmt::Error)?;
                // write!(fmt, "{:?}", r.to_vec1::<f32>().map_err(|_| std::fmt::Error)?)?;
                // v.entry(&format!("{:?}", r.to_vec1::<f32>().map_err(|_| std::fmt::Error)?));
                v.entry(&format_args!(
                    "{:?}",
                    r.to_vec1::<f32>().map_err(|_| std::fmt::Error)?
                ));
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
        Ok(Self {
            dim: sz
        })
    }
}
impl ModuleT for MaxPoolLayer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> candle_core::Result<Tensor>
    {
        xs.max_pool2d(self.dim)
    }
}

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

// https://stackoverflow.com/a/31749071  export the macro local to this file into the module.
pub(crate) use approx_equal;


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
}
