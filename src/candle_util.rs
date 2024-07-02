// use candle_core::IndexOp;
use candle_core::{Tensor, DType};

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
        write!(fmt, "Tensor: {:?} ({:?}): ", self.tensor.shape(), self.tensor.dtype())?;
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
