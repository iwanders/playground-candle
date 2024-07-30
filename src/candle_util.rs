// use candle_core::IndexOp;
use candle_core::{DType, ModuleT, Tensor};

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

pub struct Interpolate2DLayer {
    target_h: usize,
    target_w: usize,
}
impl Interpolate2DLayer {
    pub fn new(target_h: usize, target_w: usize) -> candle_core::Result<Interpolate2DLayer> {
        Ok(Self { target_h, target_w })
    }
}
impl ModuleT for Interpolate2DLayer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        let _ = train;
        xs.interpolate2d(self.target_h, self.target_w)
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

pub trait LoadInto {
    fn load_into<P: AsRef<std::path::Path>>(
        &self,
        path: P,
        detached: bool,
    ) -> candle_core::Result<()>;
}
impl LoadInto for candle_nn::VarMap {
    fn load_into<P: AsRef<std::path::Path>>(
        &self,
        path: P,
        detached: bool,
    ) -> candle_core::Result<()> {
        let path = path.as_ref();
        let file_data = unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };
        let mut tensor_data = self.data().lock().unwrap();
        let keys = file_data
            .tensors()
            .iter()
            .map(|(n, _)| n)
            .cloned()
            .collect::<Vec<_>>();
        println!("keys in safetensors: {keys:?}");
        for (name, var) in tensor_data.iter_mut() {
            println!("Varmap has {name}, contained: {}", keys.contains(name));
            if keys.contains(name) {
                let data = file_data.load(name, var.device())?;
                let data = if detached { data.detach() } else { data };
                if let Err(err) = var.set(&data) {
                    candle_core::bail!("error setting {name} using data from {path:?}: {err}",)
                }
            }
        }
        Ok(())
    }
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

#[macro_export]
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

pub use approx_equal;
pub use approx_equal_slice;
pub use error_unwrap;

#[derive(Debug, Default)]
pub struct CuMem {
    pub available: usize,
    pub total: usize,
}
impl std::fmt::Display for CuMem {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            fmt,
            "{}MiB/{}MiB",
            (self.total - self.available) / (1024 * 1024),
            self.total / (1024 * 1024)
        )
    }
}

#[cfg(feature = "cuda")]
pub fn get_vram() -> candle_core::Result<CuMem> {
    // return Ok(Default::default());
    use candle_core::cuda_backend::cudarc;
    return cudarc::driver::result::mem_get_info()
        .map_err(|e| candle_core::Error::Cuda(Box::new(e)))
        .map(|(available, total)| CuMem { available, total });
}
#[cfg(not(feature = "cuda"))]
pub fn get_vram() -> candle_core::Result<CuMem> {
    Ok(Default::default())
}

#[derive(Debug, Eq, PartialEq, Copy, Clone, clap::ValueEnum)]
pub enum Reduction {
    Mean,
    Sum,
}

/// Binary cross entropy, no reduction, expects data after sigmoid.
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
    let right = (one.broadcast_sub(&target)
        * ((one.broadcast_sub(&input_clamp)?.broadcast_add(&eps)?).log()?))?;
    let calcfinal = ((left + right)?).neg()?;
    Ok(calcfinal)
}

/// Binary cross entropy, with mean reduction, expects data after sigmoid.
pub fn binary_cross_entropy_loss(
    input: &Tensor,
    target: &Tensor,
    reduction: Reduction,
) -> candle_core::Result<Tensor> {
    let r = binary_cross_entropy(input, target)?;
    match reduction {
        Reduction::Sum => r.sum_all(),
        Reduction::Mean => r.mean_all(),
    }
}

/// Binary cross entropy, no reduction, performs sigmoid in computation.
pub fn binary_cross_entropy_logits(input: &Tensor, target: &Tensor) -> candle_core::Result<Tensor> {
    if input.dtype() != DType::F32 {
        candle_core::bail!("input has wrong type, got: {:?}", input.dtype());
    }
    if target.dtype() != DType::F32 {
        candle_core::bail!("target has wrong type, got: {:?}", target.dtype());
    }
    let device = target.device();

    // (torch.clamp(input, 0) - input * target  + torch.log(1 + torch.exp(-torch.abs(input))))
    // left: torch.clamp(input, 0) - input * target
    // right: torch.log(1 + torch.exp(-torch.abs(input)))
    let one = Tensor::full(1.0f32, (), &device)?;
    let input_clamp = input.maximum(0.0)?;
    let left = (input_clamp - (input * target)?)?;
    let right = (one.broadcast_add(&input.abs()?.neg()?.exp()?)?).log()?;
    left + right
}

/// Binary cross entropy, with mean reduction, performs sigmoid in computation.
pub fn binary_cross_entropy_logits_loss(
    input: &Tensor,
    target: &Tensor,
    reduction: Reduction,
) -> candle_core::Result<Tensor> {
    let r = binary_cross_entropy_logits(input, target)?;
    match reduction {
        Reduction::Sum => r.sum_all(),
        Reduction::Mean => r.mean_all(),
    }
}

pub fn c_u32_one_hot(input: &Tensor, max_count: usize) -> candle_core::Result<Tensor> {
    if input.dtype() != DType::U32 {
        candle_core::bail!("input has wrong type, got: {:?}", input.dtype());
    }
    // Expected input is: (1, w, h) or (N, 1, w, h)
    // Onehot will be: (max_count, w, h) or (N, max_count, w, h)
    // println!("input: {:?}", input.p());
    let input_rank = input.rank();
    // println!("input_rank: {}", input_rank);

    let device = input.device();

    let input = input.force_contiguous()?;

    let one = Tensor::full(1.0f32, input.shape(), &device)?.force_contiguous()?;
    let zero = Tensor::full(0.0f32, input.shape(), &device)?.force_contiguous()?;
    if input_rank == 3 {
        let zero_repeated = zero.repeat((max_count, 1, 1))?.force_contiguous()?;
        let r = zero_repeated.scatter_add(&input, &one, 0)?;
        // println!("r: {:?}", r.p());
        Ok(r)
    } else {
        candle_core::bail!("one hot shape not supported: {:?}", input.shape());
    }
}

pub fn deconvolution_upsample(
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
) -> candle_core::Result<Tensor> {
    use candle_core::{Device, IndexOp};
    let factor = ((kernel + 1) / 2) as f32;

    let center = if kernel.rem_euclid(2) == 1 {
        factor - 1.0
    } else {
        factor - 0.5
    };

    let mut k = Tensor::zeros(
        (in_channels, out_channels, kernel, kernel),
        DType::F32,
        &Device::Cpu,
    )?;
    for ci in 0..in_channels {
        for co in 0..out_channels {
            if ci != co {
                continue;
            }
            for y in 0..kernel {
                let ry = 1.0f32 - (y as f32 - center).abs() / factor;
                for x in 0..kernel {
                    let rx = 1.0f32 - (x as f32 - center).abs() / factor;
                    let v = ry * rx;
                    k = k.slice_assign(
                        &[ci..=ci, co..=co, x..=x, y..=y],
                        &Tensor::from_slice(&[v], (1, 1, 1, 1), &Device::Cpu)?,
                    )?;
                }
            }
        }
    }

    Ok(k)
}

#[cfg(test)]
mod test {
    use super::*;
    use candle_core::{Device, Device::Cpu, Tensor};
    #[test]
    fn test_candle_print() -> anyhow::Result<()> {
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

        #[rustfmt::skip]
        let expected = Tensor::from_slice(&[0.0539, 0.854,
                                            0.9561, 0.6282f32], (1, 1, 2, 2), &device)?;
        let expected_v = expected.flatten_all()?.to_vec1::<f32>()?;

        let loss_expected = 0.6209125518798828f32;

        #[rustfmt::skip]
        let target = Tensor::from_slice(&[1.0, 0.5,
                                          1.0, 0.2f32], (1, 1, 2, 2), &device)?;

        #[rustfmt::skip]
        let input = Tensor::from_slice(&[2.8929, -1.0923,
                                        -0.4709, -0.1996f32], (1, 1, 2, 2), &device)?;
        let input_s = candle_nn::ops::sigmoid(&input)?;

        // Finally, we go into the unit tests, first for normal bce;
        let loss = error_unwrap!(binary_cross_entropy(&input_s, &target));
        let loss_v = loss.flatten_all()?.to_vec1::<f32>()?;
        println!("expected_v: {expected_v:?}");
        println!("loss_v: {loss_v:?}");
        approx_equal_slice!(&expected_v, &loss_v, 0.02);

        // Scalar version, which is mean_all;
        let loss_scalar = binary_cross_entropy_loss(&input_s, &target, Reduction::Mean)?;
        let loss_a = loss_scalar.to_scalar::<f32>()?;
        println!("loss_a: {loss_a:?}");
        approx_equal!(loss_expected, loss_a, 0.0001);

        // The one that embeds the sigmoid function for stability;
        let loss = error_unwrap!(binary_cross_entropy_logits(&input, &target));
        let loss_v = loss.flatten_all()?.to_vec1::<f32>()?;
        println!("expected_v: {expected_v:?}");
        println!("loss_v: {loss_v:?}");
        approx_equal_slice!(&expected_v, &loss_v, 0.02);

        // And finally, the embedded sigmoid, directly to the loss scalar.
        let loss_scalar = binary_cross_entropy_logits_loss(&input, &target, Reduction::Mean)?;
        let loss_a = loss_scalar.to_scalar::<f32>()?;
        println!("loss_a: {loss_a:?}");
        approx_equal!(loss_expected, loss_a, 0.0001);

        Ok(())
    }

    #[test]
    fn test_c_u32_one_hot() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let w = Tensor::from_slice(&[1, 1, 0, 2u32], (1, 2, 2), &device)?;

        let one_hot = error_unwrap!(c_u32_one_hot(&w, 3));
        let one_hot_v = one_hot.flatten_all()?.to_vec1::<f32>()?;
        println!("one_hot: {:?}", one_hot.p());

        let z = Tensor::from_slice(
            &[
                0.0, 0.0, 1.0, 0.0f32, 1.0, 1.0, 0.0, 0.0f32, 0.0, 0.0, 0.0, 1.0f32,
            ],
            (3, 2, 2),
            &device,
        )?;
        let z_v = z.flatten_all()?.to_vec1::<f32>()?;

        assert_eq!(one_hot.shape(), z.shape());
        approx_equal_slice!(&z_v, &one_hot_v, 0.02);

        Ok(())
    }

    #[test]
    fn test_deconvolution_upscale_kernel() -> candle_core::Result<()> {
        /*
        tensor([[[[0.2500, 0.5000, 0.2500],
                  [0.5000, 1.0000, 0.5000],
                  [0.2500, 0.5000, 0.2500]]]])
        */
        let device = Device::Cpu;
        let upscale_1_1_3 = deconvolution_upsample(1, 1, 3)?;
        let upscale_1_1_3_v = upscale_1_1_3.flatten_all()?.to_vec1::<f32>()?;
        let e = Tensor::from_slice(
            &[0.25f32, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25],
            (1, 1, 3, 3),
            &device,
        )?;
        let e_v = e.flatten_all()?.to_vec1::<f32>()?;
        approx_equal_slice!(&upscale_1_1_3_v, &e_v, 0.02);

        let e = Tensor::from_slice(
            &[
                0.0625f32, 0.1875, 0.1875, 0.0625, 0.1875, 0.5625, 0.5625, 0.1875, 0.1875, 0.5625,
                0.5625, 0.1875, 0.0625, 0.1875, 0.1875, 0.0625,
            ],
            (1, 1, 4, 4),
            &device,
        )?;
        let e_v = e.flatten_all()?.to_vec1::<f32>()?;
        let upscale_1_1_4 = deconvolution_upsample(1, 1, 4)?;
        let upscale_1_1_4_v = upscale_1_1_4.flatten_all()?.to_vec1::<f32>()?;
        approx_equal_slice!(&upscale_1_1_4_v, &e_v, 0.02);

        let e = Tensor::from_slice(
            &[
                0.2500f32, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.2500, 0.2500, 0.2500, 0.2500,
            ],
            (2, 2, 2, 2),
            &device,
        )?;
        let e_v = e.flatten_all()?.to_vec1::<f32>()?;
        let upscale_kernel = deconvolution_upsample(2, 2, 2)?;
        let upscale_kernel = upscale_kernel.flatten_all()?.to_vec1::<f32>()?;
        approx_equal_slice!(&upscale_kernel, &e_v, 0.02);

        Ok(())
    }
}
