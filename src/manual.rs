use crate::util;
use candle_core::IndexOp;
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::softmax;

// Solving mnist fully connected linear layers.

// Based on
// https://medium.com/@koushikkushal95/mnist-hand-written-digit-classification-using-neural-network-from-scratch-54da85712a06
// https://github.com/Koushikl0l/Machine_learning_from_scratch/blob/main/_nn_from_scratch__mini_batch_mnist.ipynb

// Pretty much a fully connected linear network?

// Others;
// https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
// Interesting; https://www.kaggle.com/code/prashant111/mnist-deep-neural-network-with-keras

#[derive(Clone, Debug)]
struct LinearLayer {
    // Z = W.T  X + b
    w: Tensor,
    b: Tensor,
}

#[derive(Clone, Debug)]
struct TrainLinear {
    layer: LinearLayer,

    a: Tensor,
    z: Tensor,

    dw: Tensor,
    db: Tensor,
}

struct LinearNetworkManual {
    layers_sizes: Vec<usize>,
    layers: Vec<LinearLayer>,
    device: Device,
}

impl LinearNetworkManual {
    fn sigmoid(&self, z: &Tensor) -> anyhow::Result<Tensor> {
        let one = Tensor::full(1.0f32, z.shape(), &self.device)?;
        Ok((&one / (&one + z.neg()?.exp()?)?)?)
    }
    fn sigmoid_derivative(&self, z: &Tensor) -> anyhow::Result<Tensor> {
        let s = self.sigmoid(z)?;
        let one = Tensor::full(1.0f32, z.shape(), &self.device)?;
        Ok((&s * (one - &s)?)?)
    }

    pub fn create_layers(sizes: &[usize], device: &Device) -> anyhow::Result<Vec<LinearLayer>> {
        use rand::prelude::*;
        use rand_xorshift::XorShiftRng;
        let mut rng = XorShiftRng::seed_from_u64(1);

        let mut res = vec![];

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

            res.push(LinearLayer { w, b });
        }

        Ok(res)
    }

    pub fn new(layers_size: &[usize], input_size: usize, device: Device) -> anyhow::Result<Self> {
        let mut layers_sizes = layers_size.to_vec();

        // First layer is input_size long.
        layers_sizes.insert(0, input_size);

        let layers = Self::create_layers(&layers_sizes, &device)?;

        Ok(LinearNetworkManual {
            layers_sizes,
            layers,
            device,
        })
    }

    pub fn forward(
        &self,
        input: &Tensor,
        train: Option<&mut Vec<TrainLinear>>,
    ) -> anyhow::Result<Tensor> {
        let input = input.to_device(&self.device)?;
        let mut train = train;

        let mut a = input.t()?;
        let mut z = None;

        if let Some(ref mut tl) = train.as_mut() {
            let dummy = Tensor::full(0.0, (1, 1), &self.device)?;
            tl.push(TrainLinear {
                layer: LinearLayer {
                    w: dummy.clone(),
                    b: dummy.clone(),
                },
                a: dummy.clone(),
                z: dummy.clone(),
                dw: dummy.clone(),
                db: dummy.clone(),
            });
        }

        for l in 0..self.layers.len() {
            let w = &self.layers[l].w;
            let b = &self.layers[l].b;
            let zl = w.matmul(&a)?.broadcast_add(&b)?;
            a = self.sigmoid(&zl)?;
            z = Some(zl.clone());
            if let Some(ref mut tl) = train.as_mut() {
                tl.push(TrainLinear {
                    layer: self.layers[l].clone(),
                    a: a.clone(),
                    z: zl.clone(),
                    dw: Tensor::full(1.0f32, w.shape(), &self.device)?,
                    db: Tensor::full(1.0f32, b.shape(), &self.device)?,
                });
            }
        }

        let z = z.unwrap();
        let r = softmax(&z, 0)?;
        if let Some(ref mut tl) = train.as_mut() {
            tl.push(TrainLinear {
                layer: self.layers.last().unwrap().clone(),
                a: a.clone(),
                z: z.clone(),
                dw: Tensor::full(1.0f32, (1, 1), &self.device)?,
                db: Tensor::full(1.0f32, (1, 1), &self.device)?,
            });
        }

        Ok(r)
    }

    fn backward(
        &self,
        x: &Tensor,
        y: &Tensor,
        batch: usize,
        current: &mut Vec<TrainLinear>,
    ) -> anyhow::Result<()> {
        current[0].a = x.t()?;
        let batch_div = Tensor::full(batch as f32, (1,), &self.device)?;

        let l = self.layers_sizes.len() - 1;

        let a = &current[l];
        let dz = (&a.a - y.t()?)?;

        let dza = dz.matmul(&current[l - 1].a.t()?)?;
        let dw = dza.broadcast_div(&batch_div)?;

        let dzsum = dz.sum_keepdim(1)?;
        let db = dzsum.broadcast_div(&batch_div)?;

        let mut daprev = current[l].layer.w.t()?.matmul(&dz)?;

        current[l].dw = dw;
        current[l].db = db;

        for l in (1..self.layers_sizes.len() - 1).rev() {
            let sigm_deriv = self.sigmoid_derivative(&current[l].z)?;

            let dz = (&daprev * &sigm_deriv)?;

            let dz_at = dz.matmul(&current[l - 1].a.t()?)?;
            let dw = dz_at.broadcast_div(&batch_div)?;

            let dz_sum = dz.sum_keepdim(1)?;
            let db = dz_sum.broadcast_div(&batch_div)?;
            if l > 1 {
                daprev = current[l].layer.w.t()?.matmul(&dz)?;
            }
            current[l].dw = dw;
            current[l].db = db;
        }

        Ok(())
    }

    fn fit(
        &mut self,
        x: &Tensor,
        y: &Tensor,
        learning_rate: f32,
        iterations: usize,
        batch: usize,
    ) -> anyhow::Result<()> {
        let x = x.to_device(&self.device)?;
        let y = y.to_device(&self.device)?;
        for l in 0..iterations {
            let mut loss = 0.0f32;
            let mut acc = 0.0f32;
            let mini_batches = util::create_mini_batches(&x, &y, batch, &self.device)?;
            let batch_len = mini_batches.len();
            for (x_part, y_part) in mini_batches {
                let mut store = Vec::<TrainLinear>::new();

                let a = self.forward(&x_part, Some(&mut store))?;
                let small = Tensor::full(1e-8f32, a.shape(), &self.device)?.t()?;
                let a_plus_eps_log = (&(a.t()? + small)?).log()?;
                let change: f32 = (&y_part.mul(&a_plus_eps_log)?)
                    .mean_all()?
                    .to_scalar::<f32>()?;
                loss -= change;

                self.backward(&x_part, &y_part, batch, &mut store)?;

                // And finally, apply the gradient descent to the current layers...
                for l in 1..self.layers_sizes.len() {
                    let w_change = Tensor::from_slice(&[learning_rate], (1, 1), &self.device)?
                        .broadcast_mul(&store[l].dw)?;

                    self.layers[l - 1].w = (&self.layers[l - 1].w - w_change)?;
                    self.layers[l - 1].b = (&self.layers[l - 1].b
                        - Tensor::from_slice(&[learning_rate], (1, 1), &self.device)?
                            .broadcast_mul(&store[l].db)?)?;
                }
                acc += self.predict(&x_part, &y_part)?;
            }
            acc = acc / batch_len as f32;
            loss = loss / batch_len as f32;
            println!(
                "Epoch: {l}, batch size: {batch}, steps {batch_len}, loss: {loss}, acc: {acc}"
            );
        }
        Ok(())
    }

    // Determine the ratio of correct answers.
    fn predict(&self, x: &Tensor, y: &Tensor) -> anyhow::Result<f32> {
        let a = self.forward(&x, None)?;
        let y_hat = a.argmax(0)?;
        let y_real = y.argmax(1)?;
        let same = y_hat.eq(&y_real)?.to_dtype(DType::F32)?;
        let v = same.mean_all()?.to_scalar::<f32>()?;
        Ok(v)
    }

    fn classify(&self, x: &Tensor) -> anyhow::Result<Vec<u8>> {
        let x = x.to_device(&self.device)?;
        let a = self.forward(&x, None)?;
        let y_hat = a.argmax(0)?;
        let y_hat_vec = y_hat.to_vec1::<u32>()?;
        Ok(y_hat_vec.iter().map(|x| *x as u8).collect())
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

    let mut ann = LinearNetworkManual::new(&[10, 10], 28 * 28, device)?;

    let mut t = vec![];
    let r = ann.forward(&m.train_images.i((0..64, ..))?, Some(&mut t))?;
    println!("r: {:?}", r.get(0)?.to_vec1::<f32>()?);
    println!("t: {:?}", t);

    let learning_rate = 0.1;
    // let iterations = 100;
    let iterations = 10;
    let batch_size = 64;
    ann.fit(
        &m.train_images,
        &m.train_labels,
        learning_rate,
        iterations,
        batch_size,
    )?;

    let test_range = 0..100;
    let digits = ann.classify(&m.test_images.i((test_range.clone(), ..))?)?;
    for (i, digit) in test_range.zip(digits.iter()) {
        let this_image = m.test_images.get(i)?;
        let this_image = util::mnist_image(&this_image)?;
        this_image.save(format!("/tmp/image_{i}_d_{digit}.png"))?;
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn matrix_mult() -> anyhow::Result<()> {
        let data: [f32; 6] = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::new(&data, &Device::Cpu)?;
        println!("tensor: {:?}", t.to_vec1::<f32>()?);
        println!("t: {t:#?}");
        let t = t.reshape((2, 3))?;
        println!("t: {t:#?}");

        let tt = t.t()?;

        let tsq = (tt.matmul(&t))?;
        println!("t: {:#?}, {:#?}", tsq.get(0)?, tsq.get(1)?);
        // let t2 = t.t()?;
        // let t3 = (t * t2)?;

        Ok(())
    }

    #[test]
    fn test_create_mini_batches() -> anyhow::Result<()> {
        let x: [[f32; 2]; 5] = [
            [1.0f32, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
        ];
        let x = Tensor::new(&x, &Device::Cpu)?;
        let y: [u8; 5] = [1, 8, 7, 3, 1];
        let y = Tensor::new(&y, &Device::Cpu)?;
        let mini_batches = crate::util::create_mini_batches(&x, &y, 2, &Device::Cpu)?;
        println!("mini_batches: {:#?} ", mini_batches);

        for (x_part, y_part) in mini_batches {
            println!("x: {:?}", x_part.to_vec2::<f32>()?);
            println!("y: {:?}", y_part.to_vec2::<f32>()?);
        }

        Ok(())
    }
}
