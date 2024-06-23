use candle_core::{Tensor, Device, DType};
use candle_nn::ops::softmax;
use candle_core::IndexOp;
// Based on
// https://medium.com/@koushikkushal95/mnist-hand-written-digit-classification-using-neural-network-from-scratch-54da85712a06
// https://github.com/Koushikl0l/Machine_learning_from_scratch/blob/main/_nn_from_scratch__mini_batch_mnist.ipynb

// Pretty much a fully connected perceptron network?


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


struct Ann {
    layers_sizes: Vec<usize>,
    layers: Vec<LinearLayer>
}


fn sigmoid(z: &Tensor) -> anyhow::Result<Tensor> {
    let one = Tensor::full(1.0f32, z.shape(), &Device::Cpu)?;
    Ok((&one / (&one + z.neg()?.exp()?)?)?)
}
fn sigmoid_derivative(z: &Tensor) -> anyhow::Result<Tensor> {
    let s = sigmoid(z)?;
    let one = Tensor::full(1.0f32, z.shape(), &Device::Cpu)?;
    Ok((&s * (one - &s)?)?)
}

impl Ann{

    // This collects groups of 'batch' size from inputs and outputs.
    fn create_mini_batches(x: &Tensor, y: &Tensor, batch: usize) -> anyhow::Result<Vec<(Tensor, Tensor)>> {
        let mut mini = vec![];
        use rand_distr::{StandardNormal, Distribution};
        use rand::prelude::*;
        use rand_xorshift::XorShiftRng;
        let mut rng = XorShiftRng::seed_from_u64(1);

        let (input_count, input_width) = x.shape().dims2()?;
        let output_width = y.shape().dims1()?;
        // println!("input_count: {input_count:?}  input_width {input_width}     output_width {output_width}");
        // crate some shuffled indices;
        let mut shuffled_indices: Vec<usize> = (0..input_count).collect();
        shuffled_indices.shuffle(&mut rng);
        let one = Tensor::full(1.0 as f32, (1, 1), &Device::Cpu)?;

        let batch_count = input_count / batch;

        for i in 0..batch_count{
            let mut input = Tensor::full(0.0 as f32, (batch, input_width), &Device::Cpu)?;
            let mut output = Tensor::full(0.0 as f32, (batch, 10), &Device::Cpu)?;
            for k in 0..batch {
                let in_index = shuffled_indices[i * batch + k];
                let input_data = x.i((in_index..=in_index, ..))?;
                let input_data= input_data.to_dtype(DType::F32)?;
                input = input.slice_assign(&[k..=k, 0..=input_width-1], &input_data)?;

                let d = y.get(in_index)?.to_scalar::<u8>()? as usize;
                output = output.slice_assign(&[k..=k, d..=d], &one)?;
                // let output_data = y.i((in_index..=in_index))?;
                // let output_data= output_data.unsqueeze(1)?;
                // let output_data= output_data.to_dtype(DType::F32)?;
                // output = output.slice_assign(&[0..=0, k..=k], &output_data)?;
                
            }
            mini.push((input, output));
            // break;
        }

        Ok(mini)
    }

    pub fn create_layers(sizes: &[usize]) -> anyhow::Result<Vec<LinearLayer>> {
        // use rand::distributions::{Distribution, Uniform};
        // use rand::{RngCore, SeedableRng, Rng};
        use rand_distr::{StandardNormal, Distribution};
        use rand::prelude::*;

        use rand_xorshift::XorShiftRng;
        let mut rng = XorShiftRng::seed_from_u64(1);




        // let v : f64 = rng.sample(rand_distr::StandardNormal);
        
        let mut res = vec![];

        for l in 1..sizes.len() {
            let w_size = sizes[l] * sizes[l-1];
            let mut v : Vec<f32> = Vec::with_capacity(w_size);
            for _ in 0..w_size {
                v.push(rng.sample(rand_distr::StandardNormal));
            }
            let w = Tensor::from_vec(v, (sizes[l], sizes[l-1]), &Device::Cpu)?;
            let d = (sizes[l - 1] as f32).sqrt();
            let w = w.div(&Tensor::full(d, (sizes[l], sizes[l-1]), &Device::Cpu)?)?;

            let b = Tensor::full(0f32, (sizes[l], 1), &Device::Cpu)?;

            res.push(LinearLayer{w, b});
        }

        Ok(res)
    }

    pub fn new(layers_size: &[usize], input_size: usize) -> anyhow::Result<Self> {
        let mut layers_sizes = layers_size.to_vec();

        // First layer is input_size long.
        layers_sizes.insert(0, input_size);

        let layers = Self::create_layers(&layers_sizes)?;
        // let mut layers = vec![];
        // for l in layers.iter() {
            // println!("layer shapes: {:?}, {:?}", l.w.shape(), l.b.shape());
        // }
        Ok(Ann{
            layers_sizes,
            layers,
        })
    }

    pub fn forward(&self, input: &Tensor, train: Option<&mut Vec<TrainLinear>>) -> anyhow::Result<Tensor> {
        let mut train = train;
        // let input = input.unsqueeze(1)?;
        let mut a = input.t()?;
        // let mut a = input.clone();
        let mut z = None;
        for l in 0..self.layers.len() {
            // println!("l{l} a shape: {:?}", a.shape());
            // let LinearLayer{ref w, ref b} = *l;
            // This mess of transposes can probably be cleaned up a bit.
            // let zl = (a.matmul(&w.t()?)? + b.t()?)?;
            let w = &self.layers[l].w;
            let b = &self.layers[l].b;
            // println!(" w shape: {:?}", w.shape());
            // println!(" b shape: {:?}", b.shape());
            let zl = (w.matmul(&a)?.broadcast_add(&b)?);
            a = sigmoid(&zl)?;
            z = Some(zl.clone());
            if let Some(ref mut tl) =  train.as_mut() {
                tl.push(TrainLinear{
                    layer: self.layers[l].clone(),
                    a: a.clone(),
                    z: zl.clone(),
                    dw: Tensor::full(1.0f32, w.shape(), &Device::Cpu)?,
                    db: Tensor::full(1.0f32, b.shape(), &Device::Cpu)?,
                });
            }
        }

        let z = z.unwrap();
        let r = softmax(&z, 0)?;
        if let Some(ref mut tl) =  train.as_mut() {
            tl.push(TrainLinear{
                // layer: LinearLayer{w: Tensor::full(1.0f32, (1,1), &Device::Cpu)?, b: Tensor::full(1.0f32, (1,1), &Device::Cpu)?},
                layer: self.layers.last().unwrap().clone(),
                a: a.clone(),
                z: z.clone(),
                dw: Tensor::full(1.0f32, (1,1), &Device::Cpu)?,
                db: Tensor::full(1.0f32, (1,1), &Device::Cpu)?,
            });
        }

        Ok(r)
    }

    fn backward(&self, x: &Tensor, y: &Tensor, batch: usize, current: &mut Vec<TrainLinear>) -> anyhow::Result<()> {
        let mut a0 = x.t()?;
        current[0].a = a0.clone();
        // println!("x shape: {:?}, y shape: {:?}", x.shape(), y.shape());
        // println!("Current size: {}", current.len());

        let l = self.layers_sizes.len();

        let a = &current[l - 1];
        let dz  = (&a.a - y.t()?)?;

        let dza = dz.matmul(&current[l - 1].a.t()?)?;
        let batch_div = Tensor::full(batch as f32, dza.shape(), &Device::Cpu)?;
        let dw = dza.div(&batch_div)?;
        // println!("dw size: {:?}", dw.shape());
        // println!("dza size: {}", dza.shape());

        // println!("dz size: {:?}", dz.shape());
        let dzsum = dz.sum_keepdim(1)?;
        // println!("dzsum size: {:?}", dzsum.shape());
        let batch_div = Tensor::full(batch as f32, dzsum.shape(), &Device::Cpu)?;
        let db = dzsum.div(&batch_div)?;
        // println!("db size: {:?}", db.shape());

        let mut daprev = current[l - 1].layer.w.t()?.matmul(&dz)?;
        // println!("daprev size: {:?}", daprev.shape());

        current[l - 1].dw = dw;
        current[l - 1].db = db;

        for l in (1..self.layers_sizes.len() - 1).rev() {
            let sigm_deriv = sigmoid_derivative(&current[l].z)?;
            // println!("l {l} sigm_deriv size: {:?}", sigm_deriv.shape());
            let dz = (&daprev * &sigm_deriv)?;
            // println!("l {l} dz size: {:?}", dz.shape());
            let dz_at = dz.matmul(&current[l - 1].a.t()?)?;
            let batch_div = Tensor::full(batch as f32, dz_at.shape(), &Device::Cpu)?;
            let dw = dz_at.div(&batch_div)?;
            // println!("l {l} dw size: {:?}", dw.shape());
            let dz_sum = dz.sum_keepdim(1)?;
            let batch_div = Tensor::full(batch as f32, dz_sum.shape(), &Device::Cpu)?;
            let db = dz_sum.div(&batch_div)?;
            if l > 1 {
                daprev = current[l].layer.w.t()?.matmul(&dz)?;
            }
            current[l].dw = dw;
            current[l].db = db;
        }

        Ok(())
    }

    fn fit(&mut self, x: &Tensor, y: &Tensor, learning_rate: f32, iterations: usize, batch: usize) -> anyhow::Result<()>  {
        for l in 0..iterations {
            let mut loss = 0.0f32;
            let mut acc = 0.0f32;
            let mini_batches = Self::create_mini_batches(x, y, batch)?;
            let batch_len = mini_batches.len();
            for (x_part, y_part) in mini_batches {
                let mut store = Vec::<TrainLinear>::new();
                // println!("x part: {x_part:#?} y part: {y_part:#?}");
                let a = self.forward(&x_part, Some(&mut store))?;
                let small = Tensor::full(1e-8f32, a.shape(), &Device::Cpu)?.t()?;
                let a_plus_eps_log = (&(a.t()? + small)?).log()?;
                let change : f32 = (&y_part.mul(&a_plus_eps_log)?).mean_all()?.to_scalar::<f32>()?;
                loss -= change;

                let d = self.backward(&x_part, &y_part, batch, &mut store)?;

                // And finally, apply the gradient descent to the current layers...
                for l in 1..self.layers_sizes.len() {
                    let w_change = Tensor::from_slice(&[learning_rate], (1,1), &Device::Cpu)?.broadcast_mul(&store[l].dw)?;
                    // println!(" g{l} with w_change: {:?}", w_change.shape());
                    self.layers[l - 1].w = (&self.layers[l - 1].w - w_change)?;
                    self.layers[l - 1].b = (&self.layers[l - 1].b - Tensor::from_slice(&[learning_rate], (1,1), &Device::Cpu)?.broadcast_mul(&store[l].db)?)?;
                }
                acc += self.predict(&x_part, &y_part)?;
            }
            acc = acc / batch_len as f32;
            loss = loss / batch_len as f32;
            println!("Epoch: {l}, batches: {batch}, loss: {loss}, acc: {acc}");
        }
        Ok(())
    }

    // Determine the ratio of correct answers.
    fn predict(&self, x: &Tensor, y: &Tensor) -> anyhow::Result<f32> {
        let a = self.forward(&x, None)?;
        // println!("x shape {:?} y shape: {:?}  a shape {:?}", x.shape(), y.shape(), a.shape());
        let y_hat = a.argmax(0)?;
        let y_real = y.argmax(1)?;
        // println!("y_hat shape {:?} y_real shape: {:?}  ", y_hat.shape(), y_real.shape());
        let same = y_hat.eq(&y_real)?.to_dtype(DType::F32)?;
        let v = same.mean_all()?.to_scalar::<f32>()?;
        Ok(v)
    }
}





fn mnist_image(v: &Tensor) -> anyhow::Result<image::GrayImage> {
    // image is 28x28, input tensor is 1x784.
    let mut img = image::GrayImage::new(28, 28);
    for i in 0..v.shape().elem_count() {
        let c: f32 = v.get(i)?.to_vec0()?;
        let x = i as u32 % 28;
        let y = i as u32 / 28;
        let v: u8 = (c * 255.0f32) as u8;
        *img.get_pixel_mut(x, y) = image::Luma([v]);
    }
    Ok(img)
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
    let img_0 = mnist_image(&train_0)?;
    img_0.save("/tmp/image_0.png")?;

    let mut ann = Ann::new(&[10, 10], 28 * 28)?;

    let mut t = vec![];
    let r = ann.forward(&m.train_images.i((0..64,..))?, Some(&mut t))?;
    println!("r: {:?}", r.get(0)?.to_vec1::<f32>()?);
    println!("t: {:?}", t);


    let learning_rate = 0.1;
    let iterations = 100;
    let batch_size = 64;
    ann.fit(&m.train_images, &m.train_labels, learning_rate, iterations, batch_size)?;


    Ok(())
}


#[cfg(test)]
mod test{
    use super::*;
    #[test]
    fn matrix_mult() -> anyhow::Result<()> {
                
        let data: [f32; 6] = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::new(&data, &Device::Cpu)?;
        println!("tensor: {:?}", t.to_vec1::<f32>()?);
        println!("t: {t:#?}");
        let t = t.reshape((2,3))?;
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
        let x: [[f32; 2]; 3] = [[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let x = Tensor::new(&x, &Device::Cpu)?;
        let y: [[f32;1]; 3] = [[11.0f32], [21.0], [31.0]];
        let y = Tensor::new(&y, &Device::Cpu)?;
        let d = Ann::create_mini_batches(&x, &y, 2)?;
        println!("d: {:#?} ", d);

        

        Ok(())
    }
}
