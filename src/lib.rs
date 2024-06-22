use candle_core::{Tensor, Device};
use candle_nn::ops::softmax;
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

impl Ann{
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
        
        Ok(Ann{
            layers_sizes,
            layers,
        })
    }

    pub fn forward(&self, input: &Tensor, train: Option<&mut Vec<TrainLinear>>) -> anyhow::Result<Tensor> {
        let mut train = train;
        let input = input.unsqueeze(1)?;
        let mut a = input.t()?;
        let mut z = None;
        for (i, l) in self.layers.iter().enumerate() {
            let LinearLayer{ref w, ref b} = *l;
            // This mess of transposes can probably be cleaned up a bit.
            let zl = (a.matmul(&w.t()?)? + b.t()?)?;
            a = sigmoid(&zl)?;
            z = Some(zl.clone());
            if let Some(ref mut tl) =  train.as_mut() {
                tl.push(TrainLinear{
                    layer: l.clone(),
                    a: a.clone(),
                    z: zl.clone(),
                    dw: Tensor::full(1.0f32, l.w.shape(), &Device::Cpu)?,
                    db: Tensor::full(1.0f32, b.shape(), &Device::Cpu)?,
                });
            }
        }

        let z = z.unwrap();
        let r = softmax(&z, 0)?;

        Ok(r)
    }

    fn backward(&self, x: &Tensor, y: &Tensor, batch: usize, current: &[TrainLinear]) -> anyhow::Result<Vec<TrainLinear>> {
        let mut a0 = x.t()?;
        let dz  = (a0 - y.t()?)?;

        // let t = 
        // let dw = dz.matmul(self.

        Ok(vec![])
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

    let ann = Ann::new(&[10, 10], 28 * 28)?;

    let mut t = vec![];
    let r = ann.forward(&train_0, Some(&mut t))?;
    println!("r: {:?}", r.get(0)?.to_vec1::<f32>()?);
    println!("t: {:?}", t);
        

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
}
