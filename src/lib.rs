use candle_core::{Tensor, Device};

// Based on
// https://medium.com/@koushikkushal95/mnist-hand-written-digit-classification-using-neural-network-from-scratch-54da85712a06
// https://github.com/Koushikl0l/Machine_learning_from_scratch/blob/main/_nn_from_scratch__mini_batch_mnist.ipynb

// Pretty much a fully connected perceptron network?

enum Layer {
    Linear{
        // Z = W.T  X + b
        w: Tensor,
        b: Tensor,
    },
}

struct Ann {
    layers_sizes: Vec<usize>,
    layers: Vec<Layer>
}

impl Ann{
    pub fn create_layers(sizes: &[usize]) -> anyhow::Result<Vec<Layer>> {
        // use rand::distributions::{Distribution, Uniform};
        // use rand::{RngCore, SeedableRng, Rng};
        use rand_distr::{StandardNormal, Distribution};
        use rand::prelude::*;

        use rand_xorshift::XorShiftRng;
        let mut rng = XorShiftRng::seed_from_u64(1);


        let v : f64 = rng.sample(rand_distr::StandardNormal);
        
        let mut res = vec![];

        /*
        for l in range(1, len(self.layers_size)):
            self.parameters["W" + str(l)] = np.random.randn(self.layers_size[l], self.layers_size[l - 1]) / np.sqrt(
                self.layers_size[l - 1])
            self.parameters["b" + str(l)] = np.zeros((self.layers_size[l], 1))
        */
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

            res.push(Layer::Linear{w, b});
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
}


// Others;
// https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
// Interesting; https://www.kaggle.com/code/prashant111/mnist-deep-neural-network-with-keras





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
    let img_0 = mnist_image(&train_0)?;
    img_0.save("/tmp/image_0.png")?;

    let ann = Ann::new(&[10, 10], 28 * 28)?;

    Ok(())
}