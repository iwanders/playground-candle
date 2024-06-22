use candle_core::{Tensor};



fn mnist_image(v: &Tensor) -> anyhow::Result<image::GrayImage> {
    // image is 28x28, input tensor is 1x784.
    let mut img = image::GrayImage::new(28, 28);
    for i in 0..v.shape().elem_count() {
        let c: f32 = v.get(i)?.to_vec0()?;
        let x = i as u32 / 28;
        let y = i as u32 % 28;
        let v: u8 = (c * 255.0f32) as u8;
        *img.get_pixel_mut(y, x) = image::Luma([v]);
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


    Ok(())
}