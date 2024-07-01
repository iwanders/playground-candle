use candle_core::IndexOp;
use candle_core::{DType, Device, Tensor};

// This collects groups of 'batch' size from inputs and outputs.
pub fn create_mini_batches(
    x: &Tensor,
    y: &Tensor,
    batch: usize,
    device: &Device,
) -> anyhow::Result<Vec<(Tensor, Tensor)>> {
    let mut mini = vec![];
    use rand::prelude::*;
    use rand_xorshift::XorShiftRng;
    let mut rng = XorShiftRng::seed_from_u64(1);

    let (input_count, input_width) = x.shape().dims2()?;

    // crate some shuffled indices;
    let mut shuffled_indices: Vec<usize> = (0..input_count).collect();
    shuffled_indices.shuffle(&mut rng);
    let one = Tensor::full(1.0 as f32, (1, 1), device)?;

    let batch_count = input_count / batch;

    for i in 0..batch_count {
        let mut input = Tensor::full(0.0 as f32, (batch, input_width), device)?;
        let mut output = Tensor::full(0.0 as f32, (batch, 10), device)?;
        for k in 0..batch {
            let in_index = shuffled_indices[i * batch + k];
            let input_data = x.i((in_index..=in_index, ..))?;
            let input_data = input_data.to_dtype(DType::F32)?;
            input = input.slice_assign(&[k..=k, 0..=input_width - 1], &input_data)?;

            let d = y.get(in_index)?.to_scalar::<u8>()? as usize;
            output = output.slice_assign(&[k..=k, d..=d], &one)?;
        }
        mini.push((input, output));
        if batch_count >= 10 {
            // break
        }
    }

    Ok(mini)
}


pub fn mnist_image(v: &Tensor) -> anyhow::Result<image::GrayImage> {
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
