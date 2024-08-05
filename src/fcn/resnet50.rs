use crate::candle_util::SequentialT;
use crate::candle_util::*;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Activation, ModuleT, VarBuilder, VarMap};
use super::create_data;


use clap::{Args, Parser, Subcommand};

/*
 Resnet 50:
    https://arxiv.org/pdf/1512.03385

    keras; https://github.com/keras-team/keras/blob/v3.3.3/keras/src/applications/resnet.py#L382-L418
    torch; https://github.com/pytorch/vision/blob/61bd547af1e26e5d1781a800391aa616df8de31f/torchvision/models/resnet.py#L736-L763
    roughly following the torch implementation.

    print(torchvision.models.resnet50())
*/

pub struct ResNet50 {
    network: SequentialT,
    device: Device,
}

impl ResNet50 {
    const BOTTLENECK_EXPANSION: usize = 4;
    pub fn from_path<P>(path: P, device: &Device) -> Result<Self>
    where
        P: AsRef<std::path::Path> + Copy,
    {
        let vs = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
        let resnet = ResNet50::new(vs, device)?;
        Ok(resnet)
    }

    pub fn new(vs: VarBuilder, device: &Device) -> Result<Self> {
        let mut network = SequentialT::new();


        fn conv1x1(in_planes: usize, out_planes: usize, vs: VarBuilder) -> Result<candle_nn::Conv2d> {
            candle_nn::conv2d_no_bias(in_planes, out_planes, 1, Default::default(), vs.clone())
        }
        fn conv3x3(in_planes: usize, out_planes: usize, stride: usize, vs: VarBuilder) -> Result<candle_nn::Conv2d> {
            let c = candle_nn::conv::Conv2dConfig {
                stride,
                padding: 1,
                ..Default::default()
            };
            candle_nn::conv2d_no_bias(in_planes, out_planes, 3, c, vs.clone())
        }

        struct BottleneckBlock {
            pub block: SequentialT,
            pub downsample: Option<SequentialT>,
        }
        impl ModuleT for BottleneckBlock {
            fn forward_t(&self, xs: &Tensor, train: bool) -> candle_core::Result<Tensor> {
                let ident = if let Some(ds) = self.downsample.as_ref() {
                    ds.forward_t(xs, train)?
                } else {
                    xs.clone()
                };
                let out = self.block.forward_t(xs, train)?;
                println!("going to addition");
                let res = out.add(&ident)?;
                Ok(res)
            }
        }

        fn create_block(inplanes: usize, planes: usize, stride: usize, vs: VarBuilder, downsample: Option<SequentialT>) -> Result<BottleneckBlock> {
            // This is the Bottleneck Block flavour.
            let width = planes * (64 / 64) * 1;
            let mut block = SequentialT::new();
            block.set_prefix(&vs.prefix());
            // size mismatch at line below, in first block.
            block.add(conv1x1(inplanes, width, vs.pp("conv1"))?);
            let prefix = vs.prefix();
            println!("{prefix}: conv1x1 {inplanes} {width}");
            block.add(candle_nn::batch_norm::batch_norm(width, candle_nn::BatchNormConfig::default(), vs.pp("bn1"))?);
            println!("{prefix}: batch_norm {width}");
            block.add(conv3x3(width, width, stride, vs.pp("conv2"))?);
            println!("{prefix}: conv3x3 {width} {width} s{stride}");
            block.add(candle_nn::batch_norm::batch_norm(width, candle_nn::BatchNormConfig::default(), vs.pp("bn2"))?);
            println!("{prefix}: batch_norm {width}");
            block.add(conv1x1(width, planes * ResNet50::BOTTLENECK_EXPANSION, vs.pp("conv3"))?);
            let final_out = planes * ResNet50::BOTTLENECK_EXPANSION;
            println!("{prefix}: conv1x1 {width} {} s{stride}", final_out);
            block.add(candle_nn::batch_norm::batch_norm(final_out, candle_nn::BatchNormConfig::default(), vs.pp("bn3"))?);
            println!("{prefix}: batch_norm {final_out}");
            println!();
            Ok(BottleneckBlock{ block, downsample })
        }


        // Okay, we now reached the 'layer' section.
        fn make_layer(inplanes: &[usize], planes: usize, stride: usize, vs: VarBuilder) -> Result<SequentialT> {
            let _ = stride;
            // Some complex stuff here with dilation and stride values.
            // Dilation will always be false for normal resnet 50?
            // Ignore that downsample layer for now.
            let mut block = SequentialT::new();
            block.set_prefix(&vs.prefix());
            // let inplanes = planes * ResNet50::BOTTLENECK_EXPANSION;
            let ds = {
                let mut ds = SequentialT::new();
                let prefix = vs.prefix();
                let out = *inplanes.last().unwrap();
                // ds.add(conv1x1(inplanes[0], out, vs.pp("downsample").pp(0))?);
                let c = candle_nn::conv::Conv2dConfig {
                    stride,
                    ..Default::default()
                };
                ds.add(candle_nn::conv2d_no_bias(inplanes[0], out, stride, c, vs.pp("downsample").pp(0))?);
                println!("{prefix} ds: conv2d_no_bias {} {out} s{stride}", inplanes[0]);
                ds.add(candle_nn::batch_norm::batch_norm(out, candle_nn::BatchNormConfig::default(), vs.pp("downsample").pp(1))?);
                println!("{prefix} ds: batch_norm {out}");
                ds
            };


            block.add(create_block(inplanes[0], planes, stride, vs.pp("block0"), Some(ds))?);
            block.add(Activation::Relu);
            for i in 1..inplanes.len() {
                block.add(create_block(inplanes[i], planes, 1, vs.pp(format!("block{i}")), None)?);
                block.add(Activation::Relu);
            }
            Ok(block)
        }



        
        let cp3s2 = candle_nn::conv::Conv2dConfig {
            padding: 3,
            stride: 2,
            dilation: 1,
            groups: 1,
        };

        // Block 1
        network.add(candle_nn::conv2d(3, 64, 7, cp3s2, vs.pp("conv1_1"))?); // 0
        network.add(candle_nn::batch_norm::batch_norm(64, candle_nn::BatchNormConfig::default(), vs.pp("bn1"))?); // 1
        network.add(Activation::Relu); // 2
        // In the original, this padding is inside the maxpool.
        network.add(Pad2DWithValueLayer::new(1, -100000f32));
        network.add(MaxPoolStrideLayer::new(3, 2)?); // 3
        network.add(ShapePrintLayer::new("Before layers"));

        network.add(make_layer(&[64, 256, 256], 64, 1, vs.pp("layer1"))?);
        network.add(ShapePrintLayer::new("After layer 1"));
        network.add(make_layer(&[256, 512, 512, 512], 128, 2, vs.pp("layer2"))?);
        network.add(make_layer(&[512, 1024, 1024, 1024, 1024, 1024], 256, 2, vs.pp("layer3"))?);
        network.add(make_layer(&[1024, 2048, 2048, 2048 ], 512, 2, vs.pp("layer4"))?);

        // Output of the backbone is here.
        // network.add(PanicLayer::new("got here"));
        Ok(Self {
            network,
            device: device.clone(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.to_device(&self.device)?;
        self.network.forward(&x)
    }

    pub fn add_clasifier_head(&mut self, classes:usize, vs: VarBuilder) -> Result<()> {
        // AdaptiveAvgPool2d uhh, we don't have this? 
        // But it's output size 1 by 1? Probably the same as meaning the last two dimensions?
        self.network.add(Avg2DLayer::new()?);

        // The fully connected layer would be here.
        self.network.add(candle_nn::linear(
                512 * ResNet50::BOTTLENECK_EXPANSION,
                classes,
                vs.pp(format!("fc1")),
            )?);

        Ok(())
    }
}

impl ModuleT for ResNet50 {
    fn forward_t(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let x = x.to_device(&self.device)?;
        self.network.forward_t(&x, train)
    }
}




#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    data_path: std::path::PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Args, Debug, Clone)]
pub struct Infer {
    #[arg(long)]
    /// The checkpoint file to load as initialisation
    load: Option<std::path::PathBuf>,


}

#[derive(Args, Debug, Clone)]
pub struct PrintArgs {
    load: std::path::PathBuf,
}

#[derive(Subcommand)]
enum Commands {
    /// Run inference using a model.
    Infer(Infer),
    /// Print tensors found in safetensor file.
    Print(PrintArgs),
}

pub fn main() -> std::result::Result<(), anyhow::Error> {
    let device = Device::new_cuda(0)?;
    // let device = Device::Cpu;

    println!("Building network");
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let mut network = ResNet50::new(vs.clone(), &device)?;
    network.add_clasifier_head(21, vs)?;

    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    // let network = ResNet50::new(vgg16, vs, &device)?;

    let cli = Cli::parse();

    for v in varmap.all_vars() {
        println!("var: {:?}   {v:?}", v.as_tensor().id())
    }

    match &cli.command {
        Commands::Infer(s) => {
            if let Some(v) = &s.load {
                varmap.load_into(&v, false)?;
            }


            let (tensor_samples_train, tensor_samples_val) =
                create_data(&cli.data_path, &["person", "cat", "bicycle", "bird"])?;
            // create_data(&cli.data_path, &CLASSESS[1..])?;

            todo!()
        }
        Commands::Print(p) => {
            let device = Device::Cpu;
            let z = load_from_safetensors(&p.load, &device)?;
            for k in z.keys() {
                println!("{k}");
            }
        }
    }

    Ok(())
}


#[cfg(test)]
mod test {

    use super::*;

    use crate::{error_unwrap}; // approx_equal, 
    use candle_core::Device;
    use candle_nn::{VarBuilder, VarMap};
    #[test]
    fn test_resnet_instantiate() -> Result<()> {
        let device = Device::Cpu;
        // let device = Device::new_cuda(0)?;

        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut network = error_unwrap!(ResNet50::new(vs.clone(), &device));
        let desired_classess = 2000;
        assert!(network.add_clasifier_head(desired_classess, vs).is_ok());


        // Create a dummy image.
        // Image is 224x224, 3 channels,  make it 0.5 gray
        let gray = Tensor::full(0.5f32, (3, 224, 224), &device)?;

        // Make a batch of two of these.
        let batch = Tensor::stack(&[&gray, &gray], 0)?;

        // Pass that into the network..
        let r = network.forward_t(&batch, false);

        // Do this here to get nice error message without newlines.
        let r = error_unwrap!(r);
        eprintln!("r shape: {:?}", r.shape());
        assert_eq!(r.shape().dims()[0], 2);
        assert_eq!(r.shape().dims()[1], desired_classess);

        Ok(())
    }
}
