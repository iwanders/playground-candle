
use candle_core::op::{BinaryOp, Op, ReduceOp, UnaryOp};
use candle_core::{Error, Result, Tensor, TensorId, Device, Var, IndexOp, backprop::GradStore};
use candle_core::test_utils;
use std::collections::HashMap;


// arg has been reduced to node via reduce_dims, expand it back to arg.
// This has to handle keepdims.
fn broadcast_back(arg: &Tensor, node: &Tensor, reduced_dims: &[usize]) -> Result<Tensor> {
    if arg.rank() == node.rank() {
        // keepdim = true
        node.broadcast_as(arg.shape())
    } else {
        // keepdim = false
        // first expand the reduced dims.
        node.reshape(reduced_dims)?.broadcast_as(arg.shape())
    }
}

fn sorted_nodes(this: &Tensor) -> Vec<&Tensor> {
    // The vec of sorted nodes is passed as an owned value rather than a mutable reference
    // to get around some lifetime limitations.
    fn walk<'a>(
        node: &'a Tensor,
        nodes: Vec<&'a Tensor>,
        already_seen: &mut HashMap<TensorId, bool>,
    ) -> (bool, Vec<&'a Tensor>) {
        if let Some(&tg) = already_seen.get(&node.id()) {
            return (tg, nodes);
        }
        let mut track_grad = false;
        let mut nodes = if node.is_variable() {
            // Do not call recursively on the "leaf" nodes.
            track_grad = true;
            nodes
        } else if node.dtype().is_int() {
            nodes
        } else if let Some(op) = node.op() {
            match op {
                Op::IndexAdd(t1, t2, t3, _)
                | Op::ScatterAdd(t1, t2, t3, _)
                | Op::CustomOp3(t1, t2, t3, _)
                | Op::WhereCond(t1, t2, t3) => {
                    let (tg, nodes) = walk(t1, nodes, already_seen);
                    track_grad |= tg;
                    let (tg, nodes) = walk(t2, nodes, already_seen);
                    track_grad |= tg;
                    let (tg, nodes) = walk(t3, nodes, already_seen);
                    track_grad |= tg;
                    nodes
                }
                Op::Conv1D {
                    arg: lhs,
                    kernel: rhs,
                    ..
                }
                | Op::ConvTranspose1D {
                    arg: lhs,
                    kernel: rhs,
                    ..
                }
                | Op::Conv2D {
                    arg: lhs,
                    kernel: rhs,
                    ..
                }
                | Op::ConvTranspose2D {
                    arg: lhs,
                    kernel: rhs,
                    ..
                }
                | Op::CustomOp2(lhs, rhs, _)
                | Op::Binary(lhs, rhs, _)
                | Op::Gather(lhs, rhs, _)
                | Op::IndexSelect(lhs, rhs, _)
                | Op::Matmul(lhs, rhs)
                | Op::SliceScatter0(lhs, rhs, _) => {
                    let (tg, nodes) = walk(lhs, nodes, already_seen);
                    track_grad |= tg;
                    let (tg, nodes) = walk(rhs, nodes, already_seen);
                    track_grad |= tg;
                    nodes
                }
                Op::Cat(args, _) => args.iter().fold(nodes, |nodes, arg| {
                    let (tg, nodes) = walk(arg, nodes, already_seen);
                    track_grad |= tg;
                    nodes
                }),
                Op::Affine { arg, mul, .. } => {
                    if *mul == 0. {
                        nodes
                    } else {
                        let (tg, nodes) = walk(arg, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }
                }
                Op::Unary(_node, UnaryOp::Ceil)
                | Op::Unary(_node, UnaryOp::Floor)
                | Op::Unary(_node, UnaryOp::Round)
                | Op::Unary(_node, UnaryOp::Sign) => nodes,
                Op::Reshape(node)
                | Op::UpsampleNearest1D { arg: node, .. }
                | Op::UpsampleNearest2D { arg: node, .. }
                | Op::AvgPool2D { arg: node, .. }
                | Op::MaxPool2D { arg: node, .. }
                | Op::Copy(node)
                | Op::Broadcast(node)
                | Op::Cmp(node, _)
                | Op::Reduce(node, ReduceOp::Min | ReduceOp::Sum | ReduceOp::Max, _)
                | Op::ToDevice(node)
                | Op::Transpose(node, _, _)
                | Op::Permute(node, _)
                | Op::Narrow(node, _, _, _)
                | Op::Unary(node, _)
                | Op::Elu(node, _)
                | Op::Powf(node, _)
                | Op::CustomOp1(node, _) => {
                    let (tg, nodes) = walk(node, nodes, already_seen);
                    track_grad |= tg;
                    nodes
                }
                Op::ToDType(node) => {
                    if node.dtype().is_float() {
                        let (tg, nodes) = walk(node, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    } else {
                        nodes
                    }
                }
                Op::Reduce(_, ReduceOp::ArgMin | ReduceOp::ArgMax, _) => nodes,
            }
        } else {
            nodes
        };
        already_seen.insert(node.id(), track_grad);
        if track_grad {
            nodes.push(node);
        }
        (track_grad, nodes)
    }
    let (_tg, mut nodes) = walk(this, vec![], &mut HashMap::new());
    nodes.reverse();
    nodes
}

fn our_backward(this: &Tensor) -> Result<GradStore>{
    // let self = z;
    let sorted_nodes = sorted_nodes(this);
    let mut grads = GradStore::new();
    grads.insert(this, this.ones_like()?.contiguous()?);
    for node in sorted_nodes.iter() {
        if node.is_variable() {
            continue;
        }
        let grad = grads
            .remove(node)
            .expect("candle internal error - grad not populated");
        // https://github.com/huggingface/candle/issues/1241
        // Ideally, we would make these operations in place where possible to ensure that we
        // do not have to allocate too often. Here we just call `.detach` to avoid computing
        // the backprop graph of the backprop itthis. This would be an issue for second order
        // derivatives but these are out of scope at the moment.
        let do_not_detach = false;
        let grad = if do_not_detach { grad } else { grad.detach() };
        if let Some(op) = node.op() {
            match op {
                Op::Binary(lhs, rhs, BinaryOp::Add) => {
                    let lhs_sum_grad = grads.or_insert(lhs)?;
                    *lhs_sum_grad = lhs_sum_grad.add(&grad)?;
                    let rhs_sum_grad = grads.or_insert(rhs)?;
                    *rhs_sum_grad = rhs_sum_grad.add(&grad)?;
                }
                Op::Binary(lhs, rhs, BinaryOp::Sub) => {
                    let lhs_sum_grad = grads.or_insert(lhs)?;
                    *lhs_sum_grad = lhs_sum_grad.add(&grad)?;
                    let rhs_sum_grad = grads.or_insert(rhs)?;
                    *rhs_sum_grad = rhs_sum_grad.sub(&grad)?;
                }
                Op::Binary(lhs, rhs, BinaryOp::Mul) => {
                    let lhs_grad = grad.mul(rhs)?;
                    let lhs_sum_grad = grads.or_insert(lhs)?;
                    *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;
                    let rhs_grad = grad.mul(lhs)?;
                    let rhs_sum_grad = grads.or_insert(rhs)?;
                    *rhs_sum_grad = rhs_sum_grad.add(&rhs_grad)?;
                }
                Op::Binary(lhs, rhs, BinaryOp::Div) => {
                    let lhs_grad = grad.div(rhs)?;
                    let lhs_sum_grad = grads.or_insert(lhs)?;
                    *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;
                    let rhs_grad = grad.mul(lhs)?.div(&rhs.sqr()?)?;
                    let rhs_sum_grad = grads.or_insert(rhs)?;
                    *rhs_sum_grad = rhs_sum_grad.sub(&rhs_grad)?;
                }
                Op::Binary(lhs, rhs, BinaryOp::Minimum)
                | Op::Binary(lhs, rhs, BinaryOp::Maximum) => {
                    let mask_lhs = node.eq(lhs)?.to_dtype(grad.dtype())?;
                    let mask_rhs = node.eq(rhs)?.to_dtype(grad.dtype())?;

                    // If both masks are 1 one the same point, we want to scale the
                    // gradient by 0.5 rather than 1.
                    let lhs_grad = mask_lhs.mul(&grad)?.div(&(&mask_rhs + 1.)?)?;
                    let lhs_sum_grad = grads.or_insert(lhs)?;
                    *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;

                    let rhs_grad = mask_rhs.mul(&grad)?.div(&(&mask_lhs + 1.)?)?;
                    let rhs_sum_grad = grads.or_insert(rhs)?;
                    *rhs_sum_grad = rhs_sum_grad.add(&rhs_grad)?;
                }
                Op::WhereCond(pred, t, f) => {
                    let zeros = grad.zeros_like()?;
                    let t_sum_grad = grads.or_insert(t)?;
                    let t_grad = pred.where_cond(&grad, &zeros)?;
                    *t_sum_grad = t_sum_grad.add(&t_grad)?;
                    let f_sum_grad = grads.or_insert(f)?;
                    let f_grad = pred.where_cond(&zeros, &grad)?;
                    *f_sum_grad = f_sum_grad.add(&f_grad)?;
                }
                Op::Conv1D {
                    arg,
                    kernel,
                    padding,
                    stride,
                    dilation,
                } => {
                    // The output height for conv_transpose1d is:
                    // (l_in - 1) * stride - 2 * padding + dilation * (k_size - 1) + out_padding + 1
                    let grad_l_in = grad.dim(2)?;
                    let k_size = kernel.dim(2)?;
                    let out_size =
                        (grad_l_in - 1) * stride + dilation * (k_size - 1) + 1 - 2 * padding;
                    let out_padding = arg.dim(2)? - out_size;
                    let grad_arg = grad.conv_transpose1d(
                        kernel,
                        *padding,
                        out_padding,
                        *stride,
                        *dilation,
                        /* groups */ 1,
                    )?;
                    let sum_grad = grads.or_insert(arg)?;
                    *sum_grad = sum_grad.add(&grad_arg)?;

                    let grad_kernel = arg
                        .transpose(0, 1)?
                        .conv1d(&grad.transpose(0, 1)?, *padding, *dilation, *stride, 1)?
                        .transpose(0, 1)?;
                    let sum_grad = grads.or_insert(kernel)?;
                    let (_, _, k0) = kernel.dims3()?;
                    let (_, _, g_k0) = grad_kernel.dims3()?;
                    let grad_kernel = if g_k0 != k0 {
                        grad_kernel.narrow(2, 0, k0)?
                    } else {
                        grad_kernel
                    };
                    *sum_grad = sum_grad.add(&grad_kernel)?;
                }
                Op::Conv2D {
                    arg,
                    kernel,
                    padding,
                    stride,
                    dilation,
                } => {
                    // The output height for conv_transpose2d is:
                    // (i_h - 1) * stride - 2 * padding + dilation * (k_h - 1) + out_padding + 1
                    let grad_h = grad.dim(2)?;
                    let k_h = kernel.dim(2)?;
                    let out_size =
                        (grad_h - 1) * stride + dilation * (k_h - 1) + 1 - 2 * padding;
                    let out_padding = arg.dim(2)? - out_size;
                    let grad_arg = grad.conv_transpose2d(
                        kernel,
                        *padding,
                        out_padding,
                        *stride,
                        *dilation,
                    )?;
                    let sum_grad = grads.or_insert(arg)?;
                    *sum_grad = sum_grad.add(&grad_arg)?;

                    let grad_kernel = arg
                        .transpose(0, 1)?
                        .conv2d(&grad.transpose(0, 1)?, *padding, *dilation, *stride, 1)?
                        .transpose(0, 1)?;
                    let sum_grad = grads.or_insert(kernel)?;
                    let (_, _, k0, k1) = kernel.dims4()?;
                    let (_, _, g_k0, g_k1) = grad_kernel.dims4()?;
                    let grad_kernel = if g_k0 != k0 || g_k1 != k1 {
                        grad_kernel.narrow(2, 0, k0)?.narrow(3, 0, k1)?
                    } else {
                        grad_kernel
                    };
                    *sum_grad = sum_grad.add(&grad_kernel)?;
                }
                Op::ConvTranspose1D { .. } => Err(Error::BackwardNotSupported {
                    op: "conv-transpose1d",
                })?,
                Op::ConvTranspose2D {
                    arg,
                    kernel,
                    padding,
                    stride,
                    dilation,
                    output_padding: _output_padding,
                } => {
                    /*
pub fn conv_transpose2d(
    &self,
    kernel: &Self,
    padding: usize,
    output_padding: usize,
    stride: usize,
    dilation: usize,
) -> Result<Self>
pub fn conv2d(
    &self,
    kernel: &Self,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> Result<Self>
*/
                    println!("stride: {stride:?}");
                    let grad_arg = grad.conv2d(kernel, *padding, *stride, *dilation, 1)?;
                    // println!("grad_arg: {grad_arg:?}");
                    // println!("zarg: {arg:?}");
                    let sum_grad = grads.or_insert(arg)?;
                    // dbg!();
                    *sum_grad = sum_grad.add(&grad_arg)?;
                    // dbg!();

                    let grad_kernel = grad
                        .transpose(0, 1)?
                        .conv2d(&arg.transpose(0, 1)?, *padding, *dilation, *stride, 1)?
                        .transpose(0, 1)?;
                    let sum_grad = grads.or_insert(kernel)?;
                    let (_, _, k0, k1) = kernel.dims4()?;
                    let (_, _, g_k0, g_k1) = grad_kernel.dims4()?;
                    let grad_kernel = if g_k0 != k0 || g_k1 != k1 {
                        grad_kernel.narrow(2, 0, k0)?.narrow(3, 0, k1)?
                    } else {
                        grad_kernel
                    };
                    *sum_grad = sum_grad.add(&grad_kernel)?;
                }
                Op::AvgPool2D {
                    arg,
                    kernel_size,
                    stride,
                } => {
                    if kernel_size != stride {
                        candle_core::bail!("backward not supported for avgpool2d if ksize {kernel_size:?} != stride {stride:?}")
                    }
                    let (_n, _c, h, w) = arg.dims4()?;
                    let grad_arg = grad.upsample_nearest2d(h, w)?;
                    let grad_arg =
                        (grad_arg * (1f64 / (kernel_size.0 * kernel_size.1) as f64))?;
                    let sum_grad = grads.or_insert(arg)?;
                    *sum_grad = sum_grad.add(&grad_arg)?;
                }
                Op::MaxPool2D {
                    arg,
                    kernel_size,
                    stride,
                } => {
                    if kernel_size != stride {
                        candle_core::bail!("backward not supported for maxpool2d if ksize {kernel_size:?} != stride {stride:?}")
                    }
                    let (_n, _c, h, w) = arg.dims4()?;
                    // For computing the max-pool gradient, we compute a mask where a 1 means
                    // that the element is the maximum, then we apply this mask to the
                    // upsampled gradient (taking into account that multiple max may exist so
                    // we scale the gradient for this case).
                    let node_upsampled = node.upsample_nearest2d(h, w)?;
                    let mask = arg.eq(&node_upsampled)?.to_dtype(arg.dtype())?;
                    let avg = mask.avg_pool2d_with_stride(*kernel_size, *stride)?;
                    let grad_arg = ((grad * avg)?.upsample_nearest2d(h, w)? * mask)?;
                    let sum_grad = grads.or_insert(arg)?;
                    *sum_grad = sum_grad.add(&grad_arg)?;
                }
                Op::UpsampleNearest1D { arg, target_size } => {
                    let (_n, c, size) = arg.dims3()?;
                    if target_size % size != 0 {
                        candle_core::bail!("backward not supported for non integer upscaling factors")
                    }
                    let scale = target_size / size;

                    let kernel = Tensor::ones((c, 1, scale), arg.dtype(), arg.device())?;
                    let conv_sum = grad.conv1d(&kernel, 0, scale, 1, c)?;
                    let sum_grad = grads.or_insert(arg)?;
                    *sum_grad = conv_sum;
                }
                Op::UpsampleNearest2D {
                    arg,
                    target_h,
                    target_w,
                } => {
                    let (_n, c, h, w) = arg.dims4()?;
                    if target_h % h != 0 || target_w % w != 0 {
                        candle_core::bail!("backward not supported for non integer upscaling factors")
                    }
                    let scale_h = target_h / h;
                    let scale_w = target_w / w;

                    if scale_h != scale_w {
                        candle_core::bail!("backward not supported for non uniform upscaling factors")
                    };
                    let kernel =
                        Tensor::ones((c, 1, scale_h, scale_w), arg.dtype(), arg.device())?;
                    let conv_sum = grad.conv2d(&kernel, 0, scale_h, 1, c)?;
                    let sum_grad = grads.or_insert(arg)?;
                    *sum_grad = conv_sum;
                }
                Op::SliceScatter0(lhs, rhs, start_rhs) => {
                    let rhs_sum_grad = grads.or_insert(rhs)?;
                    let rhs_grad = grad.narrow(0, *start_rhs, rhs.dim(0)?)?;
                    *rhs_sum_grad = rhs_sum_grad.add(&rhs_grad)?;

                    let lhs_sum_grad = grads.or_insert(lhs)?;
                    let lhs_grad = grad.slice_scatter0(&rhs.zeros_like()?, *start_rhs)?;
                    *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?
                }
                Op::Gather(arg, indexes, dim) => {
                    let sum_grad = grads.or_insert(arg)?;
                    *sum_grad = sum_grad.scatter_add(indexes, &grad, *dim)?;
                }
                Op::ScatterAdd(init, indexes, src, dim) => {
                    let init_sum_grad = grads.or_insert(init)?;
                    *init_sum_grad = init_sum_grad.add(&grad)?;

                    let src_grad = grad.gather(indexes, *dim)?;
                    let src_sum_grad = grads.or_insert(src)?;
                    *src_sum_grad = src_sum_grad.add(&src_grad)?;
                }
                Op::IndexAdd(init, indexes, src, dim) => {
                    let init_sum_grad = grads.or_insert(init)?;
                    *init_sum_grad = init_sum_grad.add(&grad)?;

                    let src_grad = grad.index_select(indexes, *dim)?;
                    let src_sum_grad = grads.or_insert(src)?;
                    *src_sum_grad = src_sum_grad.add(&src_grad)?;
                }
                Op::IndexSelect(arg, indexes, dim) => {
                    let sum_grad = grads.or_insert(arg)?;
                    *sum_grad = sum_grad.index_add(indexes, &grad, *dim)?;
                }
                Op::Matmul(lhs, rhs) => {
                    // Skipping checks, the op went ok, we can skip
                    // the matmul size checks for now.

                    let lhs_grad = grad.matmul(&rhs.t()?)?;
                    let lhs_sum_grad = grads.or_insert(lhs)?;
                    *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;

                    let rhs_grad = lhs.t()?.matmul(&grad)?;
                    let rhs_sum_grad = grads.or_insert(rhs)?;
                    *rhs_sum_grad = rhs_sum_grad.add(&rhs_grad)?;
                }
                Op::Cat(args, dim) => {
                    let mut start_idx = 0;
                    for arg in args {
                        let len = arg.dims()[*dim];
                        let arg_grad = grad.narrow(*dim, start_idx, len)?;
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&arg_grad)?;
                        start_idx += len;
                    }
                }
                Op::Broadcast(arg) => {
                    let arg_dims = arg.dims();
                    let node_dims = node.dims();
                    // The number of dims that have been inserted on the left.
                    let left_dims = node_dims.len() - arg_dims.len();
                    let mut sum_dims: Vec<usize> = (0..left_dims).collect();
                    for (dim, (node_dim, arg_dim)) in node_dims[left_dims..]
                        .iter()
                        .zip(arg_dims.iter())
                        .enumerate()
                    {
                        if node_dim != arg_dim {
                            sum_dims.push(dim + left_dims)
                        }
                    }

                    let mut arg_grad = grad.sum_keepdim(sum_dims.as_slice())?;
                    for _i in 0..left_dims {
                        arg_grad = arg_grad.squeeze(0)?
                    }
                    let sum_grad = grads.or_insert(arg)?;
                    *sum_grad = sum_grad.add(&arg_grad.broadcast_as(sum_grad.dims())?)?;
                }
                Op::Reduce(arg, ReduceOp::Sum, reduced_dims) => {
                    let grad = broadcast_back(arg, &grad, reduced_dims)?;
                    let sum_grad = grads.or_insert(arg)?;
                    *sum_grad = sum_grad.add(&grad)?;
                }
                Op::Reduce(arg, ReduceOp::Max, reduced_dims) => {
                    let node = broadcast_back(arg, node, reduced_dims)?;
                    let grad = broadcast_back(arg, &grad, reduced_dims)?;
                    let grad = node.eq(arg)?.to_dtype(grad.dtype())?.mul(&grad)?;
                    let sum_grad = grads.or_insert(arg)?;
                    *sum_grad = sum_grad.add(&grad.broadcast_as(sum_grad.dims())?)?;
                }
                Op::Reduce(arg, ReduceOp::Min, reduced_dims) => {
                    let node = broadcast_back(arg, node, reduced_dims)?;
                    let grad = broadcast_back(arg, &grad, reduced_dims)?;
                    let grad = node.eq(arg)?.to_dtype(grad.dtype())?.mul(&grad)?;
                    let sum_grad = grads.or_insert(arg)?;
                    *sum_grad = sum_grad.add(&grad.broadcast_as(sum_grad.dims())?)?;
                }
                Op::ToDType(arg) => {
                    let sum_grad = grads.or_insert(arg)?;
                    *sum_grad = sum_grad.add(&grad.to_dtype(arg.dtype())?)?
                }
                Op::Copy(arg) => {
                    let sum_grad = grads.or_insert(arg)?;
                    *sum_grad = sum_grad.add(&grad)?
                }
                Op::Affine { arg, mul, .. } => {
                    let arg_grad = grad.affine(*mul, 0.)?;
                    let sum_grad = grads.or_insert(arg)?;
                    *sum_grad = sum_grad.add(&arg_grad)?
                }
                Op::Unary(arg, UnaryOp::Log) => {
                    let sum_grad = grads.or_insert(arg)?;
                    *sum_grad = sum_grad.add(&(grad / arg)?)?
                }
                Op::Unary(arg, UnaryOp::Sin) => {
                    let sum_grad = grads.or_insert(arg)?;
                    *sum_grad = sum_grad.add(&(&grad * arg.cos())?)?
                }
                Op::Unary(arg, UnaryOp::Cos) => {
                    let sum_grad = grads.or_insert(arg)?;
                    *sum_grad = sum_grad.sub(&(&grad * arg.sin())?)?
                }
                Op::Unary(arg, UnaryOp::Tanh) => {
                    let sum_grad = grads.or_insert(arg)?;
                    let minus_dtanh = (node.sqr()? - 1.)?;
                    *sum_grad = sum_grad.sub(&(&grad * &minus_dtanh)?)?
                }
                Op::Unary(arg, UnaryOp::Abs) => {
                    let sum_grad = grads.or_insert(arg)?;
                    let ones = arg.ones_like()?;
                    let abs_grad = arg.ge(&arg.zeros_like()?)?.where_cond(&ones, &ones.neg()?);
                    *sum_grad = sum_grad.add(&(&grad * abs_grad)?)?
                }
                Op::Unary(arg, UnaryOp::Exp) => {
                    let sum_grad = grads.or_insert(arg)?;
                    *sum_grad = sum_grad.add(&(&grad * *node)?)?
                }
                Op::Unary(arg, UnaryOp::Neg) => {
                    let sum_grad = grads.or_insert(arg)?;
                    *sum_grad = sum_grad.sub(&grad)?
                }
                Op::Unary(arg, UnaryOp::Recip) => {
                    let sum_grad = grads.or_insert(arg)?;
                    let grad = (grad / arg.sqr()?)?;
                    *sum_grad = sum_grad.sub(&grad)?
                }
                &Op::Narrow(ref arg, dim, start_idx, len) => {
                    let arg_dims = arg.dims();
                    let left_pad = if start_idx == 0 {
                        None
                    } else {
                        let mut dims = arg_dims.to_vec();
                        dims[dim] = start_idx;
                        Some(Tensor::zeros(dims, grad.dtype(), grad.device())?)
                    };
                    let right_pad = arg_dims[dim] - start_idx - len;
                    let right_pad = if right_pad == 0 {
                        None
                    } else {
                        let mut dims = arg_dims.to_vec();
                        dims[dim] = right_pad;
                        Some(Tensor::zeros(dims, grad.dtype(), grad.device())?)
                    };
                    let arg_grad = match (left_pad, right_pad) {
                        (None, None) => grad,
                        (Some(l), None) => Tensor::cat(&[&l, &grad], dim)?,
                        (None, Some(r)) => Tensor::cat(&[&grad, &r], dim)?,
                        (Some(l), Some(r)) => Tensor::cat(&[&l, &grad, &r], dim)?,
                    };
                    let sum_grad = grads.or_insert(arg)?;
                    *sum_grad = sum_grad.add(&arg_grad)?
                }
                Op::Unary(_, UnaryOp::Floor)
                | Op::Unary(_, UnaryOp::Round)
                | Op::Reduce(_, ReduceOp::ArgMin, _)
                | Op::Reduce(_, ReduceOp::ArgMax, _)
                | Op::Unary(_, UnaryOp::Sign)
                | Op::Cmp(_, _) => {}
                Op::Reshape(arg) => {
                    let arg_grad = grad.reshape(arg.dims())?;
                    let sum_grad = grads.or_insert(arg)?;
                    *sum_grad = sum_grad.add(&arg_grad)?
                }
                Op::Unary(_, UnaryOp::Ceil) => Err(Error::BackwardNotSupported { op: "ceil" })?,
                Op::Unary(arg, UnaryOp::Gelu) => {
                    let sum_grad = grads.or_insert(arg)?;
                    let cube = arg.powf(3.)?;
                    let tanh = (0.0356774 * &cube + (0.797885 * arg)?)?.tanh()?;
                    let gelu_grad = (((0.5 * &tanh)?
                        + (0.0535161 * cube + (0.398942 * arg)?)? * (1. - tanh.powf(2.)?))?
                        + 0.5)?;
                    *sum_grad = sum_grad.add(&(&grad * gelu_grad)?)?
                }
                Op::Unary(arg, UnaryOp::Erf) => {
                    let sum_grad = grads.or_insert(arg)?;
                    // d/dx erf(x) = 2/sqrt(pi) * e^(-x^2)
                    let erf_grad =
                        (2. / std::f64::consts::PI.sqrt()) * (arg.sqr()?.neg()?).exp()?;
                    *sum_grad = sum_grad.add(&(&grad * erf_grad)?)?
                }
                Op::Unary(arg, UnaryOp::GeluErf) => {
                    let sum_grad = grads.or_insert(arg)?;
                    // d/dx gelu_erf(x) = 0.5 + 0.398942 e^(-x^2/2) x + 0.5 erf(x/sqrt(2))
                    let neg_half_square = (arg.sqr()?.neg()? / 2.)?;
                    let scaled_exp_arg = (0.398942 * neg_half_square.exp()? * arg)?;
                    let arg_scaled_sqrt = (arg / 2f64.sqrt())?;
                    let erf_scaled_sqrt = (0.5 * arg_scaled_sqrt.erf()?)?;
                    let gelu_erf_grad = (0.5 + scaled_exp_arg + erf_scaled_sqrt)?;
                    *sum_grad = sum_grad.add(&(&grad * gelu_erf_grad)?)?;
                }
                Op::Unary(arg, UnaryOp::Relu) => {
                    let sum_grad = grads.or_insert(arg)?;
                    let relu_grad = arg.ge(&arg.zeros_like()?)?.to_dtype(arg.dtype())?;
                    *sum_grad = sum_grad.add(&(&grad * relu_grad)?)?
                }
                Op::Unary(arg, UnaryOp::Silu) => {
                    let sum_grad = grads.or_insert(arg)?;
                    // d/dx silu = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
                    let sigmoid_arg = (arg.neg()?.exp()? + 1.)?.recip()?;
                    let silu_grad = (&sigmoid_arg * (1. + (arg * (1. - &sigmoid_arg)?)?)?)?;
                    *sum_grad = sum_grad.add(&(&grad * silu_grad)?)?
                }
                Op::Elu(arg, alpha) => {
                    // d/dx elu(x) = 1 for x > 0, alpha * e^x for x <= 0
                    let sum_grad = grads.or_insert(arg)?;
                    let zeros = arg.zeros_like()?;
                    let positive_mask = arg.gt(&zeros)?.to_dtype(arg.dtype())?;
                    let negative_mask = arg.le(&zeros)?.to_dtype(arg.dtype())?;
                    let negative_exp_mask = ((negative_mask * arg.exp())? * *alpha)?;
                    let combined_mask = (positive_mask + negative_exp_mask)?;
                    *sum_grad = sum_grad.add(&(grad * combined_mask)?)?
                }
                Op::Powf(arg, e) => {
                    let arg_grad = (&(grad * arg.powf(e - 1.)?)? * *e)?;
                    let sum_grad = grads.or_insert(arg)?;
                    *sum_grad = sum_grad.add(&arg_grad)?
                }
                Op::CustomOp1(arg, c) => {
                    if let Some(arg_grad) = c.bwd(arg, node, &grad)? {
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&arg_grad)?
                    }
                }
                Op::CustomOp2(arg1, arg2, c) => {
                    let (arg_grad1, arg_grad2) = c.bwd(arg1, arg2, node, &grad)?;
                    if let Some(arg_grad1) = arg_grad1 {
                        let sum_grad = grads.or_insert(arg1)?;
                        *sum_grad = sum_grad.add(&arg_grad1)?
                    }
                    if let Some(arg_grad2) = arg_grad2 {
                        let sum_grad = grads.or_insert(arg2)?;
                        *sum_grad = sum_grad.add(&arg_grad2)?
                    }
                }
                Op::CustomOp3(arg1, arg2, arg3, c) => {
                    let (arg_grad1, arg_grad2, arg_grad3) =
                        c.bwd(arg1, arg2, arg3, node, &grad)?;
                    if let Some(arg_grad1) = arg_grad1 {
                        let sum_grad = grads.or_insert(arg1)?;
                        *sum_grad = sum_grad.add(&arg_grad1)?
                    }
                    if let Some(arg_grad2) = arg_grad2 {
                        let sum_grad = grads.or_insert(arg2)?;
                        *sum_grad = sum_grad.add(&arg_grad2)?
                    }
                    if let Some(arg_grad3) = arg_grad3 {
                        let sum_grad = grads.or_insert(arg3)?;
                        *sum_grad = sum_grad.add(&arg_grad3)?
                    }
                }
                Op::Unary(arg, UnaryOp::Sqr) => {
                    let arg_grad = arg.mul(&grad)?.affine(2., 0.)?;
                    let sum_grad = grads.or_insert(arg)?;
                    *sum_grad = sum_grad.add(&arg_grad)?
                }
                Op::Unary(arg, UnaryOp::Sqrt) => {
                    let arg_grad = grad.div(node)?.affine(0.5, 0.)?;
                    let sum_grad = grads.or_insert(arg)?;
                    *sum_grad = sum_grad.add(&arg_grad)?
                }
                Op::ToDevice(arg) => {
                    let sum_grad = grads.or_insert(arg)?;
                    let arg_grad = grad.to_device(sum_grad.device())?;
                    *sum_grad = sum_grad.add(&arg_grad)?
                }
                Op::Transpose(arg, dim1, dim2) => {
                    let arg_grad = grad.transpose(*dim1, *dim2)?;
                    let sum_grad = grads.or_insert(arg)?;
                    *sum_grad = sum_grad.add(&arg_grad)?
                }
                Op::Permute(arg, dims) => {
                    let mut inv_dims = vec![0; dims.len()];
                    for (i, &dim_idx) in dims.iter().enumerate() {
                        inv_dims[dim_idx] = i
                    }
                    let arg_grad = grad.permute(inv_dims)?;
                    let sum_grad = grads.or_insert(arg)?;
                    *sum_grad = sum_grad.add(&arg_grad)?
                }
            };
        }
    }
    Ok(grads)
    
}



pub fn main() -> super::MainResult {
    let device = Device::Cpu;
    let dev = &device;
    // import torch
    // torch.manual_seed(4242)
    // padding = 4
    // outpadding = 2
    // dilation = 3
    // stride = 3
    // input = torch.randn((1, 4, 7, 5), requires_grad=True)
    // kernel = torch.randn((4, 2, 3, 5), requires_grad=True)
    // print("input", input.flatten())
    // print("kernel", kernel.flatten())
    // res = torch.nn.functional.conv_transpose2d(
    //     input,
    //     kernel,
    //     stride=stride,
    //     padding=padding,
    //     dilation=dilation,
    //     output_padding=outpadding,
    // )
    // res.retain_grad()
    // print("res.shape: ", res.shape)
    // loss = (res**2).sum()
    // print(loss)
    // loss.backward()
    // print("input.grad.shape: ", input.grad.shape)
    // print("input grad", torch.round(input.grad, decimals=1))
    // print("kernel.grad.shape", kernel.grad.shape)
    // print("kernel grad", torch.round(kernel.grad.flatten(), decimals=1))

    let padding = 4;
    let outpadding = 2;
    let dilation = 3;
    let stride = 3;

    let t = Var::from_slice(
        &[
            0.4056_f32, -0.8689, -0.0773, -1.5630, -2.8012, -1.5059, 0.3972, 1.0852, 0.4997,
            3.0616, 1.6541, 0.0964, -0.8338, -1.6523, -0.8323, -0.1699, 0.0823, 0.3526, 0.6843,
            0.2395, 1.2279, -0.9287, -1.7030, 0.1370, 0.6047, 0.3770, -0.6266, 0.3529, 2.2013,
            -0.6836, 0.2477, 1.3127, -0.2260, 0.2622, -1.2974, -0.8140, -0.8404, -0.3490, 0.0130,
            1.3123, 1.7569, -0.3956, -1.8255, 0.1727, -0.3538, 2.6941, 1.0529, 0.4219, -0.2071,
            1.1586, 0.4717, 0.3865, -0.5690, -0.5010, -0.1310, 0.7796, 0.6630, -0.2021, 2.6090,
            0.2049, 0.6466, -0.5042, -0.0603, -1.6538, -1.2429, 1.8357, 1.6052, -1.3844, 0.3323,
            -1.3712, 0.9634, -0.4799, -0.6451, -0.0840, -1.4247, 0.5512, -0.1747, -0.5509, -0.3742,
            0.3790, -0.4431, -0.4720, -0.7890, 0.2620, 0.5411, -1.1715, -2.4997, 2.3249, -0.8912,
            -0.4733, -0.5701, -2.8888, -1.4112, -0.5471, -0.9234, -1.1660, 0.4189, -0.7465,
            -0.6473, 0.1402, 0.7875, 0.5377, -0.6779, -0.8088, -0.4864, -0.2312, 0.9279, 0.1264,
            1.5480, 0.8265, -0.1025, 0.5138, -0.2512, 0.1576, 1.2705, 0.3641, -0.9325, 0.6451,
            -0.8537, 0.2378, 0.1794, 0.2752, -0.3687, -1.1149, -0.1410, -0.5829, -0.0892, 1.4258,
            -2.2789, 0.5270, 0.1825, 1.7007, -0.5263, -0.2954, 0.4440, 0.5537, 0.3492, 0.6186,
            1.6475, 0.2219,
        ],
        (1, 4, 7, 5),
        dev,
    )?;

    #[rustfmt::skip]
    let w = Var::from_slice(
        &[
            -1.1744_f32, 0.3266, 2.5893, 1.0142, 0.1763, 0.7752, 0.6604, 0.2029, -0.2145, 0.7234,
            -0.3441, -1.5400, -0.6333, 0.6613, 0.2083, 0.6230, -1.7002, 0.3393, 0.4049, 1.0762,
            0.2723, 1.4181, 0.0029, -0.2122, 1.7668, 1.4168, 0.3320, -0.2719, 0.7932, -0.7204,
            0.4447, 0.1211, 0.5908, 1.0089, -0.1646, 1.8033, -0.6286, 0.2016, -0.3370, 1.2555,
            0.8009, -0.6488, -0.4652, -1.5685, 1.5860, 0.5583, 0.4623, 0.6026, 0.8828, 2.4990,
            0.6811, -0.3369, 1.3320, 1.7669, -1.1067, 1.2958, -0.9415, -0.9655, -0.4462, 0.7181,
            0.5181, -1.1658, -1.8467, -0.7763, 1.2769, 0.8651, 0.9890, 1.5092, 0.7207, -0.8481,
            0.7417, 0.3375, -1.2685, 1.4572, 1.0915, 0.1093, -0.8550, -0.5831, -0.6309, -0.2509,
            0.5220, -0.0914, 0.7900, 0.1096, 0.3258, 0.2723, -1.0942, -0.3393, -0.1653, 0.5732,
            -0.8014, 1.8194, -1.9023, 0.2127, 1.8636, -0.8979, 0.1927, -0.2778, 0.3105, 0.0071,
            -1.1823, 0.2476, -0.7178, -1.3821, 1.0769, -0.4376, -0.9967, -0.1227, 1.6197, -1.0604,
            0.1372, 0.8141, -0.6163, 0.7304, -0.8285, 2.0636, -0.7176, 0.2495, -0.2581, -0.4478,
        ],
        (4, 2, 3, 5),
        dev,
    )?;
    let res = t.conv_transpose2d(&w, padding, outpadding, stride, dilation)?;
    let loss = res.sqr()?.sum_all()?;
    assert_eq!(test_utils::to_vec0_round(&loss, 0)?, 2904.0);
    // let grads = loss.backward()?;
    let grads = our_backward(&loss)?;

    let grad_t = grads.get(&t).unwrap();
    let grad_w = grads.get(&w).unwrap();
    assert_eq!(grad_t.dims(), [1, 4, 7, 5]);
    assert_eq!(grad_w.dims(), [4, 2, 3, 5]);

    assert_eq!(
        test_utils::to_vec1_round(&grad_w.flatten_all()?, 1)?,
        [
            // torch gets 89.1
            -89.0, -135.3, 136.7, 102.0, -53.4, 117.9, 118.6, -43.9, -218.0, -58.5, -114.3, -150.0,
            -15.6, 172.1, 66.3, -64.3, -27.9, -19.8, 31.7, 62.1, 5.5, 92.6, 28.2, -29.6, 55.9,
            52.7, -72.7, -119.8, 53.8, -25.5, 128.8, 19.3, 68.0, 190.9, -64.1, -86.2, -111.2,
            106.6, -67.7, 37.8, 115.9, 50.4, -77.7, -54.9, 22.3, -4.6, 89.8, 61.7, 122.4, 192.6,
            -27.8, -104.6, 57.0, 166.4, 27.1, 6.1, 18.7, -93.2, 31.5, 168.2, -3.7, -99.5, -55.5,
            -10.8, 17.5, 20.8, 16.9, 43.8, 42.0, -89.2, 18.8, -9.6, -84.1, 212.6, 19.7, -50.0,
            -52.0, -40.0, -166.6, -73.2, -10.8, -73.3, 31.5, -23.4, -79.3, -27.0, -84.4, -42.9,
            -20.3, 51.8, -16.7, 76.3, -120.5, -65.8, 96.5, -10.7, -45.9, -88.1, 65.4, -7.0, -1.5,
            92.8, -25.1, -114.2, -5.8, -14.8, -51.2, -20.7, 54.2, -79.8, 47.7, -29.2, -8.8, 53.5,
            -28.4, 85.0, -18.3, 107.0, 28.3, -71.8
        ]
    );

    assert_eq!(
        test_utils::to_vec3_round(&grad_t.i(0)?, 1)?,
        [
            [
                [32.3, -41.6, -24.0, 14.1, 17.6],
                [-11.8, 72.5, 87.6, 46.4, 61.5],
                [115.0, 108.5, -48.6, -63.4, -50.0],
                [51.3, 5.4, 31.3, 91.1, -30.9],
                [52.7, 92.8, -68.0, -47.0, 83.0],
                // pytorch gets -107.1
                [-10.2, -107.0, -5.4, 213.1, -31.4],
                [-2.4, 65.1, 9.2, -146.2, -24.2]
            ],
            [
                [-72.6, -63.9, -61.9, 45.3, 33.0],
                [79.3, -0.5, -26.2, 78.2, 42.7],
                [90.9, 141.6, 40.1, -62.7, 37.0],
                [32.8, 198.2, -0.8, -31.1, 27.3],
                // torch gets 48.0
                [34.5, 34.9, -47.9, 127.6, -12.3],
                [-61.4, -3.2, -2.9, -10.9, -16.6],
                [74.6, 60.1, -68.9, 34.5, -50.4]
            ],
            [
                [37.5, -56.9, -43.6, -13.5, -9.9],
                [40.0, 97.3, 28.6, 14.2, -30.1],
                [-22.3, -126.3, -68.8, -8.2, 26.1],
                [-32.9, 37.3, 108.5, -54.8, 29.6],
                [34.9, -176.9, -125.0, -28.3, -13.9],
                [-54.9, 142.6, 62.1, -80.4, -65.6],
                [7.4, -91.1, -67.6, 35.0, 39.7]
            ],
            [
                [-57.2, -40.9, -10.1, 32.6, 29.4],
                [18.7, -18.0, 29.5, -1.2, 59.2],
                [-14.0, -74.4, 19.8, -117.0, 58.2],
                [-21.8, 163.5, -71.1, -99.0, 80.9],
                [-58.9, -10.9, 93.8, -139.6, 98.0],
                // torch gets 54.5
                [-54.4, 135.3, 6.0, -79.1, 134.6],
                [27.5, -76.0, 43.4, -2.8, -7.8]
            ]
        ]
    );


    // Test the same, but then with the following properties, t & w are unmodified.
    let padding = 1;
    let outpadding = 1;
    let dilation = 1;
    let stride = 2;

    let res = t.conv_transpose2d(&w, padding, outpadding, stride, dilation)?;
    let loss = res.sqr()?.sum_all()?;
    // assert_eq!(test_utils::to_vec0_round(&loss, 0)?, 3627.0); // torch gives 3626.8560
    dbg!("Get here");
    // let grads = loss.backward()?;
    let grads = our_backward(&loss)?;
    dbg!("Never get here?");
    let grad_t = grads.get(&t).unwrap();
    let grad_w = grads.get(&w).unwrap();
    assert_eq!(grad_t.dims(), [1, 4, 7, 5]);
    assert_eq!(grad_w.dims(), [4, 2, 3, 5]);


    #[rustfmt::skip]
    assert_eq!(
        test_utils::to_vec3_round(&grad_t.i(0)?, 1)?,
        [
            [
                [  13.2000,  -40.7000,   -9.7000,  -47.3000,  -82.7000],
                [ -98.2000,    9.7000,   57.7000,   -6.2000,  180.7000],
                [ 100.2000,   24.1000,    3.7000, -100.5000,  -48.1000],
                [  -0.3000,   13.5000,   -2.9000,   80.0000,  -49.8000],
                [  47.2000,  -25.6000,  -74.4000,   61.2000,  -18.4000],
                [   4.6000,  -69.5000,   27.9000,   66.5000,  -88.1000],
                 // 4th column on next row; torch is 4.2
                [ -12.0000,   79.2000,  -40.0000,    4.1000,  -97.1000],
            ],
            [
                [ -42.2000,  -36.5000,  -51.1000,    7.5000,   32.3000],
                [  74.1000,  -44.6000,  -68.8000,   19.5000,    7.7000],
                [ 137.1000,   54.2000,  153.8000,  -58.0000,   45.5000],
                [  24.4000,  -56.8000,    9.7000,  -41.0000,  -14.5000],
                [  -3.7000,   72.6000,    8.3000,  134.8000,   40.5000],
                [  43.2000,  -56.9000,  -47.5000,  -89.4000,  -95.4000],
                [  68.2000,  108.1000,  -80.0000,   57.0000, -121.1000]
            ],
            [
                [  31.1000,  -11.4000,  -34.8000,   33.1000,  -44.2000],
                [  29.4000,  -31.6000,  -40.2000,   13.7000,   13.1000],
                [  -0.8000,  -83.8000,   -7.8000,  -17.3000,   78.2000],
                [  12.0000, -118.7000,  137.5000,  -76.7000,   50.8000],
                [ -28.7000, -114.2000,   -3.7000,  -96.3000,  -13.8000],
                [ -31.8000,   28.5000,  -14.3000,    4.6000,   13.4000],
                [  28.0000,   -0.2000,  -38.9000,  -29.7000,  -59.0000]
            ],
            [
                [ -16.8000,   38.5000,   15.5000,   26.6000,   48.9000],
                [  14.5000,   49.6000,  -24.8000,   65.6000,   61.7000],
                [  22.1000,  -64.7000,   -4.3000,  -51.0000,   36.3000],
                [  31.0000,  -88.9000,   47.1000, -123.5000,   -3.8000],
                [ -14.8000,  -39.8000,  128.2000, -110.3000,   42.6000],
                // 1st column on next row; torch is -7.2
                [  -7.1000,   95.3000,  -21.3000,  -58.7000,  -13.9000], 
                [  26.9000,   21.3000,   16.1000,   70.3000,   32.1000]
            ]
        ]
    );

    #[rustfmt::skip]
    assert_eq!(
        test_utils::to_vec1_round(&grad_w.flatten_all()?, 1)?,
        [
            // 2nd value; torch gets -3.2, 3rd value; torch gets 221.8
           -2.460e+01, -3.100e+00,  2.219e+02,  7.400e+00,  5.620e+01,
            7.420e+01,  7.830e+01,  8.900e+00,  1.050e+01,  2.810e+01,
            5.100e+00, -1.046e+02, -1.572e+02,  8.710e+01, -9.840e+01,
           -4.230e+01, -1.898e+02,  1.860e+01, -3.570e+01,  9.810e+01,
            4.680e+01,  1.182e+02,  4.020e+01, -1.900e+00,  1.508e+02,
            1.094e+02,  1.018e+02, -4.620e+01,  1.591e+02, -2.320e+01,
            // 5th value; torch gets 7.1
           -8.450e+01, -4.600e+00,  6.330e+01,  1.123e+02, -7.000e+00,
            1.101e+02, -6.620e+01,  2.090e+01, -5.120e+01,  8.990e+01,
            9.050e+01, -6.990e+01,  6.800e+01, -9.250e+01,  1.380e+02,
            4.720e+01,  4.710e+01,  6.210e+01,  8.870e+01,  2.098e+02,
            3.870e+01, -1.390e+01,  6.270e+01,  1.484e+02, -9.920e+01,
           -4.200e+01, -1.505e+02, -1.480e+01, -2.620e+01,  8.220e+01,
           -3.350e+01, -2.260e+01, -1.198e+02, -5.080e+01,  1.259e+02,
            5.600e+01,  9.270e+01,  1.209e+02,  6.590e+01, -8.330e+01,
            7.000e+00, -2.600e+01, -1.133e+02,  3.870e+01,  4.020e+01,
           -6.300e+00, -8.710e+01, -5.150e+01, -8.510e+01,  2.000e-01,
            3.640e+01, -6.100e+00,  6.590e+01, -2.700e+00,  6.550e+01,
            // 4th value; torch gets 3.8
            5.300e+00, -6.760e+01, -4.270e+01, -3.900e+00,  2.880e+01,
            5.260e+01,  6.170e+01, -1.203e+02, -1.610e+01,  7.740e+01,
           -1.008e+02, -1.070e+01, -9.900e+00,  3.300e+00, -2.620e+01,
           -4.440e+01,  2.580e+01, -6.920e+01, -4.220e+01,  1.108e+02,
            1.240e+01, -3.440e+01, -2.800e+00,  7.880e+01, -6.690e+01,
            1.480e+01,  2.310e+01, -4.260e+01, -1.500e+00, -4.760e+01,
            5.350e+01, -2.260e+01,  8.000e-01, -3.840e+01, -2.500e+00
        ]
    );
    dbg!();
    Ok(())
}
