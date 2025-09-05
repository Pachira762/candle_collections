use candle::{
    CudaDevice, Module, Result, Tensor, WithDType, backend::BackendStorage, conv::CudnnFwdAlgo,
    cuda::CudaStorageSlice,
};
use candle_nn::VarBuilder;
use cudarc::{
    cudnn::{
        ConvForward, CudnnDataType,
        sys::{cudnnConvolutionFwdAlgo_t, cudnnConvolutionMode_t, cudnnTensorFormat_t},
    },
    driver::{CudaSlice, CudaView},
};

use crate::collections::common::cudnn;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv1dConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
    pub bias: bool,
    pub cudnn_fwd_algo: Option<CudnnFwdAlgo>,
}

impl Conv1dConfig {
    pub fn out_size(&self, in_size: usize) -> usize {
        (in_size + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1
    }
}

impl Default for Conv1dConfig {
    fn default() -> Self {
        Self {
            in_channels: 0,
            out_channels: 0,
            kernel_size: 0,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: true,
            cudnn_fwd_algo: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Conv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    config: Conv1dConfig,
}

impl Conv1d {
    pub fn new(config: Conv1dConfig, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(
            (
                config.out_channels,
                config.in_channels / config.groups,
                config.kernel_size,
            ),
            "weight",
        )?;
        let bias = if config.bias {
            Some(vb.get(config.out_channels, "bias")?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            config,
        })
    }
}

impl Module for Conv1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, _in_channels, in_width) = x.dims3()?;
        let out_width = self.config.out_size(in_width);

        let op = Conv2dOp {
            batch: batch as _,
            in_channels: self.config.in_channels as _,
            out_channels: self.config.out_channels as _,
            in_height: 1,
            in_width: in_width as _,
            kernel_height: 1 as _,
            kernel_width: self.config.kernel_size as _,
            out_height: 1,
            out_width: out_width as _,
            padding: [0, self.config.padding as _],
            stride: [1, self.config.stride as _],
            dilation: [1, self.config.dilation as _],
            groups: self.config.groups as _,
            cudnn_fwd_algo: self.config.cudnn_fwd_algo,
        };

        let x = if x.is_contiguous() {
            x.clone()
        } else {
            x.contiguous()?
        };
        let x = x
            .apply_op2(&self.weight, op)?
            .reshape((batch, self.config.out_channels, ()))?;

        match &self.bias {
            None => Ok(x),
            Some(bias) => {
                let channels = bias.dims1()?;
                let bias = bias.reshape((1, channels, 1))?;
                Ok(x.broadcast_add(&bias)?)
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Conv2dConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
    pub groups: usize,
    pub bias: bool,
    pub cudnn_fwd_algo: Option<CudnnFwdAlgo>,
}

impl Conv2dConfig {
    pub fn out_size(&self, in_size: (usize, usize)) -> (usize, usize) {
        let out_height =
            (in_size.0 + 2 * self.padding.0 - self.dilation.0 * (self.kernel_size.0 - 1) - 1)
                / self.stride.0
                + 1;
        let out_width =
            (in_size.1 + 2 * self.padding.1 - self.dilation.1 * (self.kernel_size.1 - 1) - 1)
                / self.stride.1
                + 1;
        (out_height, out_width)
    }
}

impl Default for Conv2dConfig {
    fn default() -> Self {
        Self {
            in_channels: 0,
            out_channels: 0,
            kernel_size: (0, 0),
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
            bias: true,
            cudnn_fwd_algo: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    config: Conv2dConfig,
}

impl Conv2d {
    pub fn new(config: Conv2dConfig, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(
            (
                config.out_channels,
                config.in_channels / config.groups,
                config.kernel_size.0,
                config.kernel_size.1,
            ),
            "weight",
        )?;
        let bias = if config.bias {
            Some(vb.get(config.out_channels, "bias")?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            config,
        })
    }
}

impl Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, _in_channels, in_height, in_width) = x.dims4()?;
        let (out_height, out_width) = self.config.out_size((in_height, in_width));

        let op = Conv2dOp {
            batch: batch as _,
            in_channels: self.config.in_channels as _,
            out_channels: self.config.out_channels as _,
            in_height: in_height as _,
            in_width: in_width as _,
            kernel_height: self.config.kernel_size.0 as _,
            kernel_width: self.config.kernel_size.1 as _,
            out_height: out_height as _,
            out_width: out_width as _,
            padding: [self.config.padding.0 as _, self.config.padding.1 as _],
            stride: [self.config.stride.0 as _, self.config.stride.1 as _],
            dilation: [self.config.dilation.0 as _, self.config.dilation.1 as _],
            groups: self.config.groups as _,
            cudnn_fwd_algo: self.config.cudnn_fwd_algo,
        };

        let x = if x.is_contiguous() {
            x.clone()
        } else {
            x.contiguous()?
        };
        let x = x.apply_op2(&self.weight, op)?;

        match &self.bias {
            None => Ok(x),
            Some(bias) => {
                let channels = bias.dims1()?;
                let bias = bias.reshape((1, channels, 1, 1))?;
                Ok(x.broadcast_add(&bias)?)
            }
        }
    }
}

struct Conv2dOp {
    batch: i32,
    in_channels: i32,
    out_channels: i32,
    in_height: i32,
    in_width: i32,
    kernel_height: i32,
    kernel_width: i32,
    out_height: i32,
    out_width: i32,
    padding: [i32; 2],
    stride: [i32; 2],
    dilation: [i32; 2],
    groups: i32,
    cudnn_fwd_algo: Option<CudnnFwdAlgo>,
}

impl Conv2dOp {
    fn launch_conv2d<T: CudnnDataType + WithDType>(
        &self,
        device: &CudaDevice,
        src: &CudaView<T>,
        filter: &CudaView<T>,
        dst: &mut CudaSlice<T>,
    ) -> Result<()> {
        let cudnn = cudnn::get(device)?;

        let desc = cudnn.create_conv2d::<T>(
            self.padding,
            self.stride,
            self.dilation,
            cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
        )?;

        let batch = self.batch;
        let x = cudnn.create_4d_tensor::<T>(
            cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            [batch, self.in_channels, self.in_height, self.in_width],
        )?;
        let w = cudnn.create_4d_filter::<T>(
            cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            [
                self.out_channels,
                self.in_channels / self.groups,
                self.kernel_height,
                self.kernel_width,
            ],
        )?;
        let y = cudnn.create_4d_tensor::<T>(
            cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            [batch, self.out_channels, self.out_height, self.out_width],
        )?;

        let conv = ConvForward {
            conv: &desc,
            x: &x,
            w: &w,
            y: &y,
        };

        let alg = match self.cudnn_fwd_algo {
            None => conv.pick_algorithm()?,
            Some(CudnnFwdAlgo::ImplicitGemm) => {
                cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
            }
            Some(CudnnFwdAlgo::ImplicitPrecompGemm) => {
                cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
            }
            Some(CudnnFwdAlgo::Gemm) => cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
            Some(CudnnFwdAlgo::Direct) => {
                cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT
            }
            Some(CudnnFwdAlgo::Fft) => cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_FFT,
            Some(CudnnFwdAlgo::FftTiling) => {
                cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING
            }
            Some(CudnnFwdAlgo::Winograd) => {
                cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD
            }
            Some(CudnnFwdAlgo::WinogradNonFused) => {
                cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
            }
            Some(CudnnFwdAlgo::Count) => {
                cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_COUNT
            }
        };

        let workspace_size = conv.get_workspace_size(alg)?;
        let mut workspace = unsafe { device.alloc::<u8>(workspace_size)? };

        unsafe {
            conv.launch::<CudaSlice<u8>, _, _, _>(
                alg,
                Some(&mut workspace),
                (T::one(), T::zero()),
                src,
                filter,
                dst,
            )?;
        }
        Ok(())
    }
}

impl candle::CustomOp2 for Conv2dOp {
    fn name(&self) -> &'static str {
        "cudnn-conv2d"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        unimplemented!()
    }

    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        _l1: &candle::Layout,
        s2: &candle::CudaStorage,
        _l2: &candle::Layout,
    ) -> candle::Result<(candle::CudaStorage, candle::Shape)> {
        let device = s1.device().clone();

        let mut out = unsafe {
            device.alloc::<f32>(
                self.batch as usize
                    * self.out_channels as usize
                    * self.out_height as usize
                    * self.out_width as usize,
            )
        }?;

        self.launch_conv2d(
            &device,
            &s1.as_cuda_slice()?.as_view(),
            &s2.as_cuda_slice()?.as_view(),
            &mut out,
        )?;
        let slice = CudaStorageSlice::F32(out);

        Ok((
            candle::CudaStorage { slice, device },
            (
                self.batch as usize,
                self.out_channels as usize,
                self.out_height as usize,
                self.out_width as usize,
            )
                .into(),
        ))
    }
}
