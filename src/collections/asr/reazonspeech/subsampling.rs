use crate::collections::{
    asr::reazonspeech::scaling::{BiasNorm, SwooshL, SwooshR},
    common::conv::{Conv2d, Conv2dConfig},
};
use candle::{Module, Result, Tensor};
use candle_nn::{Linear, Sequential, VarBuilder};

pub struct Conv2dSubsampling {
    conv: Sequential,
    convnext: ConvNeXt,
    out: Linear,
    out_norm: BiasNorm,
}

impl Conv2dSubsampling {
    pub fn new(in_channels: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        let layer1_channels = 8;
        let layer2_channels = 32;
        let layer3_channels = 128;

        let config = Conv2dConfig {
            in_channels: 1,
            out_channels: layer1_channels,
            kernel_size: (3, 3),
            padding: (0, 1),
            ..Default::default()
        };
        let conv0 = Conv2d::new(config, vb.pp("conv.0"))?;

        let config = Conv2dConfig {
            in_channels: layer1_channels,
            out_channels: layer2_channels,
            kernel_size: (3, 3),
            stride: (2, 2),
            ..Default::default()
        };
        let conv4 = Conv2d::new(config, vb.pp("conv.4"))?;

        let config = Conv2dConfig {
            in_channels: layer2_channels,
            out_channels: layer3_channels,
            kernel_size: (3, 3),
            stride: (1, 2),
            ..Default::default()
        };
        let conv7 = Conv2d::new(config, vb.pp("conv.7"))?;

        let conv = candle_nn::seq()
            .add(conv0)
            .add(SwooshR::new()?)
            .add(conv4)
            .add(SwooshR::new()?)
            .add(conv7)
            .add(SwooshR::new()?);

        let convnext = ConvNeXt::new(layer3_channels, (7, 7), vb.pp("convnext"))?;

        let out_width = (((in_channels - 1) / 2) - 1) / 2;
        let out = candle_nn::linear(out_width * layer3_channels, out_channels, vb.pp("out"))?;

        let out_norm = BiasNorm::new(out_channels, vb.pp("out_norm"))?;

        Ok(Self {
            conv,
            convnext,
            out,
            out_norm,
        })
    }

    /// \[B, T, F\] => \[B, C, T, H\]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.unsqueeze(1)?;
        let x = self.conv.forward(&x)?;
        let x = self.convnext.forward(&x)?;

        let (b, c, t, f) = x.dims4()?;
        let x = x.transpose(1, 2)?.reshape((b, t, c * f))?;
        let x = self.out.forward(&x)?;
        let x = self.out_norm.forward(&x)?;

        Ok(x)
    }
}

#[derive(Clone, Debug)]
struct ConvNeXt {
    depthwise_conv: Conv2d,
    pointwise_conv1: Conv2d,
    activation: SwooshL,
    pointwise_conv2: Conv2d,
}

impl ConvNeXt {
    fn new(channels: usize, kernel_size: (usize, usize), vb: VarBuilder) -> Result<Self> {
        let hidden_ratio = 3;
        let hidden_channels = channels * hidden_ratio;

        let padding = ((kernel_size.0 - 1) / 2, (kernel_size.1 - 1) / 2);
        let config = Conv2dConfig {
            in_channels: channels,
            out_channels: channels,
            kernel_size,
            padding,
            groups: channels,
            ..Default::default()
        };
        let depthwise_conv = Conv2d::new(config, vb.pp("depthwise_conv"))?;

        let config = Conv2dConfig {
            in_channels: channels,
            out_channels: hidden_channels,
            kernel_size: (1, 1),
            ..Default::default()
        };
        let pointwise_conv1 = Conv2d::new(config, vb.pp("pointwise_conv1"))?;

        let activation = SwooshL::new()?;

        let config = Conv2dConfig {
            in_channels: hidden_channels,
            out_channels: channels,
            kernel_size: (1, 1),
            ..Default::default()
        };
        let pointwise_conv2 = Conv2d::new(config, vb.pp("pointwise_conv2"))?;

        Ok(Self {
            depthwise_conv,
            pointwise_conv1,
            activation,
            pointwise_conv2,
        })
    }
}

impl candle::Module for ConvNeXt {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let bypass = x.clone();
        let x = self.depthwise_conv.forward(x)?;
        let x = self.pointwise_conv1.forward(&x)?;
        let x = self.activation.forward(&x)?;
        let x = self.pointwise_conv2.forward(&x)?;
        let x = (x + &bypass)?;

        Ok(x)
    }
}
