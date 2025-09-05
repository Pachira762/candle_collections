use crate::collections::common::conv::{Conv2d, Conv2dConfig};
use candle::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

pub struct ConvSubsampling {
    conv_0: Conv2d,
    conv_2: Conv2d,
    conv_3: Conv2d,
    conv_5: Conv2d,
    conv_6: Conv2d,
    out: Linear,
}

impl ConvSubsampling {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let config = Conv2dConfig {
            in_channels: 1,
            out_channels: 256,
            kernel_size: (3, 3),
            stride: (2, 2),
            padding: (1, 1),
            ..Default::default()
        };
        let conv_0 = Conv2d::new(config, vb.pp("conv.0"))?;

        let config = Conv2dConfig {
            in_channels: 256,
            out_channels: 256,
            kernel_size: (3, 3),
            stride: (2, 2),
            padding: (1, 1),
            groups: 256,
            ..Default::default()
        };
        let conv_2 = Conv2d::new(config, vb.pp("conv.2"))?;

        let config = Conv2dConfig {
            in_channels: 256,
            out_channels: 256,
            kernel_size: (1, 1),
            ..Default::default()
        };
        let conv_3 = Conv2d::new(config, vb.pp("conv.3"))?;

        let config = Conv2dConfig {
            in_channels: 256,
            out_channels: 256,
            kernel_size: (3, 3),
            stride: (2, 2),
            padding: (1, 1),
            groups: 256,
            ..Default::default()
        };
        let conv_5 = Conv2d::new(config, vb.pp("conv.5"))?;

        let config = Conv2dConfig {
            in_channels: 256,
            out_channels: 256,
            kernel_size: (1, 1),
            ..Default::default()
        };
        let conv_6 = Conv2d::new(config, vb.pp("conv.6"))?;

        let out = candle_nn::linear(4096, 1024, vb.pp("out"))?;

        Ok(Self {
            conv_0,
            conv_2,
            conv_3,
            conv_5,
            conv_6,
            out,
        })
    }

    /// \[B, T, F\] => \[B, T/8, H\]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.unsqueeze(1)?;
        let x = self.conv_0.forward(&x)?;
        let x = x.relu()?;
        let x = self.conv_2.forward(&x)?;
        let x = self.conv_3.forward(&x)?;
        let x = x.relu()?;
        let x = self.conv_5.forward(&x)?;
        let x = self.conv_6.forward(&x)?;
        let x = x.relu()?;

        let (b, _c, t, _f) = x.dims4()?;
        let x = x.transpose(1, 2)?.reshape((b, t, ()))?;
        let x = self.out.forward(&x)?;

        Ok(x)
    }
}
