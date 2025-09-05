use candle::{Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::collections::asr::parakeet::{
    attention::RelPositionEncoding, conformer::ConformerLayer, subsampling::ConvSubsampling,
};

pub struct ConformerEncoder {
    pre_encode: ConvSubsampling,
    pos_encode: RelPositionEncoding,
    layers: Vec<ConformerLayer>,
}

impl ConformerEncoder {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let pre_encode = ConvSubsampling::new(vb.pp("pre_encode"))?;

        let pos_emb_max_len = 5000;
        let pos_encode = RelPositionEncoding::new(pos_emb_max_len, vb.device())?;

        let n_layers = 24;
        let d_model = 1024;
        let d_ff = 4096;
        let n_heads = 8;
        let conv_kernel_size = 9;
        let conv_context_size = (4, 4);
        let layers = (0..n_layers)
            .map(|i| {
                ConformerLayer::new(
                    d_model,
                    d_ff,
                    n_heads,
                    conv_kernel_size,
                    conv_context_size,
                    vb.pp(format!("layers.{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            pre_encode,
            pos_encode,
            layers,
        })
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.pre_encode.forward(x)?;
        let pos_emb = self.pos_encode.forward(&x)?;

        for layer in &self.layers {
            x = layer.forward(&x, &pos_emb)?;
        }

        Ok(x)
    }
}
