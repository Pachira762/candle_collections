use candle::{D, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;

use crate::collections::{
    asr::reazonspeech::{decoder::Decoder, joiner::Joiner},
    common::tensor_ext::TensorExt,
};

pub struct GreedySearchInfer {
    decoder: Decoder,
    joiner: Joiner,
}

impl GreedySearchInfer {
    const BLANK: u32 = 0;
    const UNK: u32 = 5222;

    pub fn new(vb: VarBuilder) -> Result<Self> {
        let decoder_dim = 512;
        let joiner_dim = 512;
        let vocab_size = 5224;
        let context_size = 2;
        let decoder = Decoder::new(
            vocab_size,
            decoder_dim,
            context_size,
            joiner_dim,
            vb.pp("decoder"),
        )?;
        let joiner = Joiner::new(joiner_dim, vocab_size, vb.pp("joiner"))?;

        Ok(Self { decoder, joiner })
    }

    pub fn infer(&mut self, encoder_out: &Tensor) -> Result<Vec<u32>> {
        let device = encoder_out.device();
        let mut context = vec![0, 0];
        let mut tokens = vec![];
        let mut time_index = 0;
        let max_time = encoder_out.size(1);
        let mut cur_decoder_out = self.decode(&context, device)?;

        while time_index < max_time {
            let cur_encoder_out = encoder_out.narrow(1, time_index, 1)?;

            for _ in 0..4 {
                let logits = self
                    .joiner
                    .forward(&cur_encoder_out, &cur_decoder_out)?
                    .i((0, 0))?
                    .softmax(D::Minus1)?;
                let token = logits.argmax(D::Minus1)?.to_scalar::<u32>()? as u32;

                if token != Self::BLANK && token < Self::UNK {
                    tokens.push(token);
                    context.copy_within(1.., 0);
                    *context.last_mut().unwrap() = token;

                    cur_decoder_out = self.decode(&context, device)?;
                    continue;
                }

                break;
            }

            time_index += 1;
        }

        Ok(tokens)
    }

    fn decode(&self, context: &[u32], device: &Device) -> Result<Tensor> {
        let input = Tensor::new(context, device)?.unsqueeze(0)?;
        let output = self.decoder.forward(&input)?;

        Ok(output)
    }
}
