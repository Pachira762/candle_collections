use crate::collections::{
    asr::parakeet::{decoder::RnntDecoder, joiner::RnntJoiner},
    common::tensor_ext::TensorExt,
};
use candle::{IndexOp, Result, Tensor};
use candle_nn::VarBuilder;

pub struct GreedyTdtDecoder {
    decoder: RnntDecoder,
    joint: RnntJoiner,
}

impl GreedyTdtDecoder {
    const BLANK: u32 = 1024;
    const DURATIONS: [usize; 5] = [0, 1, 2, 3, 4];

    pub fn new(vb: VarBuilder) -> Result<Self> {
        let decoder = RnntDecoder::new(vb.pp("decoder"))?;
        let joint = RnntJoiner::new(vb.pp("joint"))?;

        Ok(Self { decoder, joint })
    }

    pub fn forward(&mut self, encoder_out: &Tensor) -> Result<Vec<u32>> {
        let x = encoder_out.get(0)?.unsqueeze(1)?;
        let max_t = x.size(0);

        let mut tokens: Vec<u32> = vec![];
        let mut last_token = Self::BLANK;
        let mut state = self.decoder.initial_state()?;
        let mut time_index = 0;

        while time_index < max_t {
            let f = x.narrow(0, time_index, 1)?;
            let mut skip = 0;

            for _ in 0..16 {
                let g = self.decoder.predict(last_token, &mut state)?;

                let logits = self.joint.joint(&f, &g)?.i((0, 0, 0))?;
                let n_logits: usize = logits.size(0);
                let n_durations = Self::DURATIONS.len();
                let n_vocab = n_logits - n_durations;

                let logp = logits.i(..n_vocab)?.softmax(0)?;
                let token = logp.argmax(0)?.to_scalar::<u32>()?;

                let duration_logp = logits.i(n_vocab..)?;
                let duration = duration_logp.argmax(0)?.to_scalar::<u32>()?;
                skip = Self::DURATIONS[duration as usize];

                if token == Self::BLANK && skip == 0 {
                    skip = 1;
                }

                if token != Self::BLANK {
                    tokens.push(token);
                    last_token = token;
                }

                time_index += skip;

                if skip > 0 {
                    break;
                }
            }

            if skip == 0 {
                time_index += 1;
            }
        }

        Ok(tokens)
    }
}
