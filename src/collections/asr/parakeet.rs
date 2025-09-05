mod attention;
mod conformer;
mod decoder;
mod encoder;
mod greedy_search;
mod joiner;
mod preprocessor;
mod subsampling;
mod tokenizer;

use candle::Result;
use candle_nn::VarBuilder;

use crate::collections::asr::parakeet::{
    encoder::ConformerEncoder, greedy_search::GreedyTdtDecoder,
    preprocessor::AudioToMelSpectrogramPreprocessor, tokenizer::Tokenizer,
};

pub struct Parakeet {
    preprocessor: AudioToMelSpectrogramPreprocessor,
    encoder: ConformerEncoder,
    decoder: GreedyTdtDecoder,
    tokenizer: Tokenizer,
}

impl Parakeet {
    pub fn new(vb: VarBuilder, sample_rate: u32) -> Result<Self> {
        let preprocessor = AudioToMelSpectrogramPreprocessor::new(sample_rate, vb.device())?;
        let encoder = ConformerEncoder::new(vb.pp("encoder"))?;
        let decoder = GreedyTdtDecoder::new(vb)?;
        let tokenizer = Tokenizer::new()?;

        Ok(Self {
            preprocessor,
            encoder,
            decoder,
            tokenizer,
        })
    }

    pub fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
        let x = self.preprocessor.forward(audio)?;
        let x = self.encoder.forward(&x)?;
        let ids: Vec<u32> = self.decoder.forward(&x)?;
        let text = self.tokenizer.ids_to_text(&ids);

        Ok(text)
    }
}
