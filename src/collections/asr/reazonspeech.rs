mod decoder;
mod encoder;
mod greedy_search;
mod joiner;
mod preprocessor;
mod scaling;
mod subsampling;
mod tokenizer;
mod zipformer;

use candle::Result;
use candle_nn::VarBuilder;

use crate::collections::asr::reazonspeech::{
    encoder::Encoder, greedy_search::GreedySearchInfer, preprocessor::FeatureExtractor,
    tokenizer::Tokenizer,
};

pub struct ReazonSpeech {
    preprocessor: FeatureExtractor,
    encoder: Encoder,
    decoder: GreedySearchInfer,
    tokenizer: Tokenizer,
}

impl ReazonSpeech {
    pub fn new(vb: VarBuilder, sample_rate: u32) -> Result<Self> {
        let preprocessor = FeatureExtractor::new(sample_rate, vb.device())?;
        let encoder = Encoder::new(512, vb.pp("encoder"))?;
        let decoder = GreedySearchInfer::new(vb)?;
        let tokenizer = Tokenizer::new()?;

        Ok(Self {
            preprocessor,
            encoder,
            decoder,
            tokenizer,
        })
    }

    pub fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
        let x = self.preprocessor.process(audio)?;
        let x = self.encoder.forward(&x)?;
        let ids = self.decoder.infer(&x)?;
        let text = self.tokenizer.ids_to_text(&ids);

        Ok(text)
    }
}
