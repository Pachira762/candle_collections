use core::f32;

use candle::{Device, Result, Tensor};

use crate::collections::asr::common::mel::{MelFilterType, MelProcessor, WindowFunc};

pub struct FeatureExtractor {
    device: Device,
    hop_len: usize,
    window_len: usize,
    n_mels: usize,
    mel_processor: MelProcessor,
}

impl FeatureExtractor {
    pub fn new(sample_rate: u32, device: &Device) -> Result<Self> {
        let hop_len = 160 * sample_rate as usize / 16000;
        let window_len = 400 * sample_rate as usize / 16000;
        let n_fft = if sample_rate == 16000 {
            window_len.next_power_of_two()
        } else {
            window_len
        };
        let n_mels = 80;
        let mel_processor = MelProcessor::new(
            0.97,
            WindowFunc::Povey,
            window_len,
            n_fft,
            MelFilterType::Kaldi,
            n_mels,
            sample_rate,
            20.0,
            8000.0 - 400.0,
        )?;

        Ok(Self {
            device: device.clone(),
            hop_len,
            window_len,
            n_mels,
            mel_processor,
        })
    }

    pub fn process(&mut self, audio: &[f32]) -> Result<Tensor> {
        assert!(audio.len() >= self.window_len);

        let n_frames = (audio.len() - self.window_len) / self.hop_len + 1;
        let n_mels = self.n_mels;
        let mut frames = vec![0.0f32; n_frames * n_mels];

        for i in 0..n_frames {
            let start = i * self.hop_len;
            let end = start + self.window_len;
            let audio = &audio[start..end];
            let mels = self.mel_processor.process(audio);

            for j in 0..n_mels {
                frames[i * n_mels + j] = mels[j].max(f32::EPSILON);
            }
        }

        let x = Tensor::from_slice(&frames, (1, n_frames, n_mels), &self.device)?;
        let x = x.log()?;

        Ok(x)
    }
}
