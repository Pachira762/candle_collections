mod collections;
pub mod stats;
pub mod utils;

use candle::{DType, Device, Result};
use candle_nn::VarBuilder;

use crate::{
    collections::asr::{parakeet::Parakeet, reazonspeech::ReazonSpeech},
    utils::load_pcm,
};

fn main() -> Result<()> {
    test_parakeet()?;
    test_reazonspeech()?;

    println!("Done!");
    Ok(())
}

#[allow(unused)]
fn test_parakeet() -> Result<()> {
    let device = Device::new_cuda(0)?;

    let vb = {
        let tensors =
            candle::safetensors::load("export\\models\\parakeet-tdt-0.6b-v2.safetensors", &device)?;
        VarBuilder::from_tensors(tensors, DType::F32, &device)
    };

    let audio = load_pcm("data\\sample-en.pcm")?;
    let mut model = Parakeet::new(vb, 16000)?;

    let t0 = std::time::Instant::now();
    let text = model.transcribe(&audio)?;
    let t1 = std::time::Instant::now();
    println!("{}\ntook {:.1?}", text, t1 - t0);

    Ok(())
}

#[allow(unused)]
fn test_reazonspeech() -> Result<()> {
    let device = Device::new_cuda(0)?;

    let vb = {
        let tensors =
            candle::safetensors::load("export\\models\\reazonspeech-k2-v2.safetensors", &device)?;
        VarBuilder::from_tensors(tensors, DType::F32, &device)
    };

    let audio = load_pcm("data\\sample-ja.pcm")?;
    let mut model = ReazonSpeech::new(vb, 16000)?;

    let t0 = std::time::Instant::now();
    let text = model.transcribe(&audio)?;
    let t1 = std::time::Instant::now();
    println!("{}\ntook {:.1?}", text, t1 - t0);

    Ok(())
}
