#![allow(unused)]

use std::{cell::LazyCell, collections::HashMap, fmt::Debug, io::Cursor, path::Path};

use candle::{DType, Device, Result, Shape, Tensor};

#[allow(unused)]
pub fn load_pcm<P: AsRef<Path>>(filename: P) -> Result<Vec<f32>> {
    let data = std::fs::read(filename)?;
    let n_samples = data.len() / 4;
    let mut pcm = vec![0.0; n_samples];
    pcm.copy_from_slice(unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const f32, n_samples)
    });
    Ok(pcm)
}

pub fn mse(x: &candle::Tensor, y: &candle::Tensor) -> f32 {
    if x.dtype() != candle::DType::F32 {
        return 0.0;
    }

    candle_nn::loss::mse(x, y)
        .expect("failed loss::mse")
        .to_scalar()
        .unwrap()
}

pub trait AllClose {
    fn all_close(&self, rhs: &Tensor, label: &str) -> candle::Result<()>;
}

impl AllClose for candle::Tensor {
    fn all_close(&self, rhs: &Tensor, label: &str) -> candle::Result<()> {
        all_close(label, self, rhs)
    }
}

pub fn all_close(label: &str, result: &Tensor, expected: &Tensor) -> candle::Result<()> {
    assert_eq!(result.shape(), expected.shape());
    assert_eq!(result.dtype(), expected.dtype());

    println!(
        "{} Shape:{:?}, MSE: {}",
        label,
        result.shape(),
        mse(result, expected)
    );

    let vec1: Vec<f32> = result.flatten_all()?.to_vec1()?;
    let vec2: Vec<f32> = expected.flatten_all()?.to_vec1()?;
    if vec1.len() < 10 {
        for (v1, v2) in vec1.iter().zip(vec2.iter()) {
            println!("\t{} / {}", v1, v2);
        }
    } else {
        let len = vec1.len();
        let mid = len / 2;
        println!("\t{} / {}", vec1[0], vec2[0]);
        println!("\t{} / {}", vec1[1], vec2[1]);
        println!("\t{} / {}", vec1[2], vec2[2]);
        println!("\t...");
        println!("\t{} / {}", vec1[mid - 1], vec2[mid - 1]);
        println!("\t{} / {}", vec1[mid], vec2[mid]);
        println!("\t{} / {}", vec1[mid + 1], vec2[mid + 1]);
        println!("\t...");
        println!("\t{} / {}", vec1[len - 3], vec2[len - 3]);
        println!("\t{} / {}", vec1[len - 2], vec2[len - 2]);
        println!("\t{} / {}", vec1[len - 1], vec2[len - 1]);
    }

    Ok(())
}
