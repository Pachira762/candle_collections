#![allow(unused)]

use std::{
    cell::RefCell,
    rc::Rc,
    time::{Duration, Instant},
};

use candle::{Device, backend::BackendDevice};

thread_local! {
    static STATS: RefCell<Stats_> = RefCell::new(Stats_::new());
}

pub struct Stats {}

impl Stats {
    const DUMMY: bool = false;

    pub fn init(device: &Device) {
        if !Self::DUMMY {
            STATS.with_borrow_mut(|stats| stats.set_device(device));
        }
    }

    pub fn start(label: &str) {
        if !Self::DUMMY {
            STATS.with_borrow_mut(|stats| stats.start(label));
        }
    }

    pub fn stop() {
        if !Self::DUMMY {
            STATS.with_borrow_mut(|stats| stats.stop());
        }
    }

    pub fn stop_and(label: &str) {
        if !Self::DUMMY {
            STATS.with_borrow_mut(|stats| stats.stop_and(label));
        }
    }

    pub fn dump() {
        if !Self::DUMMY {
            STATS.with_borrow_mut(|stats| stats.dump());
        }
    }
}

struct Record {
    label: String,
    dts: Vec<Duration>,
    depth: usize,
}

struct Stats_ {
    record: Vec<Record>,
    stack: Vec<(String, Instant)>,
    device: Option<Device>,
}

impl Stats_ {
    fn new() -> Self {
        Self {
            record: vec![],
            stack: vec![],
            device: None,
        }
    }

    fn set_device(&mut self, device: &Device) {
        self.device = Some(device.clone());
    }

    fn start(&mut self, label: &str) {
        let label = if let Some((parent, _)) = self.stack.last() {
            [parent, label].join(".")
        } else {
            label.to_string()
        };

        if self.record.iter().all(|rec| rec.label != label) {
            self.record.push(Record {
                label: label.clone(),
                dts: vec![],
                depth: self.stack.len(),
            });
        }

        self.stack.push((label, Instant::now()));
    }

    fn stop(&mut self) {
        if let Some(Device::Cuda(device)) = &self.device {
            device.synchronize().unwrap();
        }

        let (label, t0) = self.stack.pop().unwrap();
        let dt = Instant::now() - t0;

        for rec in self.record.iter_mut() {
            if rec.label == label {
                rec.dts.push(dt);
                return;
            }
        }

        self.record.push(Record {
            label,
            dts: vec![dt],
            depth: self.stack.len(),
        })
    }

    fn stop_and(&mut self, label: &str) {
        self.stop();
        self.start(label);
    }

    fn dump(&mut self) {
        for Record { label, dts, depth } in &self.record {
            let count = dts.len();
            if count == 0 {
                println!("{label} is not stopped");
                continue;
            }

            let total: Duration = dts.iter().sum();
            let ave = total / count as u32;

            let indent = "    ".repeat(*depth);

            if count == 1 {
                println!("{indent}{total:3.1?} {label}");
            } else {
                println!("{indent}{total:3.1?} {label} (ave:{ave:3.1?} x{count})");
            }
        }

        self.clear();
    }

    fn clear(&mut self) {
        self.record.clear();
        self.stack.clear();
    }
}
