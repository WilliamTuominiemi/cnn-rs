use rand::rng;
use rand_distr::{Distribution, StandardNormal};

pub struct Convolution {
    no_of_kernels: u32,
    kernel_size: u32,
    stride: u32,
    bias: u32,
    kernels: Vec<f32>,
}

impl Convolution {
    pub fn new(no_of_kernels: u32, kernel_size: u32, stride: u32, bias: u32) -> Convolution {
        let mut rng = rng();
        let normal = StandardNormal;
        let scale = (kernel_size * kernel_size) as f32;

        let kernels: Vec<f32> = (0..no_of_kernels * kernel_size * kernel_size)
            .map(|_| {
                let sample: f32 = normal.sample(&mut rng);
                sample / scale
            })
            .collect();

        Convolution {
            no_of_kernels,
            kernel_size,
            stride,
            bias,
            kernels,
        }
    }
}
