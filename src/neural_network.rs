use image::ImageReader;
use image::{DynamicImage, ImageBuffer, Luma};
use rand::distr::weighted;
use rand_distr::{Distribution, Normal};
use std::cmp;

const FLATTENED_IMAGE_SIZE: u32 = 64;

pub struct NeuralNetwork {
    filters: Vec<[f32; 9]>,
    weights: [Vec<f32>; 10],
    biases: [f32; 10],
}

impl NeuralNetwork {
    pub fn new() -> NeuralNetwork {
        let mut nn = NeuralNetwork {
            filters: vec![],
            weights: [
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
            ],
            biases: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };

        nn.initialize_filters();
        nn.initialize_weights();

        nn
    }

    pub fn train(&self, num_epochs: u32, image: DynamicImage) {
        // println!("{:?}", self.weights);
        self.forward_propagate(image);
        // for epoch in 1..=num_epochs {
        //     println!("Epoch {}/{}", epoch, num_epochs);
        // }
    }

    fn initialize_filters(&mut self) {
        let mean = 0.0;
        let std_dev = 0.1;
        let normal = Normal::new(mean, std_dev).unwrap();

        let mut kernels: Vec<[f32; 9]> = vec![];

        for _ in 0..8 {
            let mut kernel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            for i in 0..9 {
                kernel[i] = normal.sample(&mut rand::rng());
            }
            kernels.push(kernel);
        }

        self.filters = kernels;
    }

    fn initialize_weights(&mut self) {
        let mean = 0.0;
        let std_dev = (2.0 / (FLATTENED_IMAGE_SIZE as f32 + 10.0)).sqrt();
        let normal = Normal::new(mean, std_dev).unwrap();

        for i in 0..10 {
            let mut weights_for_digit = vec![];
            for _ in 0..FLATTENED_IMAGE_SIZE {
                weights_for_digit.push(normal.sample(&mut rand::rng()));
            }
            self.weights[i] = weights_for_digit;
        }
    }

    fn forward_propagate(&self, image: DynamicImage) {
        for kernel in self.filters.clone() {
            let filtered_image = image.filter3x3(&kernel);
            // Technically should have ReLU on the filtered image but I can't be arsed rn
            let bytes: Vec<f32> = filtered_image
                .as_bytes()
                .iter()
                .map(|&byte| (byte as f32) / 255.0)
                .collect();
            let max_pooled_bytes = self.max_pool(bytes);
            let weighted_bytes = self.apply_weights(&max_pooled_bytes, 0);
        }
    }

    fn apply_weights(&self, image_bytes: &Vec<f32>, digit: usize) -> f32 {
        let bias = self.biases[digit];
        let mut total = 0.0;

        for (weight, byte) in self.weights[digit].iter().zip(image_bytes) {
            total += byte * weight;
        }

        total + bias
    }

    fn max_pool(&self, bytes: Vec<f32>) -> Vec<f32> {
        let stride = 2;
        let amount_of_pixels = bytes.len();
        let side_size = amount_of_pixels.isqrt();
        let max_pooled_side_size = side_size / stride;
        let amount_of_pools = max_pooled_side_size * max_pooled_side_size;
        let mut max_pooled_bytes = vec![0.0; amount_of_pools];

        for pool in 0..amount_of_pools {
            let index = pool * stride;
            let row = pool / amount_of_pools.isqrt();

            let row_offset = row * side_size;

            let top_left = bytes[index + row_offset];
            let top_right = bytes[index + 1 + row_offset];
            let bottom_left = bytes[index + side_size + row_offset];
            let bottom_right = bytes[index + side_size + 1 + row_offset];

            // println!(
            //     "pool: {} index: {} row: {} top_left: {} top_right: {} bottom_left: {} bottom_right: {}",
            //     pool, index, row, top_left, top_right, bottom_left, bottom_right
            // );

            let pool_value = top_left
                .max(top_right)
                .max(bottom_left)
                .max(bottom_right)
                .ceil();

            max_pooled_bytes[pool] = pool_value;
        }

        max_pooled_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_pool() {
        let nn = NeuralNetwork::new();
        let bytes_2x2: Vec<f32> = vec![
            29.0, 15.0, 28.0, 184.0, 0.0, 100.0, 70.0, 38.0, 12.0, 12.0, 7.0, 2.0, 12.0, 12.0,
            45.0, 6.0,
        ];
        let expected_bytes_2x2: Vec<f32> = vec![100.0, 184.0, 12.0, 45.0];

        assert_eq!(nn.max_pool(bytes_2x2), expected_bytes_2x2);

        let bytes_3x3 = vec![
            19.0, 22.0, 20.0, 12.0, 17.0, 11.0, 16.0, 30.0, 1.0, 23.0, 7.0, 14.0, 14.0, 24.0, 7.0,
            2.0, 1.0, 7.0, 15.0, 10.0, 1.0, 1.0, 15.0, 1.0, 13.0, 13.0, 11.0, 5.0, 13.0, 7.0, 18.0,
            9.0, 18.0, 13.0, 3.0, 4.0,
        ];
        let expected_bytes_3x3: Vec<f32> =
            vec![30.0, 23.0, 17.0, 24.0, 7.0, 15.0, 18.0, 18.0, 13.0];
        assert_eq!(nn.max_pool(bytes_3x3), expected_bytes_3x3);
    }
}
