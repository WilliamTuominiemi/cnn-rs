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
        println!("{:?}", self.weights);
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
            let max_pooled = self.apply_max_pool(filtered_image);

            let flattened = max_pooled.into_luma8();

            println!("{:?}", flattened.len());

            // let flattened = match max_pooled.as_flat_samples_f32() {
            //     Some(f) => f,
            //     None => panic!("Couln't flatten pooled image"),
            // };

            // pri
        }
    }

    fn apply_weights(
        &self,
        flat_image: ImageBuffer<Luma<u8>, Vec<u8>>,
        digit: usize,
    ) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        let image_bytes = flat_image.into_raw();

        let bias = self.biases[digit];

        let mut weighted_bytes: Vec<u8> = vec![];

        for (weight, byte) in self.weights[digit].clone().iter().zip(&image_bytes) {
            let adjusted_weight = (weight * 255.0) as u8; // Don't ask
            let adjusted_bias = (bias * 255.0) as u8; // don't tell
            let weighted_byte = byte * adjusted_weight + adjusted_bias;
            weighted_bytes.push(weighted_byte);
        }

        assert_eq!(
            image_bytes.len(),
            weighted_bytes.len(),
            "weighted image bytes length differs from original"
        );

        let buffer = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(
            FLATTENED_IMAGE_SIZE,
            FLATTENED_IMAGE_SIZE,
            weighted_bytes,
        )
        .expect("weighted byte length doesn't match new_width * new_height");

        buffer
    }

    fn apply_max_pool(&self, image: DynamicImage) -> DynamicImage {
        let bytes = image.as_bytes();
        let pooled_bytes = self.max_pool(bytes);

        let new_width = image.width() / 2;
        let new_height = image.width() / 2;

        let buffer =
            ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(new_width, new_height, pooled_bytes)
                .expect("pooled byte length doesn't match new_width * new_height");

        DynamicImage::ImageLuma8(buffer)
    }

    fn max_pool(&self, bytes: &[u8]) -> Vec<u8> {
        let stride = 2;
        let amount_of_pixels = bytes.len();
        let side_size = amount_of_pixels.isqrt();
        let max_pooled_side_size = side_size / stride;
        let amount_of_pools = max_pooled_side_size * max_pooled_side_size;
        let mut max_pooled_bytes = vec![0; amount_of_pools];

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

            let pool_value = cmp::max(
                cmp::max(top_left, top_right),
                cmp::max(bottom_left, bottom_right),
            );

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
        let bytes_2x2 = [29, 15, 28, 184, 0, 100, 70, 38, 12, 12, 7, 2, 12, 12, 45, 6];
        let expected_bytes_2x2 = vec![100, 184, 12, 45];

        assert_eq!(nn.max_pool(&bytes_2x2), expected_bytes_2x2);

        let bytes_3x3 = [
            19, 22, 20, 12, 17, 11, 16, 30, 1, 23, 7, 14, 14, 24, 7, 2, 1, 7, 15, 10, 1, 1, 15, 1,
            13, 13, 11, 5, 13, 7, 18, 9, 18, 13, 3, 4,
        ];
        let expected_bytes_3x3 = vec![30, 23, 17, 24, 7, 15, 18, 18, 13];
        assert_eq!(nn.max_pool(&bytes_3x3), expected_bytes_3x3);
    }
}
