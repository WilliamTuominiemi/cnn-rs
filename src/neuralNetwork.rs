use image::DynamicImage;
use std::cmp;

pub struct NeuralNetwork {}

impl NeuralNetwork {
    pub fn new() -> NeuralNetwork {
        NeuralNetwork {}
    }

    pub fn apply_convolution(&self, image: DynamicImage, kernel: &[f32]) -> DynamicImage {
        image.filter3x3(kernel)
    }

    pub fn apply_max_pool(&self, image: DynamicImage) {}

    fn max_pool(&self, bytes: &[u8]) -> Vec<u8> {
        let stride = 2;
        let amount_of_pixels = bytes.len();
        let side_size = amount_of_pixels.isqrt();
        let max_pooled_side_size = side_size / stride;
        let amount_of_pools = max_pooled_side_size * max_pooled_side_size;
        let mut max_pooled_bytes = vec![0; amount_of_pools];

        println!("{}", amount_of_pools);

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
