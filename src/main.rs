use image::ImageReader;

use crate::{neuralNetwork::NeuralNetwork, preprocessor::Preprocessor};

mod neuralNetwork;
mod preprocessor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // let preprocessor = Preprocessor {};
    // let _ = preprocessor.preprocess_images();

    let conv = NeuralNetwork::new();

    let image = ImageReader::open("./images/processed_images/0/7.png")?.decode()?;
    let kernel = [-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0];

    conv.apply_convolution(image, &kernel);

    Ok(())
}
