use image::ImageReader;

use crate::{neuralNetwork::NeuralNetwork, preprocessor::Preprocessor};

mod neuralNetwork;
mod preprocessor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // let preprocessor = Preprocessor {};
    // let _ = preprocessor.preprocess_images();

    let nn = NeuralNetwork::new();

    let image = ImageReader::open("./images/processed_images/0/7.png")?.decode()?;
    let kernel = [-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0];

    let altered_image = nn.apply_convolution(image, &kernel);
    let _ = altered_image.save("kernel.png");

    let max_pooled_image = nn.apply_max_pool(altered_image);
    let _ = max_pooled_image.save("pooled.png");

    Ok(())
}
