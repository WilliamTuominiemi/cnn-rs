use image::ImageReader;

use crate::{neural_network::NeuralNetwork, preprocessor::Preprocessor};

mod neural_network;
mod preprocessor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // let preprocessor = Preprocessor {};
    // let _ = preprocessor.preprocess_images();

    let nn = NeuralNetwork::new();

    let image = ImageReader::open("./images/processed_images/0/7.png")?.decode()?;
    // let kernel: [f32; 9] = [-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0];

    // let altered_image = nn.apply_convolution(image, &kernel);
    // let _ = altered_image.save("kernel.png");

    // let max_pooled_image = nn.apply_max_pool(altered_image);
    // let _ = max_pooled_image.save("pooled.png");

    nn.train(100, image);

    Ok(())
}
