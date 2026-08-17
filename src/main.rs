use image::ImageReader;

use crate::{convolution::Convolution, preprocessor::Preprocessor};

mod convolution;
mod preprocessor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let preprocessor = Preprocessor {};
    let _ = preprocessor.preprocess_images();

    let conv = Convolution::new();

    let image = ImageReader::open("./images/processed_images/0/7.png")?.decode()?;
    let kernel = [-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0];

    conv.apply_convolution(image, &kernel);

    Ok(())
}
