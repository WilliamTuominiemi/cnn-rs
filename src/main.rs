use crate::preprocessor::Preprocessor;

mod preprocessor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let preprocessor = Preprocessor {};
    let _ = preprocessor.preprocess_images();

    Ok(())
}
