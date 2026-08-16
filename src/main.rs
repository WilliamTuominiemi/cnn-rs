use image::{ImageBuffer, ImageReader, Luma};
use std::fs;

const IMAGE_DIRECTORY: &str = "./src/images/";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = preprocess_images();

    Ok(())
}

fn preprocess_images() -> Result<(), Box<dyn std::error::Error>> {
    for n in 0..=9 {
        let path = format!("{}/raw_images/{}", IMAGE_DIRECTORY, n);
        let images = fs::read_dir(path).unwrap();

        for image in images {
            let _ = preprocess_image(&image.unwrap().path().display().to_string());
        }
    }

    Ok(())
}

fn preprocess_image(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let img = ImageReader::open(path)?.decode()?;
    let mut processed_img = img.grayscale();
    let _ = processed_img.invert();
    processed_img = processed_img.brighten(20).adjust_contrast(40.0);

    let resized_img = processed_img.resize(32, 32, image::imageops::FilterType::Nearest);
    let resized_luma = resized_img.into_luma8();
    let mut final_canvas = ImageBuffer::from_pixel(32, 32, Luma([255u8]));

    let x_offset = (32 - resized_luma.width()) / 2;
    let y_offset = (32 - resized_luma.height()) / 2;
    image::imageops::replace(
        &mut final_canvas,
        &resized_luma,
        x_offset as i64,
        y_offset as i64,
    );

    final_canvas.save(path.replace("raw_images", "processed_images"))?;
    Ok(())
}
