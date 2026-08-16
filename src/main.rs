use image::{ImageBuffer, ImageReader, Luma};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img = ImageReader::open("./src/images/raw_images/5/6.png")?.decode()?;
    let mut processed_img = img.grayscale().adjust_contrast(90.0);
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

    final_canvas.save("./src/images/processed_images/5/6.png")?;

    Ok(())
}
