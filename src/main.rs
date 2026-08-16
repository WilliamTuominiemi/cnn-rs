use image::ImageReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img = ImageReader::open("./src/images/raw_images/0/1.png")?.decode()?;
    let mut processed_img = img.grayscale().adjust_contrast(90.0);
    let _ = processed_img.invert();
    processed_img = processed_img.brighten(20).adjust_contrast(40.0);
    processed_img.save("./src/images/processed_images/0/1.png")?;

    Ok(())
}
