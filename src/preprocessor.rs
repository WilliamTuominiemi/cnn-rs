use image::{ImageBuffer, ImageReader, Luma};
use std::fs;

const IMAGE_DIRECTORY: &str = "./images/";

pub struct Preprocessor {}

impl Preprocessor {
    pub fn preprocess_images(&self) -> Result<(), Box<dyn std::error::Error>> {
        for n in 0..=9 {
            let path = format!("{}/raw_images/{}", IMAGE_DIRECTORY, n);
            let images = fs::read_dir(path).unwrap();

            for image in images {
                let _ = self.preprocess_image(&image.unwrap().path().display().to_string());
            }
        }

        Ok(())
    }

    fn preprocess_image(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let img = ImageReader::open(path)?.decode()?;
        let mut processed_img = img.grayscale();
        let _ = processed_img.invert();
        processed_img = processed_img.adjust_contrast(60.0).brighten(20);

        let new_width = 16;
        let new_height = new_width;

        let resized_img =
            processed_img.resize(new_width, new_height, image::imageops::FilterType::Nearest);
        let resized_luma = resized_img.into_luma8();
        let mut final_canvas = ImageBuffer::from_pixel(new_width, new_height, Luma([255u8]));

        let x_offset = (new_width - resized_luma.width()) / 2;
        let y_offset = (new_height - resized_luma.height()) / 2;
        image::imageops::replace(
            &mut final_canvas,
            &resized_luma,
            x_offset as i64,
            y_offset as i64,
        );

        final_canvas.save(path.replace("raw_images", "processed_images"))?;
        Ok(())
    }
}
