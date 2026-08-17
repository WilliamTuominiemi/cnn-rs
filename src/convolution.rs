use image::DynamicImage;

pub struct Convolution {}

impl Convolution {
    pub fn new() -> Convolution {
        Convolution {}
    }

    pub fn apply_convolution(&self, image: DynamicImage, kernel: &[f32]) {
        let altered_image = image.filter3x3(kernel);
        let _ = altered_image.save("kernel.png");
    }
}
