import io
from pathlib import Path

import numpy as np
import torch
from PIL import ExifTags, Image, ImageCms, ImageOps
from PIL.Image import Image as PilImage


def open_image_as_srgb(image_path: str | Path | io.BytesIO) -> PilImage:
    ''
    exif_colorspace_srgb = 1

    with Image.open(image_path) as img_raw:
        img = ImageOps.exif_transpose(img_raw)

    input_icc_profile = img.info.get("icc_profile")

                                                                  
    srgb_profile = ImageCms.createProfile(colorSpace="sRGB")
    if input_icc_profile is not None:
        input_profile = ImageCms.ImageCmsProfile(io.BytesIO(input_icc_profile))
        srgb_img = ImageCms.profileToProfile(img, input_profile, srgb_profile, outputMode="RGB")
    else:
                                        
        exif_data = img.getexif()
        if exif_data is not None:
                                                                          
            color_space_value = exif_data.get(ExifTags.Base.ColorSpace.value)
            if color_space_value is not None and color_space_value != exif_colorspace_srgb:
                raise ValueError(
                    "Image has colorspace tag in EXIF but it isn't set to sRGB,"
                    " conversion is not supported."
                    f" EXIF ColorSpace tag value is {color_space_value}",
                )

        srgb_img = img.convert("RGB")

                                                                                    
        srgb_profile_data = ImageCms.ImageCmsProfile(srgb_profile).tobytes()
        srgb_img.info["icc_profile"] = srgb_profile_data

    return srgb_img


def pil_image_to_tensor(image: PilImage) -> torch.Tensor:
    ''
    image_np = np.array(image, dtype=np.uint8, copy=True)
    return torch.from_numpy(image_np).permute(2, 0, 1).float().div_(255)


def save_image(image_tensor: torch.Tensor, output_path: Path | str) -> None:
    ''
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

                                                                 
    if image_tensor.ndim == 4:
                                                            
        if image_tensor.shape[1] == 1:
            image_tensor = image_tensor.squeeze(1)
        else:
            raise ValueError(f"Expected single-frame tensor with shape [C, 1, H, W], got shape {image_tensor.shape}")

    if image_tensor.ndim != 3:
        raise ValueError(f"Expected 3D tensor [C, H, W], got {image_tensor.ndim}D tensor")

    if image_tensor.shape[0] != 3:
        raise ValueError(f"Expected 3 channels (RGB), got {image_tensor.shape[0]} channels")

                                 
    if torch.is_floating_point(image_tensor) and image_tensor.max() <= 1.0:
        image_tensor = image_tensor * 255

                                                    
    image_tensor = image_tensor.clamp(0, 255)

                            
    image_np: np.ndarray = image_tensor.permute(1, 2, 0).to(torch.uint8).cpu().numpy()

                    
    Image.fromarray(image_np).save(output_path)
