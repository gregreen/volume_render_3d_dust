#!/usr/bin/env python3

import numpy as np

import matplotlib.pyplot as plt
#plt.rcParams['text.usetex'] = True

import io
from PIL import Image

from render_dust_3d import get_rotation_matrix


def crop_image_transparent(image):
    """
    Crops an image to the bounding box of non-transparent regions.
    """
    # Get the alpha channel and compute its bounding box.
    alpha = image.split()[-1]
    bbox = alpha.getbbox()
    if bbox:
        return image.crop(bbox)
    return image


def render_text_to_image(s, dpi=300, fontsize=24, pad_inches=0.1,
                         **kwargs):
    """
    Renders a string into a PIL Image using matplotlib.
    
    Parameters:
      s (str): The string to render.
      dpi (int): Resolution of the rendered image.
      fontsize (int): Font size for the rendered text.
      pad_inches (float): Padding (in inches) around the rendered text.
    
    Returns:
      PIL.Image: The image of the rendered LaTeX string.
    """
    # Create a figure with no visible axes.
    fig = plt.figure()
    fig.patch.set_alpha(0.0)  # Make the background transparent.
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Render the text. Surround with $...$ to enable math mode.
    text = ax.text(0.5, 0.5, s, fontsize=fontsize,
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes,
                   **kwargs)
    
    # Adjust figure size based on text extent.
    fig.canvas.draw()
    bbox = text.get_window_extent()
    # Convert from display to inches.
    width, height = bbox.width / dpi, bbox.height / dpi
    fig.set_size_inches(width, height)
    
    # Remove margins.
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save the figure to a bytes buffer.
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, transparent=True,
                bbox_inches='tight', pad_inches=pad_inches)
    plt.close(fig)
    buf.seek(0)
    
    # Open image with Pillow.
    img = Image.open(buf)

    # Crop off transparent margins
    img = crop_image_transparent(img)

    return img


def paste_label_on_canvas(label_img, center_pixel, target_width,
                          alpha, canvas_size,
                          oversampling=4, background_rgba=(0,0,0,0)):
    """
    Paste a label image onto a blank RGBA canvas using an affine transformation
    for smooth, continuous scaling and positioning.
    
    This version uses the inverse transformation matrix, which maps canvas (output)
    coordinates to the label image (input) coordinates.
    
    Parameters:
      label_img (PIL.Image): The input label image.
      center_pixel (tuple): (x, y) coordinates for the desired center of the label on the canvas.
      target_width (float): The desired width of the label in the final canvas.
      alpha (float): An alpha multiplier (0.0 to 1.0) applied to the label image.
      canvas_size (tuple): (width, height) for the output canvas.
      oversampling (int): The oversampling factor for the label image.
      background_rgba (tuple): The RGBA color of the background.
    
    Returns:
      PIL.Image: The canvas image with the label transformed and composited.
    """
    # Ensure the label image is in RGBA mode.
    label_img = label_img.convert("RGBA")
    
    # Get original dimensions.
    orig_width, orig_height = label_img.size
    
    # Compute the forward scale factor.
    scale = oversampling * target_width / orig_width
    inv_scale = 1.0 / scale  # Inverse scale factor for the affine transform.

    # Compute the oversampled canvas dimensions
    oversampled_canvas_size = tuple([int(x*oversampling) for x in canvas_size])
    oversampled_width, oversampled_height = oversampled_canvas_size
    x0, y0 = [x*oversampling for x in center_pixel]
    
    # Compute the inverse transformation matrix.
    # For a canvas coordinate (x, y), the corresponding label image coordinate (u, v) is:
    #   u = (x - center_pixel[0]) / scale + orig_width/2
    #   v = (y - center_pixel[1]) / scale + orig_height/2
    a = inv_scale
    b = 0
    c = orig_width/2 - x0 * inv_scale
    d = 0
    e = inv_scale
    f_val = orig_height/2 - (oversampled_height-y0) * inv_scale
    transform_matrix = (a, b, c, d, e, f_val)
    
    # Apply the affine transformation.
    transformed_label = label_img.transform(
        oversampled_canvas_size,
        Image.AFFINE,
        transform_matrix,
        resample=Image.BICUBIC
    )
    
    # Adjust the alpha channel by multiplying it with the given alpha.
    r, g, b, a_channel = transformed_label.split()
    a_channel = a_channel.point(lambda p: int(p * alpha))
    transformed_label.putalpha(a_channel)
    
    # Create a blank canvas.
    canvas = Image.new('RGBA', oversampled_canvas_size, background_rgba)
    
    # Composite the transformed label onto the canvas.
    canvas = Image.alpha_composite(canvas, transformed_label)

    # Finally, downsample the oversampled canvas to the final canvas size.
    canvas = canvas.resize(canvas_size, resample=Image.Resampling.LANCZOS)

    return canvas


def calc_label_dxyz(xyz, camera_x0, camera_rot):
    """
    Calculates the delta_xyz from the camera to the label, in a coordinate
    system oriented with the camera.
    """
    dxyz = np.asarray(xyz) - np.asarray(camera_x0)

    # Get the camera rotation matrix
    R = get_rotation_matrix(*camera_rot)
    # Label coordinates rotate inversely with camera
    R_inv = np.linalg.inv(R)
    dxyz = np.dot(R_inv, dxyz)

    return dxyz


def gnomonic_xyz_to_uv(x, y, z, fov):
    print(x, y, z)

    # Convert (x,y,z) to (theta,phi)
    phi = np.arctan2(y,x)
    theta = np.arccos(z/np.sqrt(x**2+y**2+z**2))

    # Convert (theta,phi) to projected coordinates (u,v)
    r = np.tan(theta) / (2*np.tan(0.5*np.radians(fov)))
    print(r)
    u = r * np.cos(phi)
    v = r * np.sin(phi)

    return u, v


def calc_label_center_and_width(label_xyz,
                                camera_x0, camera_rot, fov,
                                canvas_shape,
                                reference_distance=0.5,
                                reference_size=0.1):
    """
    Calculates the center and width of a label in a 2D image plane, given
    the camera's position and orientation, the label's position, and the
    field of view.

    Args:
      label_xyz (tuple): The (x, y, z) coordinates of the label in the camera's frame.
      camera_x0 (tuple): The (x, y, z) coordinates of the camera's position.
      camera_rot (tuple): The (lon, lat, roll) angles of the camera's orientation.
      fov (float): The field of view of the camera in degrees.
      canvas_shape (tuple): The (width, height) of the canvas.
      reference_distance (float): The distance at which the label's width is equal
                                  to a fraction reference_size of the canvas width.
      reference_size (float): The width of the label at the reference distance.

    Returns:
      tuple: The (center, width) of the label in the image plane.
    """
    # Calculate (x,y,z) of label in camera's native frame
    dxyz = calc_label_dxyz(label_xyz, camera_x0, camera_rot)

    # Project camera coordinates to image plane
    u, v = gnomonic_xyz_to_uv(*dxyz, fov)

    # Scale image coordinates
    w,h = canvas_shape
    u = (u+0.5) * w
    v = 0.5*h - v*w

    # Calculate image size, by setting the width to a given fraction
    # (reference_size) of the canvas width at a given reference distance
    # (reference_distance)
    r = np.linalg.norm(dxyz)
    label_width = reference_size * w * reference_distance / r

    return (u,v), label_width


# Example usage
def main():
    label_xyz = (3, 0, 0.5)
    camera_x0 = (0, 0, 0)
    camera_rot = (0, 20, 0)
    fov = 90.
    canvas_shape = (600, 400)
    alpha_val = 0.8 # Label alpha multiplier

    # Generate the label image
    label_text = r'A'
    label_img = render_text_to_image(
        label_text,
        fontsize=24,
        pad_inches=0,
        dpi=600,
        color='k'
    )
    label_img.save('frames/label_test_individual.png')

    from tqdm.auto import tqdm

    for i in tqdm(range(11)):
        label_xyz = (i*0.5+0.1, 0, 0)
        center,target_width = calc_label_center_and_width(
            label_xyz, camera_x0,
            camera_rot, fov,
            canvas_shape
        )
        print(center, target_width)

        result = paste_label_on_canvas(
            label_img, center,
            target_width, alpha_val,
            canvas_shape,
            background_rgba=(255,255,255,255)
        )

        result.save(f'frames/label_test_canvas_{i:04d}.png')

    return 0


if __name__ == '__main__':
    main()

