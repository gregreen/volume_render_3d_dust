#!/usr/bin/env python3

import numpy as np

import matplotlib.pyplot as plt
#plt.rcParams['text.usetex'] = True

import io
from PIL import Image

from render_dust_3d import get_rotation_matrix


def render_latex_to_image(latex_str, dpi=300, fontsize=24, pad_inches=0.1):
    """
    Renders a LaTeX string into a PIL Image using matplotlib.
    
    Parameters:
      latex_str (str): The LaTeX string to render (without the surrounding $...$).
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
    text = ax.text(0.5, 0.5, f'${latex_str}$', fontsize=fontsize,
                   horizontalalignment='center', verticalalignment='center',
                   color='k',
                   transform=ax.transAxes)
    
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
    plt.savefig(buf, format='png', dpi=dpi, transparent=True, bbox_inches='tight', pad_inches=pad_inches)
    plt.close(fig)
    buf.seek(0)
    
    # Open image with Pillow.
    return Image.open(buf)


def paste_label_on_canvas_old(label_img, center_pixel, target_width, alpha, canvas_size):
    """
    Paste a label image onto a blank RGBA canvas using an affine transformation
    for smooth, continuous scaling and positioning.
    
    This approach uses a floating-point scaling factor applied via an affine transform
    to avoid discrete resizing steps.
    
    Parameters:
      label_img (PIL.Image): The input label image.
      center_pixel (tuple): (x, y) coordinates for the desired center of the label on the canvas.
      target_width (float): The desired width of the label in the final canvas.
      alpha (float): An alpha multiplier (0.0 to 1.0) applied to the label image.
      canvas_size (tuple): (width, height) for the output canvas.
    
    Returns:
      PIL.Image: The canvas image with the label transformed and composited.
    """
    # Ensure the label image is in RGBA mode.
    label_img = label_img.convert("RGBA")
    
    # Get original dimensions.
    orig_width, orig_height = label_img.size
    
    # Compute the scale factor.
    scale = target_width / orig_width

    # Calculate the transformation matrix.
    # For an affine transform, Pillow uses a 6-tuple (a, b, c, d, e, f) that maps output pixel
    # (x, y) to input coordinates (x', y') as:
    #   x' = a*x + b*y + c,  y' = d*x + e*y + f.
    #
    # We want to scale the label by 'scale' and translate it so that the label's center
    # (orig_width/2, orig_height/2) maps to the specified center_pixel.
    # This gives:
    #   c = center_pixel[0] - scale * (orig_width / 2)
    #   f = center_pixel[1] - scale * (orig_height / 2)
    a = scale
    b = 0
    c = center_pixel[0] - scale * (orig_width / 2)
    d = 0
    e = scale
    f_val = center_pixel[1] - scale * (orig_height / 2)
    
    transform_matrix = (a, b, c, d, e, f_val)
    print(transform_matrix)
    
    # Apply the affine transformation.
    # The output size is the canvas size, so the label will be rendered within the canvas
    # using the computed transform. The transformation is computed with floating-point precision,
    # which leads to smooth transitions when 'target_width' changes.
    transformed_label = label_img.transform(
        canvas_size, 
        Image.AFFINE, 
        transform_matrix, 
        resample=Image.BILINEAR
    )

    return transformed_label
    
    # Adjust the alpha channel by multiplying it with the given alpha.
    r, g, b, a_channel = transformed_label.split()
    a_channel = a_channel.point(lambda p: int(p * alpha))
    transformed_label.putalpha(a_channel)
    
    # Create a blank canvas.
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    
    # Composite the transformed label onto the canvas.
    canvas = Image.alpha_composite(canvas, transformed_label)
    return canvas


def paste_label_on_canvas(label_img, center_pixel, target_width, alpha, canvas_size):
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
    
    Returns:
      PIL.Image: The canvas image with the label transformed and composited.
    """
    # Ensure the label image is in RGBA mode.
    label_img = label_img.convert("RGBA")
    
    # Get original dimensions.
    orig_width, orig_height = label_img.size
    
    # Compute the forward scale factor.
    scale = target_width / orig_width
    inv_scale = 1.0 / scale  # Inverse scale factor for the affine transform.
    
    # Compute the inverse transformation matrix.
    # For a canvas coordinate (x, y), the corresponding label image coordinate (u, v) is:
    #   u = (x - center_pixel[0]) / scale + orig_width/2
    #   v = (y - center_pixel[1]) / scale + orig_height/2
    a = inv_scale
    b = 0
    c = orig_width/2 - center_pixel[0] * inv_scale
    d = 0
    e = inv_scale
    f_val = orig_height/2 - (canvas_size[1]-center_pixel[1]) * inv_scale
    transform_matrix = (a, b, c, d, e, f_val)
    
    # Apply the affine transformation.
    transformed_label = label_img.transform(
        canvas_size, 
        Image.AFFINE, 
        transform_matrix, 
        resample=Image.BILINEAR
    )
    
    # Adjust the alpha channel by multiplying it with the given alpha.
    r, g, b, a_channel = transformed_label.split()
    a_channel = a_channel.point(lambda p: int(p * alpha))
    transformed_label.putalpha(a_channel)
    
    # Create a blank canvas.
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    
    # Composite the transformed label onto the canvas.
    canvas = Image.alpha_composite(canvas, transformed_label)
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


def calc_label_center(xyz, camera_x0, camera_rot, fov, canvas_shape):
    # Calculate (x,y,z) of label in camera's native frame
    dxyz = calc_label_dxyz(xyz, camera_x0, camera_rot)

    # Project camera coordinates to image plane
    u, v = gnomonic_xyz_to_uv(*dxyz, fov)

    # Scale image coordinates
    w,h = canvas_shape
    u = (u+0.5) * w
    v = 0.5*h - v*w

    return u, v


# Example usage
def main():
    label_xyz = (3, 0, 0)
    camera_x0 = (0, 0, 0)
    camera_rot = (0, 20, 0)
    fov = 90.
    canvas_shape = (600, 400)

    center = calc_label_center(label_xyz, camera_x0, camera_rot,
                               fov, canvas_shape)
    print(center)

    # Generate the label image
    latex_code = r'\odot'
    label_img = render_latex_to_image(latex_code, fontsize=24)

    ## Set the label location parameters
    #center = (250, 350)         # The pixel where the label's center should appear.
    target_width = 150.0        # A float value for smooth scaling.
    alpha_val = 0.8             # Alpha multiplier.
    #canvas_shape = (600, 400)   # Output canvas dimensions.
    #
    result = paste_label_on_canvas(label_img, center, target_width,
                                   alpha_val, canvas_shape)

    label_img.show()  # Displays the label image.
    result.show()  # Displays the canvas with the scaled label image.
    
    return 0


if __name__ == '__main__':
    main()

