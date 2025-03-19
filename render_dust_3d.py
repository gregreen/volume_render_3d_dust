import numpy as np

import tensorflow as tf
from tensorflow_graphics.math.interpolation.trilinear import interpolate as trilinear_interpolate

from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
plt.style.use('dark_background')

from astropy.coordinates import SkyCoord
import astropy.units as u

from scipy.interpolate import CubicSpline

from tqdm.auto import tqdm

from rotations import get_rotation_matrix
from render_labels import render_text_to_image, paste_label_on_canvas, calc_label_center_and_width, alpha_composite_many


def replace_nonfinite(v, fill):
    """
    Replaces non-finite values in a numpy ndarray with a specified fill value.
    """
    idx = ~np.isfinite(v)
    v[idx] = fill


def combine_zg25_bayestar(map_values_zg25, ln_dist_edges_zg25,
                          map_values_bayestar, ln_dist_edges_bayestar):
    """
    Combines the dust density from Bayestar with the R(V) channel
    from Zhang & Green (2025).
    """

    # Extract (1/R0 - 1/R) from ZG25 map
    color_zg25 = map_values_zg25[1] / (map_values_zg25[0] + 1e-5)

    # Determine distance grid on which to resample ZG25
    idx_pull = np.searchsorted(ln_dist_edges_zg25, ln_dist_edges_bayestar) - 1
    idx_valid = (idx_pull >= 0) & (idx_pull < len(ln_dist_edges_zg25))
    idx_pull[~idx_valid] = 0

    # Pull values from ZG25
    color_zg25_resamp = color_zg25.numpy()[:,:,idx_pull]
    color_zg25_resamp[:,:,~idx_valid] = 0

    # Multiply by Bayestar extinction density
    color_zg25_resamp *= map_values_bayestar[0].numpy()
    print(color_zg25_resamp)

    # Convert to Tensor
    color_zg25_resamp = tf.constant(color_zg25_resamp, dtype=tf.float32)

    return [map_values_bayestar[0],color_zg25_resamp], ln_dist_edges_bayestar


def load_zhang_green_2025(n_lon, n_lat, return_tensor=True):
    """
    Loads the Zhang & Green (2025) 3D dust map, which contains not only dust
    density, but also R(V). We will load differential versions of both A(V)
    and R(55), which will be used to render the videos.

    Args:
        n_lon (int): Number of grid points in longitude.
        n_lat (int): Number of grid points in latitude.
        return_tensor (bool, optional): If True, converts the dust map values to a TensorFlow tensor.
                                        Defaults to True.

    Returns:
        tuple: A tuple (map_values, ln_dist_edges) where:
            - map_values (tf.Tensor): Tensor of differential [A(V),R(55)] values (dtype=tf.float32).
            - ln_dist_edges (np.ndarray): Array of logarithm of distance bin edges.
    """

    from astropy_healpix import HEALPix, npix_to_nside
    import h5py

    fname = 'data/zhang_green_2025.h5'

    # Load needed datasets
    keys = ('E', 'R55')
    keys_dset = [f'map_64/{k}_map_diff' for k in keys]
    with h5py.File(fname, 'r') as f:
        d = {k0: f[k1][:].astype('f4')
             for k0,k1 in zip(keys,keys_dset)}
        dist_bin_edges = f['distance_bins'][:].astype('f4')
    
    # Replace negative, NaN, inf values in E and R(55)
    idx = (d['E'] < 0)
    d['E'][idx] = 0
    replace_nonfinite(d['E'], 0)
    replace_nonfinite(d['R55'], np.nanmedian(d['R55']))

    # Convert to (E, E * (1/R55_zp - 1/R55))
    d['R55'] = d['E'] * (0.36278367 - 1/d['R55'])
    
    # Grid of longitudes and latitudes
    lon = np.linspace(0, 360, n_lon, endpoint=False)
    lat = np.linspace(-90, 90, n_lat, endpoint=False)
    lon += 0.5 * (lon[1] - lon[0])
    lat += 0.5 * (lat[1] - lat[0])
    lon, lat = np.meshgrid(lon, lat, indexing='ij')

    # Datasets are in nested order. Sample on a grid of lon,lat,dist.
    npix = d[list(d.keys())[0]].shape[1]
    print(f'npix = {npix}')
    nside = npix_to_nside(npix)
    hpix = HEALPix(nside=nside, order='nested')
    pix_idx = hpix.lonlat_to_healpix(lon*u.deg, lat*u.deg)
    map_values = np.stack(
        [d[k][:,pix_idx].transpose(1,2,0) for k in keys],
        axis=0
    )

    print('map_values.shape =', map_values.shape)

    if return_tensor:
        map_values = tf.constant(map_values, dtype=tf.float32)

    return map_values, np.log(dist_bin_edges[1:])


def save_dustmap_cube(fname, map_values, ln_dist_edges):
    """
    Saves a 3D dust map in a Cartesian sky projection to an NPZ file. This map
    is faster to load than the original files used by the dustmaps package.

    Args:
        fname (str): Filename (with path) where the dust map will be saved.
        map_values (np.ndarray or tf.Tensor): A 3D array/tensor containing dust map density values.
        ln_dist_edges (np.ndarray or tf.Tensor): A 1D array/tensor containing logarithm of distance bin edges.

    Returns:
        None
    """
    map_values = np.asarray(map_values)
    ln_dist_edges = np.asarray(ln_dist_edges)
    np.savez(fname, map_values=map_values, ln_dist_edges=ln_dist_edges)


def load_dustmap_cube(fname):
    """
    Loads a 3D dust map from an NPZ file that has been converted to a Cartesian
    sky projection. This is faster than loading the maps using the dustmaps
    package.

    Args:
        fname (str): Filename (with path) of the NPZ file containing the dust map.

    Returns:
        tuple: A tuple (map_values, ln_dist_edges) where:
            - map_values (tf.Tensor): Tensor of dust map values (dtype=tf.float32).
            - ln_dist_edges (np.ndarray): Array of logarithm of distance bin edges.
    """
    with np.load(fname) as f:
        map_values = tf.constant(f['map_values'], dtype=tf.float32)
        ln_dist_edges = f['ln_dist_edges']
    return map_values, ln_dist_edges


def load_bayestar_decaps(n_lon, n_lat):
    """
    Loads a combined dust map using Bayestar19 and DECaPS, preferring DECaPS values.

    Args:
        n_lon (int): Number of grid points in longitude.
        n_lat (int): Number of grid points in latitude.

    Returns:
        tuple: A tuple (map_values, ln_dist_edges) where:
            - map_values (tf.Tensor): Tensor of combined dust map values (dtype=tf.float32).
            - ln_dist_edges (np.ndarray): Array of logarithm of distance bin edges.
    """
    # Load Bayestar19 and DECaPS
    map_values_b19, ln_dist_edges_b19 = load_dustmap(
        n_lon, n_lat,
        which='bayestar2019',
        return_tensor=False  # Don't convert to tensor yet
    )
    map_values_decaps, ln_dist_edges_decaps = load_dustmap(
        n_lon, n_lat,
        which='decaps',
        return_tensor=False
    )

    # Convert Bayestar19 to the same units as DECaPS
    ratio = 2.742 / 3.32
    map_values_b19 *= ratio

    # Paste DECaPS values into the Bayestar19 map
    idx = np.isfinite(map_values_decaps)
    map_values_b19[idx] = map_values_decaps[idx]

    # Convert to tensor
    map_values = tf.constant(map_values_b19, dtype=tf.float32)

    return map_values, ln_dist_edges_b19


def load_dustmap(n_lon, n_lat, which='bayestar2019', return_tensor=True):
    """
    Loads a dust map (Bayestar or DECaPS) and converts the sky projection into a Cartesian projection.

    Args:
        n_lon (int): Number of grid points in longitude.
        n_lat (int): Number of grid points in latitude.
        which (str, optional): Identifier for the dust map ('bayestar201?' or 'decaps'). Defaults to 'bayestar2019'.
        return_tensor (bool, optional): If True, converts the dust map values to a TensorFlow tensor.
                                        Defaults to True.

    Returns:
        tuple: A tuple (map_values, ln_dist_edges) where:
            - map_values (np.ndarray or tf.Tensor): The dust map density values.
            - ln_dist_edges (np.ndarray): Logarithm of distance bin edges.
    """
    # Load the dust map
    if which.startswith('bayestar'):
        from dustmaps.bayestar import BayestarQuery
        q = BayestarQuery(version=which, max_samples=100)
    elif which == 'decaps':
        from dustmaps.decaps import DECaPSQuery
        q = DECaPSQuery(mean_only=True)
    else:
        raise NotImplementedError(f'Dust map "{which}" not implemented.')

    ln_dist_edges = np.log(q.distances.to('kpc').value)
    n_dist_bins = len(ln_dist_edges) - 1
    print('Distance edges:', q.distances.to('kpc'))

    # Grid of longitudes and latitudes
    lon = np.linspace(0, 360, n_lon, endpoint=False)
    lat = np.linspace(-90, 90, n_lat, endpoint=False)
    lon += 0.5 * (lon[1] - lon[0])
    lat += 0.5 * (lat[1] - lat[0])
    lon, lat = np.meshgrid(lon, lat, indexing='ij')

    # Query the dust map
    map_values = np.stack([
        q.query_gal(
            lon, lat,
            d=np.exp(ln_dist_edges[i]),
            mode='mean'
        ).astype('f4')
        for i in tqdm(range(n_dist_bins + 1))
    ], axis=-1)

    # Convert to difference map
    map_values = np.concatenate([
        np.expand_dims(map_values[..., 0], axis=-1),
        np.diff(map_values, axis=-1)
    ], axis=-1)

    # Convert differences to densities
    delta_dist = np.hstack([
        np.exp(ln_dist_edges[0]),
        np.diff(np.exp(ln_dist_edges))
    ]).astype('f4')
    map_values = map_values / delta_dist[None, None]

    if return_tensor:
        map_values = tf.constant(map_values, dtype=tf.float32)

    return map_values, ln_dist_edges


def gnomonic_projection(fov, image_shape, dtype=tf.float32):
    """
    Generates ray directions for a gnomonic projection given a field of view and image shape.

    Args:
        fov (float): Field of view in degrees.
        image_shape (tuple): Tuple (width, height) of the output image.
        dtype (tf.DType, optional): Data type for the output tensor. Defaults to tf.float32.

    Returns:
        tf.Tensor: A tensor of shape (image_shape[0], image_shape[1], 3) containing normalized ray directions.
    """
    x_max = tf.constant(1., dtype=dtype)
    y_max = tf.constant(image_shape[1] / image_shape[0], dtype=dtype)
    fov = tf.constant(np.radians(fov), dtype=dtype)
    u = tf.linspace(-x_max, x_max, image_shape[0])
    v = tf.linspace(-y_max, y_max, image_shape[1])
    u, v = tf.meshgrid(u, v, indexing='ij')
    focal_length = 1.0 / tf.tan(fov / 2.0)  # Focal length
    ray_directions = tf.stack([u, v, tf.ones_like(u) * focal_length], axis=-1)
    ray_directions = tf.linalg.normalize(ray_directions, axis=-1)[0]  # Normalize
    return ray_directions


def xyz_to_spherical(xyz):
    """
    Converts Cartesian coordinates to spherical coordinates.

    Args:
        xyz (tf.Tensor): Tensor of shape (..., 3) representing Cartesian coordinates (x, y, z).

    Returns:
        tuple: A tuple (r, theta, phi) where:
            - r: Radial distances.
            - theta: Polar angles (in radians).
            - phi: Azimuthal angles (in radians).
    """
    x, y, z = tf.unstack(xyz, axis=-1)
    r = tf.sqrt(x**2 + y**2 + z**2)
    theta = tf.acos(z / r)
    phi = tf.atan2(y, x)
    return r, theta, phi


def spherical_to_ijk(r, theta, phi, proj_shape,
                     ln_dist_min, ln_dist_max, n_dists):
    """
    Converts spherical coordinates to pixel coordinates in the dust map grid.

    Args:
        r (tf.Tensor): Tensor of shape (...) representing radial distances.
        theta (tf.Tensor): Tensor of shape (...) representing polar angles (radians).
        phi (tf.Tensor): Tensor of shape (...) representing azimuthal angles (radians).
        proj_shape (tuple or list): Tuple representing the (longitude, latitude) dimensions of the dust map grid.
        ln_dist_min (float): Minimum logarithm of distance.
        ln_dist_max (float): Maximum logarithm of distance.
        n_dists (int): Number of distance bins.

    Returns:
        tf.Tensor: A tensor of shape (..., 3) with pixel coordinates in the order
                   (longitude pixel, latitude pixel, distance pixel).
    """
    i = tf.math.mod(phi / tf.constant(2 * np.pi, dtype=r.dtype), 1)
    j = 1 - theta / tf.constant(np.pi, dtype=r.dtype)
    ij = tf.stack([i, j], axis=-1)
    ij = ij * tf.expand_dims(
        tf.cast(proj_shape, dtype=r.dtype),
        axis=0
    )
    k = (
        (tf.math.log(r) - ln_dist_min)
        / (ln_dist_max - ln_dist_min)
        * (tf.cast(n_dists - 1, dtype=r.dtype))
    )
    return tf.concat([ij, tf.expand_dims(k, axis=1)], axis=-1)


@tf.function
def integrate_dust_along_rays(map_values,
                              ln_dist_min, ln_dist_max,
                              x0, ray_dir,
                              min_ray_dist,
                              max_ray_dist,
                              n_steps):
    """
    Integrates dust density along rays through the dust map.

    Args:
        map_values (tf.Tensor): 3D tensor of dust map densities with shape (lon, lat, dist).
        ln_dist_min (float or tf.Tensor): Minimum logarithm of distance.
        ln_dist_max (float or tf.Tensor): Maximum logarithm of distance.
        x0 (tf.Tensor): Tensor of shape (3,) representing the ray origin in Cartesian coordinates.
        ray_dir (tf.Tensor): Tensor of shape (num_pixels, 3) representing normalized ray direction vectors.
        min_ray_dist (float or tf.Tensor): Minimum distance to integrate along each ray.
        max_ray_dist (float or tf.Tensor): Maximum distance to integrate along each ray.
        n_steps (int): Number of sampling steps along each ray.

    Returns:
        tf.Tensor: A 1D tensor of integrated dust column densities for each ray.
    """
    print('Tracing integrate_dust_along_rays...')
    t_vals = tf.cast(tf.linspace(min_ray_dist, max_ray_dist, n_steps),
                     dtype=ray_dir.dtype)
    sample_xyz = tf.expand_dims(ray_dir, 1) * tf.reshape(t_vals, (1, -1, 1))
    sample_xyz = tf.reshape(sample_xyz, (-1, 3))
    sample_xyz = sample_xyz + tf.expand_dims(x0, axis=0)
    r, theta, phi = xyz_to_spherical(sample_xyz)
    proj_shape, n_dists = tf.split(tf.shape(map_values), [2, 1], axis=0)
    sample_ijk = spherical_to_ijk(
        r, theta, phi,
        proj_shape,
        ln_dist_min, ln_dist_max, n_dists
    )
    sample_values = trilinear_interpolate(
        tf.expand_dims(map_values, -1),
        sample_ijk
    )
    sample_values = tf.reshape(sample_values, (-1, n_steps))
    nan_mask = tf.math.is_nan(sample_values)
    sample_values = tf.where(
        nan_mask,
        tf.constant(0, dtype=sample_values.dtype),
        sample_values
    )
    return tf.reduce_sum(sample_values, axis=-1) * (max_ray_dist / n_steps)


@tf.function
def integrate_dust_along_rays_binned(map_values,
                                     ln_dist_min, ln_dist_max,
                                     x0, ray_dir,
                                     min_ray_dist,
                                     max_ray_dist,
                                     n_steps,
                                     dist_bin_edges):
    """
    Integrates dust density along rays in specified distance bins using cumulative integration.
    
    Args:
        map_values (tf.Tensor): 3D tensor of dust map densities with shape (lon, lat, dist).
        ln_dist_min, ln_dist_max: Minimum/maximum logarithm of distance.
        x0 (tf.Tensor): Ray origin (shape (3,)).
        ray_dir (tf.Tensor): Normalized ray directions (shape (num_pixels, 3)).
        min_ray_dist, max_ray_dist: Integration bounds along the ray.
        n_steps (int): Number of samples along each ray.
        dist_bin_edges (tf.Tensor): 1D tensor of distance bin boundaries (shape (num_bins+1,)).
    
    Returns:
        tf.Tensor: Tensor of shape (num_bins, num_pixels) with the integrated dust density per bin.
    """
    # Create sample positions along each ray.
    t_vals = tf.cast(tf.linspace(min_ray_dist, max_ray_dist, n_steps), dtype=ray_dir.dtype)
    dt = (max_ray_dist - min_ray_dist) / tf.cast(n_steps - 1, ray_dir.dtype)
    
    # Compute sample positions (shape: (num_pixels, n_steps, 3)).
    sample_xyz = tf.expand_dims(ray_dir, 1) * tf.reshape(t_vals, (1, -1, 1))
    sample_xyz = tf.reshape(sample_xyz, (-1, 3)) + tf.expand_dims(x0, axis=0)
    
    # Convert Cartesian to spherical coordinates.
    r, theta, phi = xyz_to_spherical(sample_xyz)
    
    # Map spherical coordinates to index space.
    proj_shape, n_dists = tf.split(tf.shape(map_values), [2, 1], axis=0)
    sample_ijk = spherical_to_ijk(
        r, theta, phi,
        proj_shape,
        ln_dist_min, ln_dist_max, n_dists
    )
    
    # Interpolate values from the dust map.
    sample_values = trilinear_interpolate(tf.expand_dims(map_values, -1), sample_ijk)
    sample_values = tf.reshape(sample_values, (-1, n_steps))
    sample_values = tf.where(tf.math.is_nan(sample_values), tf.zeros_like(sample_values), sample_values)
    
    # Compute the cumulative integral along each ray.
    cumsum = tf.cumsum(sample_values, axis=-1) * dt  # shape: (num_pixels, n_steps)
    
    # Find indices corresponding to bin boundaries.
    # t_vals is shape (n_steps,), dist_bin_edges is shape (num_bins+1,).
    indices = tf.searchsorted(t_vals, dist_bin_edges)
    indices = tf.clip_by_value(indices, 0, n_steps - 1)
    
    # Gather cumulative values at the bin boundaries.
    lower_vals = tf.gather(cumsum, indices[:-1], axis=1)
    upper_vals = tf.gather(cumsum, indices[1:], axis=1)
    
    # The difference yields the integrated value in each bin.
    bin_integrals = upper_vals - lower_vals  # shape: (num_pixels, num_bins)
    
    # Transpose so that shape is (num_bins, num_pixels)
    return bin_integrals#tf.transpose(bin_integrals)


def location_by_name(name, dist, offset_by=None):
    """
    Computes the Cartesian coordinates of a celestial object by its name at a
    specified distance.

    Args:
        name (str): Name of the celestial object (e.g., 'Orion A').
        dist (float): Distance (in kpc) to the object.
        offset_by (tuple, optional): Offset the Galactic sky coordinates
                                     by (position angle, distance).

    Returns:
        np.ndarray: A 1D array of shape (3,) containing the (x, y, z)
                    Cartesian coordinates scaled by dist.
    """
    c = SkyCoord.from_name(name).galactic
    # Add offset to the Galactic coordinates
    if offset_by is not None:
        c = c.directional_offset_by(offset_by[0]*u.deg, offset_by[1]*u.deg)
    # Convert to cartesian coordinates and scale by the distance
    return c.cartesian.xyz.value * dist


def camera_path_from_key_locations(key_locations, closed_loop=False):
    """
    Generates a camera path by fitting cubic splines through a set of key frame locations.

    Args:
        key_locations (np.ndarray): A 2D array of shape (num_keys, 3) containing key frame positions.
        closed_loop (bool, optional): If True, creates a closed (cyclic) camera path. Defaults to False.

    Returns:
        dict: A dictionary with the following keys:
            - 'x_splines': List of CubicSpline objects for each spatial dimension.
            - 'key_locations': The input key_locations array.
            - 'closed_loop': Boolean indicating if the path is closed.
    """
    if closed_loop:
        key_locations = np.vstack([key_locations, key_locations[0]])
        bc_type = 'periodic'
    else:
        bc_type = 'natural'
    
    dx = np.diff(key_locations, axis=0)
    dist_btw_keys = np.linalg.norm(dx, axis=1)
    dist_cumulative = np.insert(np.cumsum(dist_btw_keys), 0, 0)
    t_key = dist_cumulative / dist_cumulative[-1]

    x_splines = [
        CubicSpline(t_key, key_locations[:, i], bc_type=bc_type)
        for i in range(3)
    ]

    return dict(x_splines=x_splines,
                key_locations=key_locations,
                closed_loop=closed_loop)


def smooth_spline(spl, n_frames, closed_loop=False):
    """
    Smooths a given spline by averaging neighboring values and re-fitting a cubic spline.

    Args:
        spl (CubicSpline): The original spline to be smoothed.
        n_frames (int): Number of frames (evaluation points) for the spline.
        closed_loop (bool, optional): If True, treats the spline as periodic. Defaults to False.

    Returns:
        CubicSpline: A new smoothed CubicSpline object.
    """
    t = np.linspace(0, 1, n_frames)
    x_t = spl(t)

    if closed_loop:
        x_t_padded = np.concatenate([x_t[-2:-1], x_t, x_t[1:2]])
        bc_type = 'periodic'
    else:
        x0 = x_t[0] - (x_t[1] - x_t[0])
        x1 = x_t[-1] + (x_t[-1] - x_t[-2])
        x_t_padded = np.concatenate([[x0], x_t, [x1]])
        bc_type = 'natural'
    
    x_t = (x_t_padded[:-2] + x_t_padded[1:-1] + x_t_padded[2:]) / 3.0

    return CubicSpline(t, x_t, bc_type=bc_type)


def smooth_camera_path(camera_path, n_frames):
    """
    Smooths the camera path by applying smoothing to each positional spline.

    Args:
        camera_path (dict): Dictionary containing camera path information (keys: 'x_splines', 'key_locations', 'closed_loop').
        n_frames (int): Number of frames to evaluate the smoothed splines.

    Returns:
        dict: A new camera path dictionary with updated 'x_splines' after smoothing.
    """
    x_splines = [
        smooth_spline(spl, n_frames,
                      closed_loop=camera_path['closed_loop'])
        for spl in camera_path['x_splines']
    ]
    return dict(x_splines=x_splines,
                key_locations=camera_path['key_locations'],
                closed_loop=camera_path['closed_loop'])


def plot_camera_path(camera_path):
    """
    Plots the camera path in three projections: (x,y), (x,z), and (y,z).

    Args:
        camera_path (dict): Dictionary containing camera path data with keys 'x_splines' and optionally 'key_locations'.

    Returns:
        matplotlib.figure.Figure: Figure object containing the plotted camera path.
    """
    x_splines = camera_path['x_splines']

    fig, ax = plt.subplots(1, 3, figsize=(8, 8 / 3))
    projs = [(0, 1), (0, 2), (1, 2)]

    for i, (j, k) in enumerate(projs):
        ax[i].set_aspect('equal')

        if 'key_locations' in camera_path:
            key_locations = camera_path['key_locations']
            ax[i].plot(
                key_locations[:, j],
                key_locations[:, k],
                'ro', label='keys'
            )

        t = np.linspace(0, 1, 100)
        ax[i].plot(x_splines[j](t), x_splines[k](t), 'c-', label='path')

        ax[i].set_xlabel('xyz'[j])
        ax[i].set_ylabel('xyz'[k])
        if i == 0:
            ax[i].legend()
        ax[i].grid(True, alpha=0.2)
        ax[i].grid(True, which='minor', alpha=0.05)

    fig.tight_layout()

    return fig


def direction_to_lonlat(dx, dy, dz):
    """
    Converts a 3D direction vector into longitude and latitude angles.

    Args:
        dx (float or np.ndarray): x-component of the direction vector.
        dy (float or np.ndarray): y-component of the direction vector.
        dz (float or np.ndarray): z-component of the direction vector.

    Returns:
        tuple: A tuple (lon, lat) where both angles are in degrees.
    """
    lon = np.degrees(np.arctan2(dy, dx))
    lat = np.degrees(np.arctan2(dz, np.sqrt(dx**2 + dy**2)))
    return lon, lat


def generate_orientations_from_tangents(camera_path, n_frames, n_smooth=0):
    """
    Generates camera orientations based on the tangents of the camera path.

    Args:
        camera_path (dict): Dictionary containing camera path information with keys 'x_splines', 'key_locations', and 'closed_loop'.
        n_frames (int): Number of frames for which to generate orientations.
        n_smooth (int, optional): Number of times to smooth the tangent splines. Defaults to 0.

    Returns:
        dict: A dictionary containing:
            - 'orient_splines': List of spline functions for the orientation components.
            - 't': Array of time values corresponding to each frame.
            - 'lon': Array of computed longitude angles (degrees).
            - 'lat': Array of computed latitude angles (degrees).
            - 'roll': Array of roll angles (degrees, set to zero).
            - 'closed_loop': Boolean indicating whether the orientation path is closed.
    """
    x_splines = camera_path['x_splines']

    t = np.linspace(0, 1, n_frames)
    v_splines = [spl.derivative() for spl in x_splines]

    for i in range(n_smooth):
        v_splines = [
            smooth_spline(spl, n_frames,
                          closed_loop=camera_path['closed_loop'])
            for spl in v_splines
        ]
    v_t = np.array([spl(t) for spl in v_splines])
    v_t = v_t / np.linalg.norm(v_t, axis=0)

    lon, lat = direction_to_lonlat(*v_t)
    roll = np.zeros_like(lon)

    return dict(orient_splines=v_splines, t=t,
                lon=lon, lat=lat, roll=roll,
                closed_loop=camera_path['closed_loop'])


def generate_orientations_staring_at_location(camera_path, x_stare, n_frames):
    """
    Generates camera orientations so that the camera continuously faces a specific target location.

    Args:
        camera_path (dict): Dictionary containing camera path information (key 'x_splines' and 'closed_loop').
        x_stare (array-like): A 3-element array-like object representing the target Cartesian coordinates.
        n_frames (int): Number of frames for which to generate orientations.

    Returns:
        dict: A dictionary containing:
            - 'orient_splines': List of spline functions for the unit-vector components pointing toward x_stare.
            - 't': Array of time values corresponding to each frame.
            - 'lon': Array of computed longitude angles (degrees).
            - 'lat': Array of computed latitude angles (degrees).
            - 'roll': Array of roll angles (degrees, set to zero).
            - 'closed_loop': Boolean indicating if the orientation is a closed loop.
    """
    x_splines = camera_path['x_splines']
    t = np.linspace(0, 1, n_frames)
    dxyz = [x - spl(t) for x, spl in zip(x_stare, x_splines)]
    norm = np.sqrt(dxyz[0]**2 + dxyz[1]**2 + dxyz[2]**2)
    dxyz = [x / norm for x in dxyz]

    if camera_path['closed_loop']:
        for i, d in enumerate(dxyz):
            dxyz[i] = np.hstack([d, d[0]])
        dt = t[-1] - t[-2]
        t = np.hstack([t, t[-1] + dt])
        bc_type = 'periodic'
    else:
        bc_type = 'natural'
    dxyz_splines = [CubicSpline(t, d, bc_type=bc_type) for d in dxyz]

    lon, lat = direction_to_lonlat(*dxyz)
    roll = np.zeros_like(lon)

    return dict(orient_splines=dxyz_splines, t=t,
                lon=lon, lat=lat, roll=roll,
                closed_loop=camera_path['closed_loop'])


def average_orientations(orient0, orient1, weight1=None):
    """
    Averages two sets of camera orientations.

    Args:
        orient0 (dict): First orientation dictionary containing keys 'orient_splines', 't', and 'closed_loop'.
        orient1 (dict): Second orientation dictionary with a similar structure.
        weight1 (np.ndarray, optional): Array of weights for orient1. If None, equal weighting is used.

    Returns:
        dict: A dictionary containing the averaged orientation with keys:
            - 'orient_splines': List of spline functions for the averaged unit-vector components.
            - 't': Array of time values.
            - 'lon': Array of averaged longitude angles (degrees).
            - 'lat': Array of averaged latitude angles (degrees).
            - 'roll': Array of roll angles (degrees, set to zero).
            - 'closed_loop': Boolean indicating whether the orientation path is closed.
    """
    t = orient0['t']

    if weight1 is None:
        w = np.ones_like(t)
    else:
        w = weight1

    d0 = [spl(t) for spl in orient0['orient_splines']]
    d1 = [spl(t) for spl in orient1['orient_splines']]
    dxyz = d0 + w * d1

    norm = np.sqrt(dxyz[0]**2 + dxyz[1]**2 + dxyz[2]**2)
    dxyz = [x / norm for x in dxyz]

    if orient0['closed_loop']:
        for d in dxyz:
            d.append(d[0])
        bc_type = 'periodic'
    else:
        bc_type = 'natural'
    dxyz_splines = [CubicSpline(t, d, bc_type=bc_type) for d in dxyz]

    lon, lat = direction_to_lonlat(*dxyz)
    roll = np.zeros_like(lon)

    return dict(orient_splines=dxyz_splines, t=t,
                lon=lon, lat=lat, roll=roll,
                closed_loop=orient0['closed_loop'])


def camera_path_from_txyz(t, x, y, z, closed_loop=False):
    """
    Generates a camera path from given time and spatial coordinates.

    Args:
        t (array-like): 1D array of time values.
        x (array-like): 1D array of x coordinates.
        y (array-like): 1D array of y coordinates.
        z (array-like): 1D array of z coordinates.
        closed_loop (bool, optional): If True, the path is closed (cyclic) by appending the first point. Defaults to False.

    Returns:
        dict: A dictionary with keys:
            - 'x_splines': List of CubicSpline objects for each spatial dimension.
            - 'closed_loop': Boolean indicating if the path is closed.
    """
    xyz = np.stack([x, y, z], axis=1)

    if closed_loop:
        xyz = np.concatenate([xyz, xyz[:1]], axis=0)
        dt = t[-1] - t[-2]
        t = np.hstack([t, t[-1] + dt])
        bc_type = 'periodic'
    else:
        bc_type = 'natural'

    x_splines = [CubicSpline(t, x, bc_type=bc_type) for x in xyz.T]

    return dict(x_splines=x_splines,
                closed_loop=closed_loop)


def batch_apply_tf(f, batch_size, *args,
                   function=False, progress=False, numpy=False):
    """
    Applies a TensorFlow function in batches to avoid memory issues and concatenates the results.

    Args:
        f (function): The function to apply, which should accept the given arguments.
        batch_size (int): Size of each batch.
        *args (array-like): Input arrays/tensors, each batched along the first axis.
        function (bool, optional): If True, wraps the function f with tf.function. Defaults to False.
        progress (bool, optional): If True, displays a progress bar via tqdm. Defaults to False.
        numpy (bool, optional): If True, converts outputs to numpy arrays. Defaults to False.

    Returns:
        The concatenated result from applying f to each batch. If f returns multiple tensors,
        returns a tuple of concatenated tensors.
    """
    res = []
    def f_batch(*x):
        return f(*x)
    if function:
        f_batch = tf.function(f_batch)
    iterator = range(0, len(args[0]), batch_size)
    if progress:
        iterator = tqdm(iterator)
    for i in iterator:
        batch = [a[i:i+batch_size] for a in args]
        res_batch = f_batch(*batch)
        if numpy:
            if isinstance(res_batch, tuple):
                res_batch = tuple((r.numpy() for r in res_batch))
            else:
                res_batch = res_batch.numpy()
        res.append(res_batch)
    
    if numpy:
        f_concat = np.concatenate
    else:
        f_concat = tf.concat

    if isinstance(res[0], tuple):
        n = len(res[0])
        res = tuple((
            f_concat([r[i] for r in res], axis=0)
            for i in range(n)
        ))
    else:
        res = f_concat(res, axis=0)

    return res


def batched_ray_integration(batch_size, map_values,
                            ln_dist_min, ln_dist_max,
                            x0, ray_dir,
                            min_ray_dist,
                            max_ray_dist,
                            n_steps,
                            dist_bin_edges=None):
    """
    Performs batched integration of dust density along rays.

    Args:
        batch_size (int): Number of rays per batch.
        map_values (tf.Tensor): 3D tensor of dust map densities.
        ln_dist_min (tf.Tensor or float): Minimum logarithm of distance.
        ln_dist_max (tf.Tensor or float): Maximum logarithm of distance.
        x0 (tf.Tensor): Tensor of shape (3,) representing the camera position.
        ray_dir (tf.Tensor): Tensor of shape (num_rays, 3) representing ray direction vectors.
        min_ray_dist (tf.Tensor or float): Minimum integration distance along each ray.
        max_ray_dist (tf.Tensor or float): Maximum integration distance along each ray.
        n_steps (int): Number of integration steps along each ray.
        dist_bin_edges (tf.Tensor, optional): Distance bin edges for the integrated values.

    Returns:
        tf.Tensor: A 1D tensor of integrated dust column densities for each ray.
    """
    if dist_bin_edges is None:
        def f_int_batch(ray_dir_batch):
            return integrate_dust_along_rays(
                map_values,
                ln_dist_min, ln_dist_max,
                x0, ray_dir_batch,
                min_ray_dist, max_ray_dist, n_steps
            )
    else:
        def f_int_batch(ray_dir_batch):
            return integrate_dust_along_rays_binned(
                map_values,
                ln_dist_min, ln_dist_max,
                x0, ray_dir_batch,
                min_ray_dist, max_ray_dist, n_steps,
                dist_bin_edges
            )
    return batch_apply_tf(f_int_batch, batch_size, ray_dir)


def flythrough_camera(n_frames):
    """
    Generates a flythrough camera path and corresponding orientations based on key celestial locations.

    Args:
        n_frames (int): Number of frames for the camera path and orientations.

    Returns:
        tuple: A tuple (camera_path, camera_orientations) where:
            - camera_path (dict): Dictionary containing camera path splines and key locations.
            - camera_orientations (dict): Dictionary containing orientation splines and Euler angles.
    """
    key_locations = np.array([
        [0, 0, 0],
        location_by_name('Orion A', 0.4),
        location_by_name('Monoceros R2', 0.8),
        location_by_name('rho Ophiuchus', 0.2),
        location_by_name('Coalsack', 1.85)
    ])

    camera_path = camera_path_from_key_locations(key_locations,
                                                 closed_loop=True)
    for i in range(10):
        camera_path = smooth_camera_path(camera_path, 100)
    camera_orientations = generate_orientations_from_tangents(
        camera_path,
        n_frames,
        n_smooth=0
    )

    return camera_path, camera_orientations


def solar_orbit_camera(n_frames, radius, lon0, lat0, dist0, x0=[0,0,0]):
    """
    Generates a solar orbit camera path and corresponding orientations.

    Args:
        n_frames (int): Number of frames for the camera path and orientations.
        radius (float): Radius of the circular orbit around the sun.
        lon0 (float): Longitude (in degrees) of the target location to stare at.
        lat0 (float): Latitude (in degrees) of the target location to stare at.
        dist0 (float): Distance scaling factor for the target location (typically in kpc).
        x0 (array of floats): Origin of the orbit. Defaults to [0,0,0], the Solar coords.

    Returns:
        tuple: A tuple (camera_path, camera_orientations) where:
            - camera_path (dict): Dictionary containing camera path splines.
            - camera_orientations (dict): Dictionary containing orientation splines and Euler angles.
    """
    t = np.linspace(0, 1, n_frames)
    
    theta = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    x = np.cos(theta) * radius + x0[0]
    y = np.sin(theta) * radius + x0[1]
    z = np.zeros_like(x) + x0[2]
    camera_path = camera_path_from_txyz(t, x, y, z, closed_loop=True)

    l = np.radians(lon0)
    b = np.radians(lat0)
    x_stare = dist0 * np.array([
        np.cos(l) * np.cos(b),
        np.sin(l) * np.cos(b),
        np.sin(b)
    ])
    camera_orientations = generate_orientations_staring_at_location(
        camera_path, x_stare, n_frames
    )

    return camera_path, camera_orientations


def vertical_bobbing_camera(n_frames, z_max, lon0, lat0, dist0):
    """
    Generates a vertical bobbing camera path and corresponding orientations.

    Args:
        n_frames (int): Number of frames for the camera path and orientations.
        z_max (float): Maximum vertical displacement.
        lon0 (float): Longitude (in degrees) of the target location to stare at.
        lat0 (float): Latitude (in degrees) of the target location to stare at.
        dist0 (float): Distance scaling factor for the target location (typically in kpc).

    Returns:
        tuple: A tuple (camera_path, camera_orientations) where:
            - camera_path (dict): Dictionary containing camera path splines.
            - camera_orientations (dict): Dictionary containing orientation splines and Euler angles.
    """
    t = np.linspace(0, 1, n_frames)
    
    theta = np.linspace(0, np.pi, n_frames, endpoint=True)
    z = np.cos(theta) * z_max
    x = np.zeros_like(z)
    y = np.zeros_like(z)
    camera_path = camera_path_from_txyz(t, x, y, z, closed_loop=False)

    l = np.radians(lon0)
    b = np.radians(lat0)
    x_stare = dist0 * np.array([
        np.cos(l) * np.cos(b),
        np.sin(l) * np.cos(b),
        np.sin(b)
    ])
    camera_orientations = generate_orientations_staring_at_location(
        camera_path, x_stare, n_frames
    )

    return camera_path, camera_orientations


def save_image(img, fname, subtract_min=False):
    if fname.endswith('png'):
        # Convert the image array to PNG
        if subtract_min:
            img = img - np.nanmin(img)
        img = (255 * img / np.nanmax(img)).astype('u1')
        img = Image.fromarray(img.T, mode='L')
        img.save(fname)
    elif fname.endswith('npy'):
        # Save as numpy *.npy format
        np.save(fname, img)


def create_labels(dpi=600, scaling=0.8):
    # Define the labels to be rendered
    labels = [
        {'xyz':[0, 0, 0], 'text':r'$\odot$'},
        {'xyz':[8.3, 0, 0], 'text':r'Galactic Center'},
        {'xyz':location_by_name('Orion A',0.4,offset_by=(180,2.7)),
         'text':'Ori A'},
        {'xyz':location_by_name('Mon R2',0.8,offset_by=(-90,3)),
         'text':r'Mon R2'},
        {'xyz':location_by_name('lambda Orionis',0.4),
         'text':r'$\lambda$ Ori'},
        {'xyz':location_by_name('IC 348',0.25,offset_by=(-90,6)),
         'text':r'Perseus'},
        {'xyz':location_by_name('Taurus',0.22,offset_by=(90,7)),
         'text':r'Taurus'},
        {'xyz':location_by_name('rho Ophiuchus',0.25,offset_by=(-90,8)),
         'text':r'$\rho$ Oph'},
        {'xyz':location_by_name('Pipe nebula',0.25,offset_by=(-90,8)),
         'text':r'Pipe'},
    ]

    # Render the labels
    for l in labels:
        img = render_text_to_image(l['text'], dpi=dpi, color='g')
        l['img'] = img
        l['scale'] = scaling * img.size[0] / dpi

    return labels


def save_image_stack_to_npy(images, fname):
    """
    Stacks a list of Image objects along a new first axis
    and saves the resulting array to a .npy file.

    Args:
        images (list of PIL.Image.Image): List of Pillow images.
        filename (str): Path to the output .npy file.
    """
    stacked_array = np.stack([np.array(img) for img in images], axis=0)
    np.save(fname, stacked_array)


def main():
    """
    Main function to generate a volume-rendered dust map video.

    The function:
        1. Defines image and camera parameters.
        2. Generates a camera path and orientations using the vertical bobbing model.
        3. Plots and saves the camera path.
        4. Loads labels that will be embedded at specific locations.
        5. Loads a dust map from a file.
        6. Renders labels on each frame.
        7. Renders frames by integrating dust along camera rays.
        8. Saves the data necessary to composite each frame into a .npz file.

    Returns:
        int: Exit status code (0 indicates successful execution).
    """
    img_shape = (4*256, 4*192)  # (width, height) in pixels
    fov = 90.            # Field of view in degrees
    max_ray_dist = 3.    # Maximum ray integration distance, in kpc
    n_ray_steps = 300    # Number of distance steps to take along each ray
    batch_size = 32*32*4 # How many rays to integrate at once (memory usage)
    n_frames = 400       # Number of frames in video
    gen_every_n_frames = 1 # If >1, only every nth frame will be generated
    label_scaling = 0.5  # Overall scaling to apply to label sizes
    frame_fname = 'frames/frame_data_{frame:04d}.npz'

    # Generate the camera path and orientations
    print('Generating camera path ...')
    # stare_coords = SkyCoord.from_name('lambda Orionis').galactic
    # lon0, lat0, dist0 = stare_coords.l.deg, stare_coords.b.deg, 1.0
    # lon0, lat0, dist0 = 0., 0., 1.
    # camera_path, camera_orientations = vertical_bobbing_camera(
    #     n_frames, 0.5,
    #     lon0, lat0, dist0
    # )
    # Alternatively, use solar_orbit_camera:
    # camera_path, camera_orientations = solar_orbit_camera(
    #    n_frames, 0.025,
    #    lon0, lat0, dist0
    # )
    # Alternatively, use the flythrough_camera:
    camera_path, camera_orientations = flythrough_camera(n_frames)

    # Plot the camera path
    fig = plot_camera_path(camera_path)
    fig.savefig('frames/camera_path.svg')
    plt.close(fig)

    # Generate the labels
    print('Generating labels ...')
    labels = create_labels(dpi=img_shape[0]*600/512, scaling=label_scaling)

    # Load the 3D dust map
    print('Loading dust map ...')

    # fname = 'data/zhang_green_2025_cube.npz'
    # # map_values, ln_dist_edges = load_zhang_green_2025(4*512, 4*256,
    # #                                                   return_tensor=False)
    # # save_dustmap_cube(fname, map_values, ln_dist_edges)
    # map_values, ln_dist_edges = load_dustmap_cube(fname)
    # map_values = [tf.squeeze(v) for v in tf.split(map_values, 2, axis=0)]
    # print(ln_dist_edges)
    # print(map_values[0].shape)
    # print(map_values)

    fname = 'data/bayestar_decaps_cube.npz'
    # # Uncomment the following two lines to generate the NPZ file:
    # map_values, ln_dist_edges = load_bayestar_decaps(4*512, 4*256)
    # save_dustmap_cube(fname, map_values, ln_dist_edges)
    map_values, ln_dist_edges = load_dustmap_cube(fname)
    map_values = [map_values]
    # print(map_values)
    # print(ln_dist_edges)

    # fname = 'data/zhang_green_2025_cube.npz'
    # map_values, ln_dist_edges_zg25 = load_dustmap_cube(fname)
    # map_values_zg25 = [tf.squeeze(v) for v in tf.split(map_values, 2, axis=0)]
    # fname = 'data/bayestar_decaps_cube.npz'
    # map_values_bd, ln_dist_edges_bd = load_dustmap_cube(fname)
    # map_values_bd = [map_values_bd]
    # map_values, ln_dist_edges = combine_zg25_bayestar(
    #     map_values_zg25, ln_dist_edges_zg25,
    #     map_values_bd, ln_dist_edges_bd
    # )

    # # Plot map slices
    # for i,img in enumerate(tf.unstack(map_values, axis=2)):
    #     save_image(img.numpy(), f'frames/map_slice_{i:03d}.png')

    # Render the video frames
    print('Rendering frames ...')
    for i in tqdm(range(0, n_frames, gen_every_n_frames)):
        # Load the camera properties. The camera ultimately consists of two
        # things:
        #   1. The camera location, x0.
        #   2. A 2D array of unit vectors (camera_rays), each representing
        #      the direction that corresponds to one pixel in the image.
        t = camera_orientations['t'][i]
        x0 = [spl(t) for spl in camera_path['x_splines']]
        x0 = tf.constant(x0, dtype=tf.float32)
        camera_rot = (
            camera_orientations['lon'][i],
            camera_orientations['lat'][i],
            camera_orientations['roll'][i]
        )

        # Project the labels onto the camera, and determine
        # the distance to each label
        label_proj = []
        label_distance = []
        for l in labels:
            center,target_width,r = calc_label_center_and_width(
                l['xyz'], x0.numpy(),
                camera_rot, fov,
                img_shape
            )
            if r <= 0: # Don't render labels behind camera
                continue
            img = paste_label_on_canvas(
                l['img'], center,
                target_width*l['scale'],
                1.0, # label alpha
                img_shape,
                background_rgba=(0,0,0,0)
            )
            label_proj.append(img)
            label_distance.append(r)
        
        # Sort the label images by distance
        idx_sort = np.argsort(label_distance)
        label_proj = [label_proj[i] for i in idx_sort]
        label_distance = np.sort(label_distance)
        if len(label_proj) == 0:
            stacked_labels = np.array([])
        else:
            stacked_labels = np.stack([
                np.array(img) for img in label_proj
            ], axis=0)

        # # Save the composited label images
        # img = alpha_composite_many(label_proj)
        # img.save(f'frames/labels_{i:04d}.png')

        # Determine the distance ranges in which to volume render.
        # There should be a break at the distance to each label,
        # so that the labels can be embedded into the volume rendering.
        int_dist_edges = np.hstack([0, label_distance])
        idx = np.sum(int_dist_edges < max_ray_dist)
        int_dist_edges = int_dist_edges[:idx]
        int_dist_edges = np.hstack([int_dist_edges, max_ray_dist])

        # Generate the camera-ray unit vectors
        R = get_rotation_matrix(*camera_rot)
        camera_rays = gnomonic_projection(fov, img_shape)
        camera_rays = tf.einsum(
            'ij,...j->...i',
            R, camera_rays,
            optimize='optimal'
        )

        # Generate the image by integrating each ray
        img_list = [
            batched_ray_integration(
                batch_size,
                v,
                tf.constant(ln_dist_edges[0], dtype=tf.float32),
                tf.constant(ln_dist_edges[-1], dtype=tf.float32),
                x0, tf.reshape(camera_rays, (-1, 3)),
                tf.constant(0., dtype=tf.float32),
                tf.constant(max_ray_dist, dtype=tf.float32),
                n_ray_steps,
                dist_bin_edges=tf.constant(int_dist_edges, dtype=tf.float32)
            )
            for v in map_values
        ]

        # Stack the volume-rendered images into a single array
        volrender_data = []
        for img in img_list:
            img = img.numpy()
            img.shape = img_shape + (-1,)
            # Rotate last axis to first position
            img = np.moveaxis(img, -1, 0)
            volrender_data.append(img)
        volrender_data = np.stack(volrender_data, axis=1)

        # Store all the frame data in a single .npz file
        frame_data = dict(
            labels=stacked_labels,   # shape = (n_labels, w, h)
            volrender=volrender_data # shape = (n_dist_bins, n_channels, w, h)
        )
        np.savez(
            frame_fname.format(frame=i),
            **frame_data
        )
    
    return 0


if __name__ == '__main__':
    main()
