import numpy as np

import tensorflow as tf
import tensorflow_graphics as tfg
from tensorflow_graphics.math.interpolation.trilinear import interpolate as trilinear_interpolate
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d, quaternion

from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
plt.style.use('dark_background')

from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy_healpix

from scipy.interpolate import CubicSpline

from tqdm.auto import tqdm


def save_dustmap_cube(fname, map_values, ln_dist_edges):
    """
    Saves a 3D dust map in a Cartesian sky projection to an NPZ file.

    Args:
        fname (str): Filename (with path) where the dust map will be saved.
        map_values (np.ndarray or tf.Tensor): A 3D array/tensor containing dust map density values.
        ln_dist_edges (np.ndarray or tf.Tensor): A 1D array/tensor containing logarithm of distance bin edges.

    Returns:
        None
    """
    if isinstance(map_values, tf.Tensor):
        map_values = map_values.numpy()
    if isinstance(ln_dist_edges, tf.Tensor):
        ln_dist_edges = ln_dist_edges.numpy()
    np.savez(fname, map_values=map_values, ln_dist_edges=ln_dist_edges)


def load_dustmap_cube(fname):
    """
    Loads a 3D dust map from an NPZ file that has been converted to a Cartesian sky projection.

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
                              max_ray_dist, n_steps):
    """
    Integrates dust density along rays through the dust map.

    Args:
        map_values (tf.Tensor): 3D tensor of dust map densities with shape (lon, lat, dist).
        ln_dist_min (float or tf.Tensor): Minimum logarithm of distance.
        ln_dist_max (float or tf.Tensor): Maximum logarithm of distance.
        x0 (tf.Tensor): Tensor of shape (3,) representing the ray origin in Cartesian coordinates.
        ray_dir (tf.Tensor): Tensor of shape (num_pixels, 3) representing normalized ray direction vectors.
        max_ray_dist (float or tf.Tensor): Maximum distance to integrate along each ray.
        n_steps (int): Number of sampling steps along each ray.

    Returns:
        tf.Tensor: A 1D tensor of integrated dust column densities for each ray.
    """
    print('Tracing integrate_dust_along_rays...')
    t_vals = tf.cast(tf.linspace(0., max_ray_dist, n_steps),
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


def interpolate_rotation_matrices(R0, R1, t):
    """
    Interpolates between two 3D rotation matrices using SLERP (spherical linear interpolation) via quaternions.

    Args:
        R0 (tf.Tensor): A 3x3 rotation matrix representing the initial orientation.
        R1 (tf.Tensor): A 3x3 rotation matrix representing the final orientation.
        t (float or tf.Tensor): Interpolation factor in the range [0, 1].

    Returns:
        tf.Tensor: A 3x3 rotation matrix representing the interpolated rotation.
    """
    q0 = quaternion.from_rotation_matrix(R0)
    q1 = quaternion.from_rotation_matrix(R1)
    dot = tf.reduce_sum(q0 * q1)
    q1 = q1 * tf.sign(dot)
    dot = tf.reduce_sum(q0 * q1)
    theta = tf.acos(tf.clip_by_value(dot, -1.0, 1.0))
    sin_theta = tf.sin(theta)
    q = (tf.sin((1 - t) * theta) / sin_theta) * q0 + (tf.sin(t * theta) / sin_theta) * q1
    return rotation_matrix_3d.from_quaternion(q)


def get_rotation_matrix(lon, lat, roll):
    """
    Computes a rotation matrix based on given Euler angles (longitude, latitude, and roll).

    Args:
        lon (float): Longitude angle in degrees.
        lat (float): Latitude angle in degrees.
        roll (float): Roll angle in degrees.

    Returns:
        tf.Tensor: A 3x3 rotation matrix computed from the input angles.
    """
    R2 = rotation_matrix_3d.from_axis_angle(
        tf.cast([0., 0., 1.], tf.float32),
        tf.cast([np.radians(lon - 90)], tf.float32)
    )
    R1 = rotation_matrix_3d.from_axis_angle(
        tf.cast([1., 0., 0.], tf.float32),
        tf.cast([np.radians(lat - 90)], tf.float32)
    )
    R0 = rotation_matrix_3d.from_axis_angle(
        tf.cast([0., 0., 1.], tf.float32),
        tf.cast([-np.radians(roll)], tf.float32)
    )
    R = tf.einsum('ij,jk,kl->il', R2, R1, R0)
    return R


def location_by_name(name, dist):
    """
    Computes the Cartesian coordinates of a celestial object by its name at a specified distance.

    Args:
        name (str): Name of the celestial object (e.g., 'Orion A').
        dist (float): Distance scaling factor (e.g., in kpc) to apply to the coordinates.

    Returns:
        np.ndarray: A 1D array of shape (3,) containing the (x, y, z) Cartesian coordinates scaled by dist.
    """
    c = SkyCoord.from_name(name)
    return c.galactic.cartesian.xyz.value * dist


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
                            max_ray_dist, n_steps):
    """
    Performs batched integration of dust density along rays.

    Args:
        batch_size (int): Number of rays per batch.
        map_values (tf.Tensor): 3D tensor of dust map densities.
        ln_dist_min (tf.Tensor or float): Minimum logarithm of distance.
        ln_dist_max (tf.Tensor or float): Maximum logarithm of distance.
        x0 (tf.Tensor): Tensor of shape (3,) representing the camera position.
        ray_dir (tf.Tensor): Tensor of shape (num_rays, 3) representing ray direction vectors.
        max_ray_dist (tf.Tensor or float): Maximum integration distance along each ray.
        n_steps (int): Number of integration steps along each ray.

    Returns:
        tf.Tensor: A 1D tensor of integrated dust column densities for each ray.
    """
    def f_int_batch(ray_dir_batch):
        return integrate_dust_along_rays(
            map_values,
            ln_dist_min, ln_dist_max,
            x0, ray_dir_batch,
            max_ray_dist, n_steps
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


def solar_orbit_camera(n_frames, radius, lon0, lat0, dist0):
    """
    Generates a solar orbit camera path and corresponding orientations.

    Args:
        n_frames (int): Number of frames for the camera path and orientations.
        radius (float): Radius of the circular orbit around the sun.
        lon0 (float): Longitude (in degrees) of the target location to stare at.
        lat0 (float): Latitude (in degrees) of the target location to stare at.
        dist0 (float): Distance scaling factor for the target location (typically in kpc).

    Returns:
        tuple: A tuple (camera_path, camera_orientations) where:
            - camera_path (dict): Dictionary containing camera path splines.
            - camera_orientations (dict): Dictionary containing orientation splines and Euler angles.
    """
    t = np.linspace(0, 1, n_frames)
    
    theta = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    x = np.cos(theta) * radius
    y = np.sin(theta) * radius
    z = np.zeros_like(x)
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


def main():
    """
    Main function to generate a volume-rendered dust map video.

    The function:
        1. Defines image and camera parameters.
        2. Generates a camera path and orientations using the vertical bobbing model.
        3. Plots and saves the camera path.
        4. Loads a dust map from a file.
        5. Renders frames by integrating dust along camera rays.
        6. Saves each frame as a PNG image.

    Returns:
        int: Exit status code (0 indicates successful execution).
    """
    img_shape = (4*256, 4*192)  # (width, height) in pixels
    fov = 90.          # Field of view in degrees
    max_ray_dist = 5.  # Maximum ray integration distance, in kpc
    n_ray_steps = 1000 # Number of distance steps to take along each ray
    batch_size = 32*32 # How many rays to integrate at once (memory usage)
    n_frames = 125     # Number of frames in video
    gen_every_n_frames = 1 # If >1, only every nth frame will be generated

    # Generate the camera path and orientations
    print('Generating camera path ...')
    lon0, lat0, dist0 = 0., 0., 1.
    camera_path, camera_orientations = vertical_bobbing_camera(
        n_frames, 0.5,
        lon0, lat0, dist0
    )
    # Alternatively, use solar_orbit_camera:
    # camera_path, camera_orientations = solar_orbit_camera(
    #    n_frames, 0.025,
    #    lon0, lat0, dist0
    # )
    # Alternatively, use the flythrough_camera:
    # camera_path, camera_orientations = flythrough_camera(n_frames)

    # Plot the camera path
    fig = plot_camera_path(camera_path)
    fig.savefig('frames/camera_path.svg')
    plt.close(fig)

    # Load the 3D dust map
    print('Loading dust map ...')
    fname = 'data/bayestar_decaps_cube.npz'
    # Uncomment one of the following as needed:
    # map_values, ln_dist_edges = load_bayestar_decaps(4*512, 4*256)
    # save_dustmap_cube(fname, map_values, ln_dist_edges)
    map_values, ln_dist_edges = load_dustmap_cube(fname)

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
        R = get_rotation_matrix(
            camera_orientations['lon'][i],
            camera_orientations['lat'][i],
            camera_orientations['roll'][i]
        )
        camera_rays = gnomonic_projection(fov, img_shape)
        camera_rays = tf.einsum('ij,...j->...i', R, camera_rays, optimize='optimal')

        # Generate the image by integrating each ray
        col_density = batched_ray_integration(
            batch_size,
            map_values,
            tf.constant(ln_dist_edges[0], dtype=tf.float32),
            tf.constant(ln_dist_edges[-1], dtype=tf.float32),
            x0, tf.reshape(camera_rays, (-1, 3)),
            tf.constant(max_ray_dist, dtype=tf.float32),
            n_ray_steps
        )

        # Convert the image array to PNG
        img = col_density.numpy().reshape(img_shape)
        img = img - np.nanmin(img)
        img = (255 * img / np.max(img)).astype('u1')
        img = Image.fromarray(img.T, mode='L')
        img.save(f'frames/dust_video_{i:04d}.png')
    
    return 0

# ffmpeg -y -r 25 -i frames/dust_video_%04d.png -c:v libx264 -crf 25 -pix_fmt yuv420p -profile:v baseline -movflags +faststart -r 25 videos/dust_video.mp4

if __name__ == '__main__':
    main()
