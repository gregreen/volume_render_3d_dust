Volume rendering of 3D dust maps
================================

Volume render 3D dust maps provided by the `dustmaps` package.

The script `render_dust_3d.py` contains various functions that can be used to
volume-render 3D dust maps. The key elements that go into the rendering are:

  1. A 3D dust map, reprojected to a Cartesian sky projection.
  2. A set of camera positions and orientations.
  3. A set of camera properties (field of view, resolution, etc.).

The `main()` function contains an example of how to generate the frames of a
volume-rendered video. The helper functions can be used to generate different
camera paths or to load different 3D dust maps.

After the frames have been generated, `ffmpeg` can be used to turn them into
a video. Two scripts are provided to do this:

  1. `gen_ping_pong_video.sh`: Generates a video by first running through the
     frames in the forward direction, and then running through them in the
     reverse direction. This is useful for the "vertical bobbing" videos, for
     example.
  2. `gen_forward_video.sh`: Generates a video by running through the frames
     in the forward direction. This is the most straightforward type of
     video.