#!/usr/bin/env bash

# Generates a video by first running through the frames in the forward
# direction, and then running through them in the reverse direction. This is
# useful for the "vertical bobbing" videos, for example.
# 
# Example usage:
#   bash gen_ping_pong_video.sh frame_base_filename output_video_filename.mp4
# 
# This example with use frames with the filename pattern:
#   frame_base_filename_%04d.png

frames_base_fname=$1
video_fname=$2

ffmpeg -y -r 25 -i "${frames_base_fname}_%04d.png" \
	-filter_complex "[0:v]split=2[orig][tmp]; \
	[tmp]trim=start_frame=1:end_frame=249,setpts=PTS-STARTPTS,reverse,setpts=PTS-STARTPTS[rev]; \
	[orig][rev]concat=n=2:v=1:a=0" \
	-c:v libx264 -crf 20 \
	-pix_fmt yuv420p -profile:v baseline -movflags +faststart -r 25 \
	${video_fname}

