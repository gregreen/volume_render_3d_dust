#!/usr/bin/env bash

# Generates a video by first running through the frames once in the
# forward direction. This is the most straightforward type of video.
# 
# Example usage:
#   bash gen_ping_pong_video.sh frame_base_filename output_video_filename.mp4
# 
# This example with use frames with the filename pattern:
#   frame_base_filename_%04d.png

frames_base_fname=$1
video_fname=$2

ffmpeg -y -r 25 -i "${frames_base_fname}_%04d.png" \
	-c:v libx264 -crf 20 \
    -pix_fmt yuv420p -profile:v baseline -movflags +faststart -r 25 \
	${video_fname}