# Task 3: Video-Based Object Detection for Traffic Safety

Hey there! This is my third task for the internship, where I explored using object detection on videos.

## My Chosen Project Idea:
I decided to work on **Traffic Flow Analysis and Pedestrian Safety Monitoring**.

## Why I Picked This:
This project feels really useful! Imagine helping cities understand traffic better, making pedestrian crossings safer, and generally improving how our streets work. It's a direct way to use AI to make daily life a bit smoother and safer for everyone, which feels like a good fit for this internship's goals.

## How I Did It:
My approach was to use a YOLOv8 model for segmentation. Think of it like this:
1.  I fed it a video of traffic.
2.  The model then looks at each single frame (like a picture) in that video.
3.  For each frame, it draws boxes around things it recognizes (people, cars, buses, bikes) and even colors in their shapes (segmentation).
4.  Finally, it stitches all those processed frames back together to create a new video with all the detections visually overlaid.


- `video_analyzer.py`: This is the Python script I wrote that handles everything – from loading the video to running the detection and saving the output.

## My Input:
I used a 5-second video clip, `input_video/traffic_footage_5s.mp4`, which shows typical city traffic and people moving around.

## My Output:
The script created an output video called `traffic_footage_5s.avi`. This video clearly shows all the cars, people, and other objects that the model identified and segmented.

**Why no video file here?**
Just a heads-up, this output video is around 55 MB, and GitHub has a file size limit of 25 MB for direct uploads. So, I can't put the actual video file here. But I'll describe it!

*Description: The `traffic_footage_5s.avi` shows a street scene with several pedestrians, cars, and motorcycles moving. Each detected object has a clear bounding box and a colored segmentation mask around it, along with its class label and a confidence score.*

 What's Next:
Completing this task shows how we can use AI to automatically understand what's happening in video feeds. This is just the start; the next big step (which we'll cover in training) would be to fine-tune the model to recognize even more specific things or situations, which is called "transfer learning."

I'm excited about the potential of this!
