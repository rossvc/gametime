Using two datasets:
- Screenshots from NBA broadcasts (includes halftime presentations, stat cards, anything that is shown during a typical NBA game)
- Screenshots of various commericals (Past 10 years of superbowl, current NBA, local and national commericals)

We train a CNN to determine the likelihood of a random screenshot being a part of an NBA broadcast or not.
We use a flask server and create an endpoint that takes in pictures, in the response will be the likelihood.

Ideally, the images should be resized locally before sending to the server to reduce upload times.

NBA broadcasts are heavily compressed, usually in 1920x1080 resolution.
The input image dimensions are 256x256, so resize the image to 256x144, and add 56 px padding on the top and bottom.

The method of capturing screenshots is up to you: 

You could perhaps use something like a capture card or HDMI splitter to stream the broadcast to a mini-PC and your TV, run the image compression and send it to the server, or you could run the model locally on the mini-PC.

In my setup, I use [IINA](https://iina.io) and an IPTV stream of the broadcast. I created a custom plugin through their [plugin API](https://docs.iina.io) that takes a screenshot of the stream every second. The images are sent to my server and if 3 images in a row are considered not to be NBA footage, the plugin mutes the stream and blacks out the screen.
In the backgroumd, the plugin is still taking screenshots of the stream, and if the model determines that the 3 straight images are part of an NBA broadcast, the plugin unmutes and puts the video back onto the player.


Currently, we train our own model with PyTorch, I have plans to swap it out a pre-trained model (Resnet, EfficientNet) and fine-tune that on the images instead.
