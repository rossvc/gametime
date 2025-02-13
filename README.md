Using two datasets:
- Screenshots from NBA broadcasts (includes halftime presentations, stat cards, and anything that is shown during a typical NBA game)
- Screenshots of various commercials (Past 10 years of the Super Bowl, current NBA, local and national commercials)

The screenshots (we are using the broadcast for educational/research purposes, which arguably gives us grounds for fair use), are hosted on Kaggle:

[Shortened collection of NBA screenshots](https://www.kaggle.com/datasets/obviouslygelb/game-screenshots)

We train a CNN to determine whether or not a random screenshot is part of an NBA broadcast.
We use a flask server and create an endpoint that takes in pictures, in the response will be the class.

Ideally, the images should be resized locally before sending to the server to reduce bandwdith and upload times.

NBA broadcasts are heavily compressed, usually in 1920x1080 resolution.
The input image dimensions are 224x224, so resize the image to 224x126 and add 49 px padding on the top and bottom.

The method of capturing screenshots is up to you: 

You could perhaps use something like a capture card (or anything that acts like a webcam) connected to an HDMI splitter to stream the broadcast to a Raspberry Pi and your TV, run the image resizing on the Pi, and send it to the server, or run the model locally on the Pi (performance might vary, could use a mini-PC instead). From there you'll need some sort of controller to manage the TV's audio and video, one idea would be to attach an RF transmitter to the Pi and send signals to the TV to change the channel or mute the audio.

I use [IINA](https://iina.io) and an IPTV stream of the broadcast in my setup. I created a custom plugin through their [plugin API](https://docs.iina.io) that takes a screenshot of the stream every second. The images are sent to my server, and if three images in a row are considered not to be NBA footage, the plugin mutes the stream and switches to an image of a random artwork.
In the background, the plugin is still taking screenshots of the stream. If the model determines that the three straight images are part of an NBA broadcast, the plugin unmutes and returns the video to the player.


Currently, we train our own model with PyTorch. I plan to swap it out for a pre-trained model (Resnet, EfficientNet) and fine-tune that on the dataset instead.
