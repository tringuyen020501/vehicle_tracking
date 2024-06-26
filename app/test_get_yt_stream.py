# import required libraries
from vidgear.gears import CamGear
import cv2


# define suitable tweak parameters for your stream.
options = {"STREAM_RESOLUTION": "480p", "CAP_PROP_FRAME_WIDTH": 640, "CAP_PROP_FRAME_HEIGHT": 480, "CAP_PROP_FPS": 40, "CAP_PROP_BUFFERSIZE": 120}

# To open live video stream on webcam at first index(i.e. 0) 
# device and apply source tweak parameters
stream = CamGear(source="https://www.youtube.com/watch?v=8vHp9b8ZlJc&ab_channel=Ph%C3%A1tTri%E1%BB%83n%C4%90%C3%A0N%E1%BA%B5ng",stream_mode=True, logging=True, **options).start()
# stream = CamGear(source="https://www.youtube.com/watch?v=Fu3nDsqC1J0c&ab_channel=Ph%C3%A1tTri%E1%BB%83n%C4%90%C3%A0N%E1%BA%B5ng",stream_mode=True, logging=True, **options).start()

# loop over
while True:

    # read frames from stream
    frames = [stream.read() for i in range(2)]  # Read 2 frames at once

    for frame in frames:
        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # Show output window
        cv2.imshow("Output", frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()
