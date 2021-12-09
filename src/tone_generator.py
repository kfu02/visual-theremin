import pyo
import numpy as np

class ToneGenerator():
    def __init__(self):
        # Initialize a Server object
        self.server = pyo.Server().boot()

        # not sure why but doesn't work on airpods
        # pyo.pa_list_devices()
        # s.setOutputDevice(0)

        self.server.start()
        self.sine_wave = pyo.Sine(mul=0.01)

    # TODO: separate out the height/width calculations
    def generate_tone(self, left_dist, right_dist):
        """Generate and play tone from hand_tracker outputted normalized left/right_dist
                   ---------------- (0.0, 0.0)
                   *              .
                   *              .
                   *              .
        (1.0, 1.0) ---------------- 
        """

        h_min = 0.5
        h_max = 0.8
        w_min = 0.0
        w_max = 0.5

        # clip and renormalize height/width
        height = (np.clip(left_dist, h_min, h_max).item(0) - h_min) / (h_max - h_min)
        width = (np.clip(right_dist, w_min, w_max).item(0) - w_min) / (w_max - w_min)

        # flip directions (so h = 1 is top, w = 1 is right)
        height = 1 - height
        width = 1 - width

        # print(height, width)

        # convert height/width to amplitude/freq
        mul = float(height * 0.1) + 0.001 # prevent mul from going to 0 to eliminate static click
        freq_min = 261.626 # c3 in hz
        freq_max = 880.000 # a5 in hz
        freq = float(width * (freq_max - freq_min) + freq_min)
        print(mul, freq)

        self.sine_wave.setMul(mul)
        self.sine_wave.setFreq(freq)
        self.sine_wave.out()

    def stop_tone(self):
        self.sine_wave.stop()


