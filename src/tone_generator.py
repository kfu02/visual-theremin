import pyo
import numpy as np

class ToneGenerator():
    def __init__(self):
        # Initialize a Server object
        self.server = pyo.Server()

        # print all current audio devices, see which one is set to output
        pyo.pa_list_devices()
        print(pyo.pa_get_default_output())
        # self.server.setOutputDevice(2)

        # start server
        self.server.boot()
        self.server.start()

        # set tone to Sine wave (TODO: experiment with modulating existing waves saved to files)
        lfo3 = pyo.Sine(.1).range(0, .18)
        osc3 = pyo.SineLoop(freq=187.5, feedback=lfo3, mul=0.3)
        self.tone = osc3

    def generate_tone(self, handed_dists):
        """Generate and play tone from hand_tracker outputted normalized left/right_dist
                   ---------------- (0.0, 0.0)
                   |              |
                   |              |
                   |              |
        (1.0, 1.0) ---------------- 
        """
        left_dist, right_dist = handed_dists

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
        # print(mul, freq)

        self.tone.setMul(mul)
        self.tone.setFreq(freq)

        self.tone.out(0) # L channel
        self.tone.out(1) # R channel

    def stop_tone(self):
        self.tone.stop()


