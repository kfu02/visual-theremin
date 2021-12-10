import numpy as np
import mediapipe as mp
mp_holistic = mp.solutions.holistic

class HandPoseTracker():
    """Uses mp_holistic to get pose, face, and hand tracking data, then discards pose and face data.

    (This provides more accurate hand detection than mp_hands alone.)
    """

    def __init__(self):
        self.processor = mp_holistic.Holistic(
            static_image_mode=False,
            # max_num_hands=2,
            model_complexity=0, # 0 = low, 1 = med, 2 = high
            # min_detection_confidence=0.3,
            # min_tracking_confidence=0.3
        )

        self.results = None
        self.update_success = False
        self.handed_landmarks = [None, None] # left, right

    def update(self, image):
        """Call process from mp_holistic, then store L/R hand results.

        Set update_success flag if self.handed_landmarks is successfully filled in.
        """
        # google "mediapipe github" for explanation of results
        self.results = self.processor.process(image)
        assert self.results is not None

        if self.results.left_hand_landmarks and self.results.right_hand_landmarks:
            self.handed_landmarks[0] = self.results.left_hand_landmarks
            self.handed_landmarks[1] = self.results.right_hand_landmarks
            self.update_success = True
        else:
            self.update_success = False

    def single_hand_to_coords(self, single_hand_landmarks):
        """Convert single_hand_landmarks returned by MediaPipe Hands to (21,3) np array of coordinates.
        """
        landmark_list = []
        for landmark in single_hand_landmarks.landmark:
            landmark_list.append([landmark.x, landmark.y, landmark.z])
        return np.asarray(landmark_list)

    def get_left_right_dist(self):
        return self.get_left_dist(), self.get_right_dist()

    def get_left_dist(self):
        """Find lowest point on left hand, return normalized distance from top of frame.

        Ignore palm & thumb to be true to real theremin.
        """
        hand_coords = self.single_hand_to_coords(self.handed_landmarks[0])
        min_y = np.amax(hand_coords[5:, 1], axis=0)
        return min_y

    def get_right_dist(self):
        """Find rightmost point on right hand, return normalized distance from right of frame.
        """
        hand_coords = self.single_hand_to_coords(self.handed_landmarks[1])
        min_x = np.amin(hand_coords[:, 0], axis=0)
        return min_x


