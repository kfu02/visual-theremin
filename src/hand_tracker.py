import numpy as np
import mediapipe as mp
mp_hands = mp.solutions.hands

class HandPoseTracker():
    def __init__(self):
        self.hand_pose_processor = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0, # 0 = low, 1 = high
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.results = None
        self.update_success = False
        self.handed_landmarks = [None, None] # left, right

    def update(self, image):
        """Call process from mp_hands, then use results.handedness to split results.multi_hand_landmarks
        into left and right hand_landmarks. Flag update_success if self.handed_landmarks 
        is successfully filled in.
        """
        # for explanation of fields:
        # https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/hands.py#L132
        self.results = self.hand_pose_processor.process(image)
        assert self.results is not None
        
        # find handedness, fill in left/right_hand_landmarks
        # (0,0) in top-right (flipped), normalized coords meaning x, y = [0.0, 1.0] 
        multi_hand_landmarks = self.results.multi_hand_landmarks
        multi_handedness = self.results.multi_handedness
        if multi_hand_landmarks and len(multi_hand_landmarks) == 2:
            for i in range(2):
                cls = multi_handedness[i].classification[0]
                # cls.index doesn't match the order of multi_hand_landmarks
                if cls.score > 0.5:
                    if cls.label == "Left":
                        self.handed_landmarks[1] = multi_hand_landmarks[i]
                    elif cls.label == "Right":
                        self.handed_landmarks[0] = multi_hand_landmarks[i]

            self.update_success = all(hl for hl in self.handed_landmarks)
        else:
            self.update_success = False

    def single_hand_to_coords(self, single_hand_landmarks):
        """Convert single_hand_landmarks returned by MediaPipe Hands to (21,3) np array of coordinates.
        """
        landmark_list = []
        for landmark in single_hand_landmarks.landmark:
            landmark_list.append([landmark.x, landmark.y, landmark.z])
        return np.asarray(landmark_list)

    def get_left_dist(self):
        """Find lowest point on left hand, return normalized distance from top of frame.
        """
        hand_coords = self.single_hand_to_coords(self.handed_landmarks[0])
        min_y = np.amax(hand_coords[:, 1], axis=0)
        return min_y

    def get_right_dist(self):
        """Find rightmost point on right hand, return normalized distance from right of frame.
        """
        hand_coords = self.single_hand_to_coords(self.handed_landmarks[1])
        min_x = np.amin(hand_coords[:, 0], axis=0)
        return min_x


