import cv2

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic

class Visualizer():
    def __init__(self, visuals_on=True, draw_hands=True, draw_pose=False, draw_lines=True):
        self.visuals_on = visuals_on
        self.draw_hands = draw_hands
        self.draw_pose = draw_pose
        self.draw_lines = draw_lines

        self.line_color = (255, 0, 0)
        self.line_width = 5

    def prep_img_for_drawing(self, image):
        """Ensure that image is writeable and in RGB colorspace.
        """
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def draw_annotations(self, image, multi_hand_landmarks, handed_dists, pose_landmarks):
        if not self.visuals_on:
            return image

        if self.draw_hands and multi_hand_landmarks:
            image = self.draw_hand_annotations(image, multi_hand_landmarks)
            # hand_lines can only be drawn if given multi_hand_landmarks

        if self.draw_pose and pose_landmarks:
            image = self.draw_pose_annotations(image, pose_landmarks)

        if self.draw_lines and handed_dists:
            image = self.draw_hand_lines(image, handed_dists)

        return image

    def draw_hand_annotations(self, image, multi_hand_landmarks):
        # TODO: rename all multi_hand_landmarks to handed_landmarks to match Tracker
        """Draws hand annotations on an RGB image.

        Args:
            image: RGB image represented as cv2 mat
            multi_hand_landmarks: results.multi_hand_landmarks from output of MediaPipe Hands process()
        """
        for hand_landmarks in multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        return image

    def draw_pose_annotations(self, image, pose_landmarks):
        mp_drawing.draw_landmarks(
            image,
            pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())
        return image

    def draw_hand_lines(self, image, handed_dists):
        """Show vertical pos of left hand and horizontal pos of right hand with lines.
        """
        left_dist, right_dist = handed_dists
        return self.draw_right_dist(self.draw_left_dist(image, left_dist), right_dist)

    def draw_left_dist(self, image, left_dist):
        """Visualize vertical left-hand distance with a horizontal blue line.
        """
        h, w, c = image.shape
        
        st = (0, int(left_dist * h))
        end = (w, int(left_dist * h))
        image = cv2.line(image, st, end, self.line_color, self.line_width)
        return image

    def draw_right_dist(self, image, right_dist):
        """Visualize horizontal right-hand distance with a vertical blue line.
        """
        h, w, c = image.shape
        
        st = (int(right_dist * w), 0)
        end = (int(right_dist * w), h)
        image = cv2.line(image, st, end, self.line_color, self.line_width)
        return image

