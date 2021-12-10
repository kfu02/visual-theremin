import cv2

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

class Visualizer():
    def __init__(self, visuals_on=True):
        self.visuals_on = visuals_on

        self.line_color = (255, 0, 0)
        self.line_width = 5

    def prep_img_for_drawing(self, image):
        """Ensure that image is writeable and in RGB colorspace.
        """
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def draw_hand_annotations(self, image, multi_hand_landmarks):
        """Draws hand annotations on an RGB image.

        Args:
            image: RGB image represented as cv2 mat
            multi_hand_landmarks: results.multi_hand_landmarks from output of MediaPipe Hands process()
        """
        if not self.visuals_on: 
            return image

        for hand_landmarks in multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        return image

    def draw_hand_lines(self, image, left_dist, right_dist):
        """Show vertical pos of left hand and horizontal pos of right hand with lines.
        """
        if not self.visuals_on: 
            return image

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

