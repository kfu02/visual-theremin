import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

from hand_tracker import HandPoseTracker
from tone_generator import ToneGenerator

def prepare_img_for_drawing(image):
    """Ensure that image is writeable and in RGB colorspace.
    """
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def draw_hand_annotations(image, multi_hand_landmarks):
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

def show_left_dist(image, left_dist):
    """Visualize vertical left-hand distance with a horizontal blue line.
    """
    h, w, c = image.shape
    
    st = (0, int(left_dist * h))
    end = (w, int(left_dist * h))
    image = cv2.line(image, st, end, (255,0,0), 5)
    return image

def show_right_dist(image, right_dist):
    """Visualize horizontal right-hand distance with a vertical blue line.
    """
    h, w, c = image.shape
    
    st = (int(right_dist * w), 0)
    end = (int(right_dist * w), h)
    image = cv2.line(image, st, end, (255,0,0), 5)
    return image

def main():
    # For webcam input:
    cap = cv2.VideoCapture(0)
    hand_pose_tracker = HandPoseTracker()
    tone_generator = ToneGenerator()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        hand_pose_tracker.update(image)

        image = prepare_img_for_drawing(image)

        last_dists = None
        if hand_pose_tracker.update_success:
            # image = draw_hand_annotations(image, hand_pose_tracker.handed_landmarks)

            # TODO: combine these two methods (always used in conjunction)
            left_dist = hand_pose_tracker.get_left_dist()
            right_dist = hand_pose_tracker.get_right_dist()

            # TODO: separate out into drawing class
            # image = show_left_dist(image, left_dist)
            # image = show_right_dist(image, right_dist)

            tone_generator.generate_tone(left_dist, right_dist)
            last_dists = (left_dist, right_dist)
        else:
            # TODO: make caching explicit part of Driver class
            if last_dists:
                left_dist, right_dist = last_dists
                # image = show_left_dist(image, left_dist)
                # image = show_right_dist(image, right_dist)
                tone_generator.generate_tone(left_dist, right_dist)
            # else:
            #     tone_generator.stop_tone()
                    
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

        # end loop on spacebar
        if cv2.waitKey(5) & 0xFF == ord(' '):
            break

    # release camera capture
    cap.release()

if __name__ == "__main__":
    main()
