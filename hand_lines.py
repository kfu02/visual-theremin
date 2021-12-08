import cv2
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def get_hand_pose_processor():
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1, # 0 = low, 1 = high
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

def prepare_img_for_drawing(image):
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

def single_hand_to_coords(single_hand_landmarks):
    landmark_list = []
    for landmark in single_hand_landmarks.landmark:
        landmark_list.append([landmark.x, landmark.y, landmark.z])

    return np.asarray(landmark_list)

# TODO: change these names
# TODO: break out the coordinate calculations?
def show_right_bound(image, right_hand_landmarks):
    hand_coords = single_hand_to_coords(right_hand_landmarks)
    min_x = np.amin(hand_coords[:, 0], axis=0)
    h, w, c = image.shape
    
    st = (int(min_x * w), 0)
    end = (int(min_x * w), h)
    image = cv2.line(image, st, end, (255,0,0), 5)
    return image

def show_left_bound(image, left_hand_landmarks):
    hand_coords = single_hand_to_coords(left_hand_landmarks)
    min_y = np.amin(hand_coords[:, 1], axis=0)
    h, w, c = image.shape
    
    st = (0, int(min_y * h))
    end = (w, int(min_y * h))
    image = cv2.line(image, st, end, (255,0,0), 5)
    return image

def main():
    # For webcam input:
    cap = cv2.VideoCapture(0)
    hand_pose_processor = get_hand_pose_processor()

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

        # for explanation of fields:
        # https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/hands.py#L132
        results = hand_pose_processor.process(image)

        multi_handedness = results.multi_handedness
        # (0,0) in top-left, normalized coords meaning x, y = [0.0, 1.0] 
        multi_hand_landmarks = results.multi_hand_landmarks

        image = prepare_img_for_drawing(image)
        if multi_hand_landmarks is not None:
            if len(multi_hand_landmarks) == 2:
                image = draw_hand_annotations(image, multi_hand_landmarks)

                left_hand_landmarks = None
                right_hand_landmarks = None

                for i in range(2):
                    cls = multi_handedness[i].classification[0]
                    # index doesn't match the order of multi_hand_landmarks
                    index = cls.index
                    score = cls.score
                    label = cls.label
                    if score > 0.5:
                        if label == "Left":
                            right_hand_landmarks = multi_hand_landmarks[i]
                        elif label == "Right":
                            left_hand_landmarks = multi_hand_landmarks[i]

                if right_hand_landmarks is not None and left_hand_landmarks is not None:
                    # image = draw_hand_annotations(image, [left_hand_landmarks])
                    image = show_right_bound(image, right_hand_landmarks)
                    image = show_left_bound(image, left_hand_landmarks)
                    
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == ord(' '):
            break
    cap.release()

if __name__ == "__main__":
    main()
