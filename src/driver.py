import cv2

from hand_tracker import HandPoseTracker
from tone_generator import ToneGenerator
from visualizer import Visualizer

def main():
    # For webcam input:
    cap = cv2.VideoCapture(0)
    hand_pose_tracker = HandPoseTracker()
    tone_generator = ToneGenerator()

    visuals_on = True
    visualizer = Visualizer(visuals_on)

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

        image = visualizer.prepare_img_for_drawing(image)

        last_dists = None
        if hand_pose_tracker.update_success:
            image = visualizer.draw_hand_annotations(image, hand_pose_tracker.handed_landmarks)

            # TODO: combine these two methods (always used in conjunction)
            left_dist = hand_pose_tracker.get_left_dist()
            right_dist = hand_pose_tracker.get_right_dist()

            image = visualizer.show_left_dist(image, left_dist)
            image = visualizer.show_right_dist(image, right_dist)

            tone_generator.generate_tone(left_dist, right_dist)
            last_dists = (left_dist, right_dist)
        else:
            # TODO: make caching explicit part of Driver class
            if last_dists:
                left_dist, right_dist = last_dists
                image = show_left_dist(image, left_dist)
                image = show_right_dist(image, right_dist)
                tone_generator.generate_tone(left_dist, right_dist)
            # else:
            #     tone_generator.stop_tone()
                    
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('Visual Theremin', cv2.flip(image, 1))

        # end loop on spacebar
        if cv2.waitKey(5) & 0xFF == ord(' '):
            break

    # release camera capture
    cap.release()

if __name__ == "__main__":
    main()
