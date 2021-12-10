import cv2

from hand_tracker import HandPoseTracker
from tone_generator import ToneGenerator
from visualizer import Visualizer

class Driver():
    def __init__(self):
        # webcam input
        self.cap = cv2.VideoCapture(0)

        self.hand_pose_tracker = HandPoseTracker()
        self.tone_generator = ToneGenerator()

        self.visuals_on = False
        self.draw_hands, self.draw_pose, self.draw_lines = True, False, True
        self.visualizer = Visualizer(self.visuals_on, self.draw_hands, self.draw_pose, self.draw_lines)
        self.frame_name = "Visual Theremin"

        self.last_dists = None

    def run(self):
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            self.hand_pose_tracker.update(image)

            image = self.visualizer.prep_img_for_drawing(image)

            if self.hand_pose_tracker.update_success:

                self.last_dists = self.hand_pose_tracker.get_left_right_dist()

            # last_dists is current if hand_pose_tracker updates successfully, is last successful frame otherwise
            if self.last_dists:
                left_dist, right_dist = self.last_dists
                image = self.visualizer.draw_annotations(image, self.hand_pose_tracker.handed_landmarks, self.last_dists, self.hand_pose_tracker.results.pose_landmarks)
                self.tone_generator.generate_tone(self.last_dists)
                        
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow(self.frame_name, cv2.flip(image, 1))

            # end loop on spacebar
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

        # stop tone
        self.tone_generator.stop_tone()

        # release camera capture
        self.cap.release()

def main():
    driver = Driver()
    driver.run()

if __name__ == "__main__":
    main()
