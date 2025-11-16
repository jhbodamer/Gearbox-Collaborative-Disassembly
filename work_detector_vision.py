import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

class WorkDetector:
    def __init__(self, idle_threshold=2.0):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        # inside WorkDetector.__init__()
        self.state = "IDLE"

        # Hand detection for finger tracking
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Pose detection for general presence
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Gearbox detection
        self.gearbox_bbox = None
        self.gearbox_history = deque(maxlen=10)

        # Hand tracking
        self.hand_positions = deque(maxlen=30)
        self.finger_positions = deque(maxlen=30)
        self.hands_on_gearbox_history = deque(maxlen=15)

        # Working state
        self.idle_threshold = idle_threshold
        self.last_activity_time = time.time()
        self.is_working = False

        # Calibration mode
        self.calibration_mode = True
        self.calibration_frames = []

    def detect_green_gearbox(self, frame):
        """
        Detect the green gearbox using color segmentation.
        Returns bounding box (x, y, w, h) and center point.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define range for green color
        # Lower green (adjust these if needed)
        lower_green1 = np.array([35, 40, 40])
        upper_green1 = np.array([85, 255, 255])

        # Create mask
        mask = cv2.inRange(hsv, lower_green1, upper_green1)

        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None, mask

        # Find largest contour (assumed to be gearbox)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # Filter out noise (minimum area threshold)
        if area < 1000:  # Adjust this threshold as needed
            return None, None, mask

        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        bbox = (x, y, w, h)

        # Get center point
        center = (x + w // 2, y + h // 2)

        # Store in history for stability
        self.gearbox_history.append(bbox)

        # Average bbox over recent frames for stability
        if len(self.gearbox_history) >= 3:
            avg_x = int(np.mean([b[0] for b in self.gearbox_history]))
            avg_y = int(np.mean([b[1] for b in self.gearbox_history]))
            avg_w = int(np.mean([b[2] for b in self.gearbox_history]))
            avg_h = int(np.mean([b[3] for b in self.gearbox_history]))
            bbox = (avg_x, avg_y, avg_w, avg_h)
            center = (avg_x + avg_w // 2, avg_y + avg_h // 2)

        return bbox, center, mask

    def are_hands_on_gearbox(self, hand_landmarks_list, gearbox_bbox, frame_shape):
        """
        Check if any hand landmarks are within or near the gearbox bounding box.
        """
        if gearbox_bbox is None or not hand_landmarks_list:
            return False

        h, w = frame_shape[:2]
        x, y, box_w, box_h = gearbox_bbox

        # Expand bounding box slightly for proximity detection
        margin = 30  # pixels
        expanded_x1 = max(0, x - margin)
        expanded_y1 = max(0, y - margin)
        expanded_x2 = min(w, x + box_w + margin)
        expanded_y2 = min(h, y + box_h + margin)

        # Check each hand
        for hand_landmarks in hand_landmarks_list:
            # Check all landmarks (not just fingertips) for contact
            for landmark in hand_landmarks.landmark:
                lm_x = int(landmark.x * w)
                lm_y = int(landmark.y * h)

                # Check if landmark is within expanded bounding box
                if (expanded_x1 <= lm_x <= expanded_x2 and
                        expanded_y1 <= lm_y <= expanded_y2):
                    return True

        return False

    def detect_finger_movement(self, hand_landmarks):
        """
        Track finger tip positions to detect fine motor movements.
        """
        if not hand_landmarks:
            return 0

        # Get finger tip positions (indices 4, 8, 12, 16, 20)
        finger_tips = [4, 8, 12, 16, 20]
        current_positions = []

        for landmark_idx in finger_tips:
            landmark = hand_landmarks.landmark[landmark_idx]
            current_positions.append((landmark.x, landmark.y, landmark.z))

        # Average position of all fingertips
        avg_position = np.mean(current_positions, axis=0)
        self.finger_positions.append(avg_position)

        if len(self.finger_positions) < 2:
            return 0

        # Calculate movement over recent frames
        movement = 0
        for i in range(1, min(10, len(self.finger_positions))):
            prev = self.finger_positions[-i - 1]
            curr = self.finger_positions[-i]
            movement += np.linalg.norm(curr - prev)

        return movement

    def is_person_working(self, frame):
        """
        Determine if person is working based on hands being on the gearbox.
        """
        h, w = frame.shape[:2]

        # 1. Detect green gearbox
        gearbox_bbox, gearbox_center, gearbox_mask = self.detect_green_gearbox(frame)

        # 2. Hand Detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)

        finger_movement = 0
        hands_detected = False
        person_detected = False
        hands_on_gearbox = False

        # Check for person presence
        if pose_results.pose_landmarks:
            person_detected = True
            self.mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        # Check for hands and finger movement
        hand_landmarks_list = []
        if hand_results.multi_hand_landmarks:
            hands_detected = True
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_landmarks_list.append(hand_landmarks)

                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
                )

                # Track finger movements
                finger_movement += self.detect_finger_movement(hand_landmarks)

        # 3. Check if hands are on gearbox
        if gearbox_bbox is not None:
            hands_on_gearbox = self.are_hands_on_gearbox(hand_landmarks_list, gearbox_bbox, frame.shape)
            self.hands_on_gearbox_history.append(hands_on_gearbox)
        else:
            # If gearbox not detected, assume hands are NOT on it
            self.hands_on_gearbox_history.append(False)

        # Average over recent frames for stability
        hands_on_gearbox_avg = np.mean(self.hands_on_gearbox_history) if self.hands_on_gearbox_history else 0

        # 4. Determine working indicators
        working_indicators = {
            'hands_on_gearbox': hands_on_gearbox_avg > 0.4,  # Primary indicator!
            'finger_movement': finger_movement > 0.03,  # Increased threshold - needs actual movement
            'gearbox_detected': gearbox_bbox is not None,
            'person_present': person_detected
        }

        # Calculate confidence (heavily weighted toward hands on gearbox)
        weights = {
            'hands_on_gearbox': 0.50,  # Important but not required
            'finger_movement': 0.40,  # Can indicate working (grabbing tools, etc)
            'gearbox_detected': 0.05,
            'person_present': 0.05
        }

        confidence = sum(weights[k] * (1 if v else 0) for k, v in working_indicators.items())

        # Update working state
        is_working = confidence > 0.30  # Lowered threshold so movement alone can trigger

        if is_working:
            self.last_activity_time = time.time()
            self.is_working = True
        else:
            idle_time = time.time() - self.last_activity_time
            if idle_time > self.idle_threshold:
                self.is_working = False

        # Update state string for robot communication
        self.state = "WORKING" if self.is_working else "IDLE"

        debug_info = {
            'hands_on_gearbox': hands_on_gearbox,
            'hands_on_gearbox_avg': hands_on_gearbox_avg,
            'finger_movement': finger_movement,
            'hands_detected': hands_detected,
            'gearbox_detected': gearbox_bbox is not None,
            'person_detected': person_detected,
            'idle_time': time.time() - self.last_activity_time,
            'indicators': working_indicators,
            'gearbox_bbox': gearbox_bbox,
            'gearbox_mask': gearbox_mask
        }

        return self.is_working, confidence, debug_info

    def get_state(self):
        """Return current working state as a string."""
        return self.state

    def draw_status(self, frame, is_working, confidence, debug_info):
        """Draw status information on frame."""
        h, w = frame.shape[:2]

        # Draw gearbox bounding box
        if debug_info['gearbox_bbox'] is not None:
            x, y, box_w, box_h = debug_info['gearbox_bbox']
            box_color = (0, 255, 255) if debug_info['hands_on_gearbox'] else (255, 0, 255)
            cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), box_color, 3)

            # Label
            label = "HANDS ON GEARBOX" if debug_info['hands_on_gearbox'] else "Gearbox Detected"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # Status box
        status_color = (0, 255, 0) if is_working else (0, 0, 255)
        status_text = "WORKING" if is_working else "IDLE - ROBOT READY"

        cv2.rectangle(frame, (10, 10), (480, 250), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (480, 250), status_color, 2)

        cv2.putText(frame, status_text, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.0%}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Debug info
        y_offset = 115
        cv2.putText(frame,
                    f"Hands on Gearbox: {debug_info['hands_on_gearbox']} ({debug_info['hands_on_gearbox_avg']:.0%})",
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Finger Movement: {debug_info['finger_movement']:.3f}",
                    (20, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Hands Detected: {debug_info['hands_detected']}",
                    (20, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Gearbox Detected: {debug_info['gearbox_detected']}",
                    (20, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Idle Time: {debug_info['idle_time']:.1f}s",
                    (20, y_offset + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame


def main():
    """Main function to run the work detection system."""
    detector = WorkDetector(idle_threshold=2.0)

    # Open webcam
    cap = cv2.VideoCapture(6)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("=" * 60)
    print("Work Detection System Started")
    print("=" * 60)
    print("The system will detect:")
    print("  1. Green gearbox location")
    print("  2. Whether hands are touching/near the gearbox")
    print("  3. Finger movements for bolt work")
    print()
    print("Commands:")
    print("  'q' - Quit")
    print("  'r' - Reset idle timer")
    print("  'm' - Toggle mask view (see green detection)")
    print("=" * 60)

    show_mask = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Flip frame for mirror view
        frame = cv2.flip(frame, 1)

        # Detect if person is working
        is_working, confidence, debug_info = detector.is_person_working(frame)

        # Draw status on frame
        frame = detector.draw_status(frame, is_working, confidence, debug_info)

        # Signal to robot system
        if not is_working:
            print(f"\n{'=' * 60}")
            print(f"🤖 SIGNAL: Robot can proceed!")
            print(f"   Idle for {debug_info['idle_time']:.1f}s")
            print(f"   Hands off gearbox: {not debug_info['hands_on_gearbox']}")
            print(f"{'=' * 60}\n")

        elif is_working:
            print("Work resumed - Stop Robot")

        # Display frame
        if show_mask and debug_info['gearbox_mask'] is not None:
            # Show mask view
            mask_display = cv2.cvtColor(debug_info['gearbox_mask'], cv2.COLOR_GRAY2BGR)
            combined = np.hstack([frame, mask_display])
            cv2.imshow('Work Detection System', combined)
        else:
            cv2.imshow('Work Detection System', frame)

        # Check for user input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.last_activity_time = time.time()
            print("⏱️  Timer reset!")
        elif key == ord('m'):
            show_mask = not show_mask
            print(f"Mask view: {'ON' if show_mask else 'OFF'}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.hands.close()
    detector.pose.close()


if __name__ == "__main__":
    main()
