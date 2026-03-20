""" This file was created by John
currently trying to implement automatic alignment similar to 
loop_pickup.py with the DOBOT and its attached USB webcam
A lot of the code here will need to be replaced so the file does not work currently
"""

#!/usr/bin/env python3
import numpy as np
import cv2
from math import atan2, cos, sin, sqrt, pi

# ---------- USER CONFIG ----------
# Camera Index (0 is usually the default USB webcam)
WEBCAM_INDEX = 0 
STREAM_W, STREAM_H = 640, 480
FPS = 30

# Contour filtering thresholds (adjust based on your object size/camera dist)
MIN_AREA = 5000
MAX_AREA = 100000
THRESHOLD_VALUE = 80  # Threshold for binary conversion (0-255)
# ----------------------------------

def drawAxis(img, p_, q_, color, scale):
    """
    Draw axis arrows for visualization.
    """
    p = list(p_)
    q = list(q_)
    
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    
    # Lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
    
    # Create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
    
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

def getOrientation(pts, img):
    """
    Get orientation using PCA analysis on contour points.
    Returns angle (rad) and center point (x, y).
    """
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    
    # Visualization: Draw the principal components
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    
    drawAxis(img, cntr, p1, (255, 255, 0), 1) # Yellow axis
    drawAxis(img, cntr, p2, (0, 0, 255), 5)   # Red axis (Major)
    
    # Calculate angle
    angle = atan2(eigenvectors[1,1], eigenvectors[1,0])
    
    # Label with the rotation angle
    label = f"{np.degrees(angle):.1f} deg"
    
    # Text positioning
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_x = cntr[0] - text_width // 2
    text_y = cntr[1] - 40
    
    # Draw text outline and fill
    cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    
    return angle, cntr

def main():
    print("[INFO] Starting Contour Detection via USB Webcam...")
    print("[INFO] Press 'q' to quit.")

    # --- USB Webcam Setup ---
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_H)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam (index {WEBCAM_INDEX})")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame")
                continue

            ui = frame.copy()
            
            # 1. Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 2. Thresholding (Binary conversion)
            # You can switch to cv2.THRESH_OTSU if lighting varies significantly
            _, bw = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
            
            # 3. Find Contours
            contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            
            detected_count = 0

            for i, c in enumerate(contours):
                # 4. Filter by Area
                area = cv2.contourArea(c)
                if area < MIN_AREA or area > MAX_AREA:
                    continue
                
                detected_count += 1
                
                # Draw contour
                cv2.drawContours(ui, contours, i, (0, 0, 255), 2)
                
                # 5. Get Orientation (PCA) & Visualize
                try:
                    angle, center_point = getOrientation(c, ui)
                    
                    # Display Center Coordinates
                    coord_text = f"Px: {center_point}"
                    cv2.putText(ui, coord_text, (center_point[0] + 10, center_point[1] + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                               
                except Exception as e:
                    print(f"[WARN] PCA Error on contour {i}: {e}")

            # Display status
            cv2.putText(ui, f"Objects: {detected_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show the result
            cv2.imshow('Contour Detection & PCA Visualization', ui)
            
            # Show the binary mask (useful for debugging lighting/thresholds)
            cv2.imshow('Binary Mask', bw)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Program terminated.")

if __name__ == "__main__":
    main()