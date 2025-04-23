#!/usr/bin/env python3
import cv2
import numpy as np
import math
from ultralytics import YOLO
import argparse
import os # Added for path operations
import json # Added for JSON output
import csv # Added for CSV output

# --- Constants for Drawing ---
# COCO Keypoint indices (0-16)
# Used for drawing skeleton
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]
]
# Adjust indices to be 0-based if the above list assumes 1-based
# YOLOv8 uses 0-based COCO indices: 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar,
# 5:LShoulder, 6:RShoulder, 7:LElbow, 8:RElbow, 9:LWrist, 10:RWrist,
# 11:LHip, 12:RHip, 13:LKnee, 14:RKnee, 15:LAnkle, 16:RAnkle
SKELETON_YOLO = [
    # Head
    [0, 1], [0, 2], [1, 3], [2, 4],
    # Body
    [5, 6], [5, 11], [6, 12], [11, 12],
    # Arms
    [5, 7], [6, 8], [7, 9], [8, 10],
    # Legs
    [11, 13], [12, 14], [13, 15], [14, 16]
]

# Colors (BGR format)
KP_COLOR = (0, 255, 0)  # Green for keypoints
SK_COLOR = (255, 0, 0)  # Blue for skeleton
TEXT_COLOR = (0, 0, 255) # Red for text
LOW_CONF_KP_COLOR = (0, 0, 255) # Red for low confidence keypoints

# --- Helper Functions (to be implemented based on Dart logic) ---

def find_angle(pointA, pointB, pointC):
    """Calculates the angle between three points (angle at pointA)."""
    # Vector AB
    vecAB = np.array(pointB) - np.array(pointA)
    # Vector AC
    vecAC = np.array(pointC) - np.array(pointA)

    dot_product = np.dot(vecAB, vecAC)
    magAB = np.linalg.norm(vecAB)
    magAC = np.linalg.norm(vecAC)

    if magAB == 0 or magAC == 0:
        return 0.0 # Avoid division by zero

    cos_theta = dot_product / (magAB * magAC)
    # Clamp value to handle potential floating point inaccuracies
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle_radians = math.acos(cos_theta)
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees

# Add constants for keypoint indices within this scope or pass them if needed
L_HIP_IDX, R_HIP_IDX = 11, 12
L_KNEE_IDX, R_KNEE_IDX = 13, 14
# L_ANKLE_IDX, R_ANKLE_IDX = 15, 16 # Add if needed for future extensions

def get_peak_index(y_coords, confidences, keypoint_index, is_front_view=False, conf_threshold=0.1):
    """Finds the frame index where the keypoint reaches its minimum y-coordinate (highest point).

    Args:
        y_coords: List of lists, where each inner list contains the Y-coordinate(s) for a frame.
                  For front view, it's the averaged Y; otherwise, the side's Y.
        confidences: List of lists, original confidence scores for ALL keypoints for each frame.
        keypoint_index: The primary keypoint index to check confidence for (e.g., L_HIP for hip peak).
        is_front_view: Boolean indicating if the front view averaging logic was used.
        conf_threshold: The minimum confidence score required.
    """
    peak_index = 0
    min_y = float('inf')
    valid_indices = []

    # Determine which indices to check for confidence based on view
    check_indices = [keypoint_index]
    if is_front_view:
        if keypoint_index == L_HIP_IDX: check_indices.append(R_HIP_IDX)
        elif keypoint_index == R_HIP_IDX: check_indices.append(L_HIP_IDX)
        elif keypoint_index == L_KNEE_IDX: check_indices.append(R_KNEE_IDX)
        elif keypoint_index == R_KNEE_IDX: check_indices.append(L_KNEE_IDX)
        # Add other pairs if needed (e.g., ankles)

    # Find frames where at least one of the relevant keypoints has sufficient confidence
    for i, conf_list in enumerate(confidences):
        has_sufficient_confidence = False
        for check_idx in check_indices:
             # Check if confidence list is long enough for the index
             if len(conf_list) > check_idx and conf_list[check_idx] > conf_threshold:
                 has_sufficient_confidence = True
                 break # Found one good keypoint for this frame
        if has_sufficient_confidence:
            valid_indices.append(i)

    if not valid_indices:
        print(f"---> DEBUG: get_peak_index failed for primary keypoint {keypoint_index} (checking indices: {check_indices}).")
        print(f"---> DEBUG: No frames found with confidence > {conf_threshold} for these keypoints.")
        # Optional: Print confidences for the first few frames to see the low values
        # max_debug_frames = min(5, len(confidences))
        # for dbg_i in range(max_debug_frames):
        #     dbg_confs = [f"{confidences[dbg_i][idx]:.2f}" for idx in check_indices if len(confidences[dbg_i]) > idx] 
        #     print(f"---> DEBUG: Frame {dbg_i} confidences for {check_indices}: {dbg_confs}")

        print(f"Warning: get_peak_index found no frames with sufficient confidence (>{conf_threshold}) for keypoint indices {check_indices}.")
        return 0 # No valid points found

    # --- Simplified Initialization and Loop --- #
    peak_index = -1 # Initialize to -1 (not found)
    min_y = float('inf')

    for i in valid_indices:
        # Ensure y_coords for this index exists and has the keypoint
        # For front view, y_coords inner list always has length 1 (the average)
        # For side view, it also has length 1 (the specific side's coord)
        # So, we always check index 0 of the inner list.
        if len(y_coords) > i and len(y_coords[i]) > 0: # Check if inner list exists and is not empty
            current_y = y_coords[i][0] # Always use index 0 for the extracted Y-coord
            if current_y < min_y:
                min_y = current_y
                peak_index = i

    # If peak_index remained -1, no valid points were processed
    if peak_index == -1:
         print(f"---> DEBUG: get_peak_index found valid_indices but failed to process any point for keypoint {keypoint_index}. Returning 0.")
         return 0

    return peak_index

def get_start_index(y_coords, peak_index, keypoint_index):
    """Finds the takeoff frame index."""
    if len(y_coords) < 3:
         print("Warning: Not enough frames (<3) for baseline y in get_start_index.")
         return 0
    # Check if baseline frame (index 2) has the keypoint
    if len(y_coords[2]) <= keypoint_index:
         print("Warning: Keypoint missing in baseline frame (idx 2) for get_start_index.")
         return 0 # Need baseline

    baseline_y = y_coords[2][keypoint_index]
    start_index = 0
    # Iterate backwards from peak
    for i in range(peak_index, 1, -1): # Stop at index 2
         # Check if current frame has the keypoint
        if len(y_coords) > i and len(y_coords[i]) > keypoint_index:
            if y_coords[i][keypoint_index] > baseline_y:
                start_index = i
                break
        else:
             # Handle missing keypoint in comparison frames - maybe return 0 or skip?
             # print(f"Warning: Missing keypoint {keypoint_index} at frame {i} during start index search.")
             pass # Continue searching, might lead to inaccurate start index

    # If loop finishes without break, start_index remains 0
    return start_index


def get_end_index(y_coords, peak_index, keypoint_index):
    """Finds the landing frame index."""
    if len(y_coords) < 3:
         print("Warning: Not enough frames (<3) for baseline y in get_end_index.")
         return len(y_coords) -1
     # Check if baseline frame (index 2) has the keypoint
    if len(y_coords[2]) <= keypoint_index:
        print("Warning: Keypoint missing in baseline frame (idx 2) for get_end_index.")
        return len(y_coords) - 1 # Return last frame index if baseline is not available

    baseline_y = y_coords[2][keypoint_index]
    end_index = len(y_coords) - 1 # Default to last frame if condition not met

    # Iterate forwards from peak
    for i in range(peak_index, len(y_coords)):
        # Check if current frame has the keypoint
        if len(y_coords) > i and len(y_coords[i]) > keypoint_index:
            if y_coords[i][keypoint_index] > baseline_y:
                end_index = i
                break
        else:
            # Handle missing keypoint
            # print(f"Warning: Missing keypoint {keypoint_index} at frame {i} during end index search.")
            pass # Continue searching

    # Ensure end_index is at least 1 greater than peak_index if possible
    if end_index <= peak_index and peak_index < len(y_coords) -1 :
        end_index = peak_index + 1

    return end_index

def draw_debug_info(frame, keypoints_xy, confidences, frame_count, kp_confidence_threshold=0.5):
    """Draws keypoints, skeleton, and frame number onto the frame."""
    height, width, _ = frame.shape
    annotated_frame = frame.copy()

    # Draw Frame Number
    cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_AA)

    if keypoints_xy is not None and confidences is not None and len(keypoints_xy) > 0:
        # Assuming keypoints_xy is a list/array of [x, y] for one person
        # And confidences is a list/array of confidence scores for that person

        # Draw Skeleton
        for kp_pair in SKELETON_YOLO:
            idx1, idx2 = kp_pair

            # Ensure keypoints exist and have high enough confidence
            if len(keypoints_xy) > max(idx1, idx2) and len(confidences) > max(idx1, idx2):
                if confidences[idx1] > kp_confidence_threshold and confidences[idx2] > kp_confidence_threshold:
                    pt1 = (int(keypoints_xy[idx1][0] * width), int(keypoints_xy[idx1][1] * height))
                    pt2 = (int(keypoints_xy[idx2][0] * width), int(keypoints_xy[idx2][1] * height))
                    cv2.line(annotated_frame, pt1, pt2, SK_COLOR, 2)

        # Draw Keypoints
        for i, (x, y) in enumerate(keypoints_xy):
             if len(confidences) > i:
                conf = confidences[i]
                # Only draw points above the threshold now
                if conf > kp_confidence_threshold:
                    color = KP_COLOR
                    radius = 5
                    center = (int(x * width), int(y * height))
                    cv2.circle(annotated_frame, center, radius, color, -1) # Draw filled circle
                # else: # Optionally draw low-confidence points differently or not at all
                #     color = LOW_CONF_KP_COLOR
                #     radius = 3
                #     center = (int(x * width), int(y * height))
                #     cv2.circle(annotated_frame, center, radius, color, -1)

    return annotated_frame

# --- Main Processing Logic ---

def process_video(video_path, model_path='yolov8m-pose.pt', output_video_path=None, output_json_path=None):
    # Load the YOLO model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading YOLO model from {model_path}: {e}")
        default_results = {
            "Status": "Failed",
            "Orientation Detected": "N/A",
            "Calculated Jump Height (cm)": 0.0,
            "Calculated Flight Time (s)": 0.0,
            "Take-off Velocity (est.) (m/s)": 0.0,
            "Knee Angle (at lowest point) (degrees)": 0.0,
            "Hip Angle (at peak) (degrees)": 0.0,
            "Error Message": f"Error loading YOLO model: {e}"
        }
        return default_results

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        default_results = {
            "Status": "Failed",
            "Orientation Detected": "N/A",
            "Calculated Jump Height (cm)": 0.0,
            "Calculated Flight Time (s)": 0.0,
            "Take-off Velocity (est.) (m/s)": 0.0,
            "Knee Angle (at lowest point) (degrees)": 0.0,
            "Hip Angle (at peak) (degrees)": 0.0,
            "Error Message": f"Could not open video file: {video_path}"
        }
        return default_results # Return failure dict

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps == 0:
        print("Warning: Could not get FPS from video. Using default 30.")
        fps = 30.0
    if frame_width == 0 or frame_height == 0:
         print("Warning: Could not get frame dimensions. Video output might fail.")
         # Attempt to read the first frame to get dimensions
         ret, frame = cap.read()
         if ret:
             frame_height, frame_width, _ = frame.shape
             print(f"Got dimensions from first frame: {frame_width}x{frame_height}")
             cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video to beginning
         else:
             print("Error: Could not read first frame to get dimensions.")
             cap.release()
             default_results = {
                 "Status": "Failed",
                 "Orientation Detected": "N/A",
                 "Calculated Jump Height (cm)": 0.0,
                 "Calculated Flight Time (s)": 0.0,
                 "Take-off Velocity (est.) (m/s)": 0.0,
                 "Knee Angle (at lowest point) (degrees)": 0.0,
                 "Hip Angle (at peak) (degrees)": 0.0,
                 "Error Message": "Could not read first frame to get dimensions."
             }
             return default_results # Return failure dict


    frame_time_ms = 1000.0 / fps # Time per frame in milliseconds

    # Initialize Video Writer if output path is provided
    video_writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for path: {output_video_path}")
            video_writer = None # Disable writing
        else:
            print(f"Saving annotated video to: {output_video_path}")

    all_x_coords = []
    all_y_coords = []
    all_confidences = []
    timestamps_ms = []
    frame_count = 0

    print("Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        # Perform pose estimation
        results = model(frame, verbose=False) # verbose=False to reduce console output

        current_frame_kps_xy = None
        current_frame_confs = None
        num_keypoints = model.model.yaml.get('kpt_shape', [17, 3])[0] # Get expected number of kpts


        if results and results[0].keypoints:
            kpts = results[0].keypoints
            # Get normalized coordinates (xy) and confidences
            # Handle cases where no keypoints are detected in a frame
            if kpts.conf is not None and kpts.xy is not None and kpts.xy.numel() > 0:
                # Assuming only one person is detected, take the first set of keypoints
                current_frame_confs = kpts.conf[0].cpu().numpy().tolist()
                 # Normalize coordinates
                current_frame_kps_xy_raw = kpts.xy[0].cpu().numpy() # Shape (num_kpts, 2)
                current_frame_kps_x = (current_frame_kps_xy_raw[:, 0] / frame_width).tolist()
                current_frame_kps_y = (current_frame_kps_xy_raw[:, 1] / frame_height).tolist()
                # Store raw normalized xy pairs for drawing function
                current_frame_kps_xy = np.stack((current_frame_kps_x, current_frame_kps_y), axis=-1).tolist()


            else: # No person/keypoints detected
                current_frame_kps_x = [0.0] * num_keypoints
                current_frame_kps_y = [0.0] * num_keypoints
                current_frame_confs = [0.0] * num_keypoints
                current_frame_kps_xy = [[0.0, 0.0]] * num_keypoints # Placeholder

            all_x_coords.append(current_frame_kps_x)
            all_y_coords.append(current_frame_kps_y)
            all_confidences.append(current_frame_confs)
            timestamps_ms.append(frame_count * frame_time_ms)
        else:
             # Handle frames where results might be empty or lack keypoints structure
            all_x_coords.append([0.0] * num_keypoints)
            all_y_coords.append([0.0] * num_keypoints)
            all_confidences.append([0.0] * num_keypoints)
            timestamps_ms.append(frame_count * frame_time_ms)
            current_frame_kps_xy = [[0.0, 0.0]] * num_keypoints # Placeholder
            current_frame_confs = [0.0] * num_keypoints

        # Draw debug info and write frame if video writer is enabled
        if video_writer:
             # Pass normalized xy keypoints and confidences
            annotated_frame = draw_debug_info(frame, current_frame_kps_xy, current_frame_confs, frame_count)
            video_writer.write(annotated_frame)


        frame_count += 1
        if frame_count % 30 == 0: # Print progress every 30 frames
            print(f"Processed {frame_count} frames...")

    cap.release()
    if video_writer:
        video_writer.release()
        print(f"Finished writing video.")

    print(f"Finished processing {frame_count} frames.")

    # --- Save Raw Data to JSON if requested --- #
    if output_json_path:
        print(f"Saving raw detection data to: {output_json_path}...")
        raw_data = {
            "frame_count": frame_count,
            "fps": fps,
            "timestamps_ms": timestamps_ms,
            "all_x_coords_normalized": all_x_coords,
            "all_y_coords_normalized": all_y_coords,
            "all_confidences": all_confidences
        }
        try:
            with open(output_json_path, 'w') as f:
                json.dump(raw_data, f, indent=4)
            print("Successfully saved JSON data.")
        except Exception as e:
            print(f"Error saving JSON data: {e}")

    if frame_count < 5: # Need sufficient frames for analysis
        print("Error: Video too short or too few keypoints detected for analysis.")
        default_results = {
            "Status": "Failed",
            "Orientation Detected": "N/A",
            "Calculated Jump Height (cm)": 0.0,
            "Calculated Flight Time (s)": 0.0,
            "Take-off Velocity (est.) (m/s)": 0.0,
            "Knee Angle (at lowest point) (degrees)": 0.0,
            "Hip Angle (at peak) (degrees)": 0.0,
            "Error Message": "Video too short or too few keypoints detected for analysis."
        }
        return default_results # Return failure dict

    # --- Metric Calculation (Ported Dart Logic) ---
    print("Calculating metrics...")

    # Keypoint indices (based on COCO format used by YOLOv8)
    # 0: nose, 5: L shoulder, 6: R shoulder, 11: L hip, 12: R hip,
    # 13: L knee, 14: R knee, 15: L ankle, 16: R ankle
    L_HIP_IDX, R_HIP_IDX = 11, 12
    L_KNEE_IDX, R_KNEE_IDX = 13, 14
    L_ANKLE_IDX, R_ANKLE_IDX = 15, 16
    L_SHOULDER_IDX, R_SHOULDER_IDX = 5, 6

    # --- Check for Portrait Mode --- #
    is_portrait = frame_height > frame_width
    if is_portrait:
        print("Portrait mode video detected (height > width).")

    # 1. Determine Orientation (Simplified version)
    # Calculate average X difference between hips and knees over the whole sequence
    # This assumes orientation is consistent throughout the jump.
    valid_frames_count = 0
    avg_right_hip_knee_x_diff = 0
    avg_left_hip_knee_x_diff = 0

    for i in range(frame_count):
         # Check list length before indexing
         has_right_hip = len(all_x_coords[i]) > R_HIP_IDX and len(all_confidences[i]) > R_HIP_IDX and all_confidences[i][R_HIP_IDX] > 0.1
         has_right_knee = len(all_x_coords[i]) > R_KNEE_IDX and len(all_confidences[i]) > R_KNEE_IDX and all_confidences[i][R_KNEE_IDX] > 0.1
         has_left_hip = len(all_x_coords[i]) > L_HIP_IDX and len(all_confidences[i]) > L_HIP_IDX and all_confidences[i][L_HIP_IDX] > 0.1
         has_left_knee = len(all_x_coords[i]) > L_KNEE_IDX and len(all_confidences[i]) > L_KNEE_IDX and all_confidences[i][L_KNEE_IDX] > 0.1


         if has_right_hip and has_right_knee and has_left_hip and has_left_knee:
              avg_right_hip_knee_x_diff += all_x_coords[i][R_HIP_IDX] - all_x_coords[i][R_KNEE_IDX]
              avg_left_hip_knee_x_diff += all_x_coords[i][L_HIP_IDX] - all_x_coords[i][L_KNEE_IDX]
              valid_frames_count +=1

    orientation = "front" # Default
    hip_y_coords, knee_y_coords, ankle_y_coords = [], [], []
    hip_idx, knee_idx, ankle_idx, shoulder_idx = -1, -1, -1, -1 # Indices to use based on orientation

    if valid_frames_count > 0 :
         avg_right_hip_knee_x_diff /= valid_frames_count
         avg_left_hip_knee_x_diff /= valid_frames_count

         # Original orientation logic based on X difference
         if avg_right_hip_knee_x_diff > 0 and avg_left_hip_knee_x_diff > 0:
              temp_orientation = "right"
         elif avg_right_hip_knee_x_diff < 0 and avg_left_hip_knee_x_diff < 0:
              temp_orientation = "left"
         else:
              temp_orientation = "front"

         # --- Override for Portrait Mode --- #
         if is_portrait:
              print(f"Overriding calculated orientation ({temp_orientation}) to 'Front' due to portrait mode.")
              orientation = "front"
         else:
              orientation = temp_orientation
              print(f"Detected orientation: {orientation}")

         # Set indices based on final orientation
         if orientation == "right":
              hip_idx, knee_idx, ankle_idx = L_HIP_IDX, L_KNEE_IDX, L_ANKLE_IDX
              shoulder_idx = L_SHOULDER_IDX
         elif orientation == "left":
              hip_idx, knee_idx, ankle_idx = R_HIP_IDX, R_KNEE_IDX, R_ANKLE_IDX
              shoulder_idx = R_SHOULDER_IDX
         else: # Front
               hip_idx, knee_idx, ankle_idx = -1, -1, -1 # Signal to use averaging logic later

    else:
         print("Warning: Could not determine orientation due to insufficient keypoint data. Assuming Front.")
         orientation = "front"
         hip_idx, knee_idx, ankle_idx = -1, -1, -1 # Signal to use averaging logic later


    # Extract relevant Y coordinates based on orientation
    if orientation == "front":
        # Average left and right side if confidence is good, otherwise use the side with higher confidence
        for i in range(frame_count):
            # Check list lengths before accessing indices
            conf_l_hip = all_confidences[i][L_HIP_IDX] if len(all_confidences[i]) > L_HIP_IDX else 0
            conf_r_hip = all_confidences[i][R_HIP_IDX] if len(all_confidences[i]) > R_HIP_IDX else 0
            conf_l_knee = all_confidences[i][L_KNEE_IDX] if len(all_confidences[i]) > L_KNEE_IDX else 0
            conf_r_knee = all_confidences[i][R_KNEE_IDX] if len(all_confidences[i]) > R_KNEE_IDX else 0
            conf_l_ankle = all_confidences[i][L_ANKLE_IDX] if len(all_confidences[i]) > L_ANKLE_IDX else 0
            conf_r_ankle = all_confidences[i][R_ANKLE_IDX] if len(all_confidences[i]) > R_ANKLE_IDX else 0

            y_hip, y_knee, y_ankle = 0.0, 0.0, 0.0 # Defaults

            has_l_hip = len(all_y_coords[i]) > L_HIP_IDX
            has_r_hip = len(all_y_coords[i]) > R_HIP_IDX
            if has_l_hip and has_r_hip and conf_l_hip > 0.1 and conf_r_hip > 0.1:
                 y_hip = (all_y_coords[i][L_HIP_IDX] + all_y_coords[i][R_HIP_IDX]) / 2
            elif has_l_hip and conf_l_hip >= conf_r_hip: y_hip = all_y_coords[i][L_HIP_IDX]
            elif has_r_hip: y_hip = all_y_coords[i][R_HIP_IDX]

            has_l_knee = len(all_y_coords[i]) > L_KNEE_IDX
            has_r_knee = len(all_y_coords[i]) > R_KNEE_IDX
            if has_l_knee and has_r_knee and conf_l_knee > 0.1 and conf_r_knee > 0.1:
                 y_knee = (all_y_coords[i][L_KNEE_IDX] + all_y_coords[i][R_KNEE_IDX]) / 2
            elif has_l_knee and conf_l_knee >= conf_r_knee: y_knee = all_y_coords[i][L_KNEE_IDX]
            elif has_r_knee: y_knee = all_y_coords[i][R_KNEE_IDX]

            has_l_ankle = len(all_y_coords[i]) > L_ANKLE_IDX
            has_r_ankle = len(all_y_coords[i]) > R_ANKLE_IDX
            if has_l_ankle and has_r_ankle and conf_l_ankle > 0.1 and conf_r_ankle > 0.1:
                 y_ankle = (all_y_coords[i][L_ANKLE_IDX] + all_y_coords[i][R_ANKLE_IDX]) / 2
            elif has_l_ankle and conf_l_ankle >= conf_r_ankle: y_ankle = all_y_coords[i][L_ANKLE_IDX]
            elif has_r_ankle: y_ankle = all_y_coords[i][R_ANKLE_IDX]


            hip_y_coords.append([y_hip]) # Wrap in list to match structure needed by get_peak_index etc.
            knee_y_coords.append([y_knee])
            ankle_y_coords.append([y_ankle])
        # Indices for angle calculations when front facing (use left side arbitrarily, angle logic might need adjustment)
        hip_idx, knee_idx, ankle_idx, shoulder_idx = L_HIP_IDX, L_KNEE_IDX, L_ANKLE_IDX, L_SHOULDER_IDX

    else: # Left or Right orientation
        for i in range(frame_count):
             y_hip = all_y_coords[i][hip_idx] if len(all_y_coords[i]) > hip_idx else 0
             y_knee = all_y_coords[i][knee_idx] if len(all_y_coords[i]) > knee_idx else 0
             y_ankle = all_y_coords[i][ankle_idx] if len(all_y_coords[i]) > ankle_idx else 0
             hip_y_coords.append([y_hip]) # Wrap in list
             knee_y_coords.append([y_knee])
             ankle_y_coords.append([y_ankle])


    # 2. Find Peak, Start, End Indices
    # Note: get_peak_index etc expect a list of lists structure for y_coords,
    # and the keypoint_index is always 0 because we extracted only the relevant y coord.
    # Confidence check needs the original all_confidences and the correct side index (hip_idx/knee_idx)
    peak_conf_check_idx_hip = L_HIP_IDX if orientation == "front" else hip_idx # Use L_HIP confidence for front avg check
    # Pass is_front_view flag to get_peak_index
    jump_peak_index_hip = get_peak_index(hip_y_coords, all_confidences, peak_conf_check_idx_hip, is_front_view=(orientation == "front"))

    peak_conf_check_idx_knee = L_KNEE_IDX if orientation == "front" else knee_idx # Use L_KNEE confidence for front avg check
    # Pass is_front_view flag to get_peak_index
    jump_peak_index_knee = get_peak_index(knee_y_coords, all_confidences, peak_conf_check_idx_knee, is_front_view=(orientation == "front"))

    # peak_conf_check_idx = L_ANKLE_IDX if orientation == "front" else ankle_idx
    # jump_peak_index_ankle = get_peak_index(ankle_y_coords, all_confidences, peak_conf_check_idx) # Ankle peak not directly used in Dart height calc

    # Indices are relative to the *filtered* y_coords list (hip_y_coords etc.)
    jump_start_index_hip = get_start_index(hip_y_coords, jump_peak_index_hip, 0)
    jump_start_index_knee = get_start_index(knee_y_coords, jump_peak_index_knee, 0)

    jump_end_index_hip = get_end_index(hip_y_coords, jump_peak_index_hip, 0)
    jump_end_index_knee = get_end_index(knee_y_coords, jump_peak_index_knee, 0)

    # Basic validation of indices
    if not (0 < jump_start_index_hip < jump_peak_index_hip < jump_end_index_hip < frame_count):
         print(f"Error: Invalid hip jump indices: Start={jump_start_index_hip}, Peak={jump_peak_index_hip}, End={jump_end_index_hip}. Cannot calculate metrics reliably.")
         # Output default metrics and return
         default_results = {
             "Status": "Failed",
             "Orientation Detected": "N/A",
             "Calculated Jump Height (cm)": 0.0,
             "Calculated Flight Time (s)": 0.0,
             "Take-off Velocity (est.) (m/s)": 0.0,
             "Knee Angle (at lowest point) (degrees)": 0.0,
             "Hip Angle (at peak) (degrees)": 0.0,
             "Error Message": f"Invalid hip jump indices: Start={jump_start_index_hip}, Peak={jump_peak_index_hip}, End={jump_end_index_hip}"
         }
         return default_results # Return failure dict


    # Adjust indices for safety, ensuring they are within bounds for timestamp access
    # Dart used start+1 and end-1. Let's try that.
    start_hip_ts_idx = min(max(0, jump_start_index_hip + 1), frame_count - 1)
    end_hip_ts_idx = min(max(0, jump_end_index_hip - 1), frame_count - 1)
    start_knee_ts_idx = min(max(0, jump_start_index_knee + 1), frame_count - 1)
    end_knee_ts_idx = min(max(0, jump_end_index_knee - 1), frame_count - 1)

    # Ensure start < end after adjustments
    if start_hip_ts_idx >= end_hip_ts_idx:
         print(f"Warning: Hip start index ({start_hip_ts_idx}) >= end index ({end_hip_ts_idx}) after adjustment. Setting flight time to 0.")
         time_in_air_hips_ms = 0
    else:
         time_in_air_hips_ms = timestamps_ms[end_hip_ts_idx] - timestamps_ms[start_hip_ts_idx]

    if start_knee_ts_idx >= end_knee_ts_idx:
         print(f"Warning: Knee start index ({start_knee_ts_idx}) >= end index ({end_knee_ts_idx}) after adjustment. Setting flight time to 0.")
         time_in_air_knee_ms = 0
    else:
         time_in_air_knee_ms = timestamps_ms[end_knee_ts_idx] - timestamps_ms[start_knee_ts_idx]



    # 3. Calculate Flight Time
    # time_in_air_hips_ms = timestamps_ms[end_hip_ts_idx] - timestamps_ms[start_hip_ts_idx]
    # time_in_air_knee_ms = timestamps_ms[end_knee_ts_idx] - timestamps_ms[start_knee_ts_idx]

    time_in_air_hips = time_in_air_hips_ms / 1000.0 # Convert to seconds
    time_in_air_knee = time_in_air_knee_ms / 1000.0

    # 4. Calculate Jump Height
    # Formula: height = (1/8) * g * time^2 = 0.125 * 9.81 * time^2 (height in meters)
    # Ensure time is non-negative before squaring
    jump_height_hip_m = 0.125 * 9.81 * (max(0, time_in_air_hips) ** 2)
    jump_height_knee_m = 0.125 * 9.81 * (max(0, time_in_air_knee) ** 2)


    # Combine hip and knee heights (logic from Dart)
    jump_height_m = (jump_height_hip_m + jump_height_knee_m) / 2.0
    time_in_air_s = (time_in_air_hips + time_in_air_knee) / 2.0

    # If knee height is significantly different, use hip height (Dart logic: 0.15 -> 15%)
    # Check hip_m > 0 to avoid division by zero or weird ratios if hip height is 0
    if jump_height_hip_m > 1e-3 and (jump_height_knee_m < jump_height_hip_m * 0.85 or jump_height_knee_m > jump_height_hip_m * 1.15):
        print("Knee height differs significantly from hip height, using hip-based calculation.")
        jump_height_m = jump_height_hip_m
        time_in_air_s = time_in_air_hips

    jump_height_cm = jump_height_m * 100 # Convert to cm

    # 5. Validity Checks (Similar to Dart)
    # Check confidence scores (e.g., low confidence for hips in many frames)
    low_confidence_hip_count = 0
    hip_indices_to_check = [L_HIP_IDX, R_HIP_IDX]
    for i in range(frame_count):
        # Check list length before indexing
        if len(all_confidences[i]) > max(hip_indices_to_check):
             if all_confidences[i][L_HIP_IDX] < 0.3 and all_confidences[i][R_HIP_IDX] < 0.3:
                  low_confidence_hip_count += 1

    confidence_threshold = 0.6 # 60% of frames with low confidence
    if low_confidence_hip_count > confidence_threshold * frame_count:
         print(f"Warning: Low confidence for hip keypoints in over {confidence_threshold*100:.0f}% of frames ({low_confidence_hip_count}/{frame_count}). Results may be inaccurate.")
         # Consider returning or marking results as unreliable

    # Check jump height range and time in air
    if jump_height_cm < 2 or jump_height_cm > 140 or time_in_air_s <= 0: # time_in_air must be positive
        print(f"Error: Calculated metrics out of range: Height={jump_height_cm:.1f} cm, TimeInAir={time_in_air_s:.3f} s. Invalid jump detected.")
        # Handle error - maybe return default values or raise an exception
        default_results = {
            "Status": "Failed",
            "Orientation Detected": "N/A",
            "Calculated Jump Height (cm)": 0.0,
            "Calculated Flight Time (s)": 0.0,
            "Take-off Velocity (est.) (m/s)": 0.0,
            "Knee Angle (at lowest point) (degrees)": 0.0,
            "Hip Angle (at peak) (degrees)": 0.0,
            "Error Message": f"Calculated metrics out of range: Height={jump_height_cm:.1f} cm, TimeInAir={time_in_air_s:.3f} s"
        }
        return default_results # Return failure dict


    # 6. Calculate Pixel-to-Meter Ratio and Velocity
    velocity_mps = 0.0
    ratio = 0.0
    if jump_start_index_hip > 0 and jump_peak_index_hip < frame_count:
        # Ensure hip coordinates exist at start and peak indices
        if len(hip_y_coords) > jump_start_index_hip and len(hip_y_coords[jump_start_index_hip]) > 0 and \
           len(hip_y_coords) > jump_peak_index_hip and len(hip_y_coords[jump_peak_index_hip]) > 0:

            # Y coordinates are normalized (0=top, 1=bottom), so peak Y is smaller
            jump_height_pixels = hip_y_coords[jump_start_index_hip][0] - hip_y_coords[jump_peak_index_hip][0]

            if jump_height_pixels > 1e-6: # Avoid division by zero or negative height
                ratio = jump_height_m / jump_height_pixels # meters per pixel (normalized coord)

                # Velocity calculation (based on Dart logic: hip movement just before takeoff)
                idx_prev = max(0, jump_start_index_hip - 1)
                idx_curr = jump_start_index_hip

                # Ensure coordinates and timestamps exist for these indices
                if idx_prev != idx_curr and \
                   len(hip_y_coords) > idx_curr and len(hip_y_coords[idx_curr]) > 0 and \
                   len(hip_y_coords) > idx_prev and len(hip_y_coords[idx_prev]) > 0 and \
                   len(timestamps_ms) > idx_curr and len(timestamps_ms) > idx_prev: # Check timestamps_ms too

                    distance_pixels = hip_y_coords[idx_prev][0] - hip_y_coords[idx_curr][0] # Y difference
                    distance_meters = distance_pixels * ratio
                    time_diff_s = (timestamps_ms[idx_curr] - timestamps_ms[idx_prev]) / 1000.0

                    if time_diff_s > 1e-6: # Avoid division by zero
                        velocity_mps = abs(distance_meters / time_diff_s)
                    else:
                        print("Warning: Time difference for velocity calculation is zero or negative.")
                else:
                     print("Warning: Could not calculate velocity due to missing data or invalid indices at takeoff frames.")
            else:
                print("Warning: Jump height in normalized pixels is zero or negative. Cannot calculate ratio or velocity.")
        else:
            print("Warning: Missing hip coordinates at start or peak index. Cannot calculate ratio or velocity.")


    # 7. Calculate Angles
    knee_angle_deg = 0.0
    hip_angle_deg = 0.0

    # Find lowest point before jump (knee angle) -> highest Y coordinate for hip
    knee_angle_index = 0
    max_y_before_jump = -float('inf')
    conf_threshold_angle = 0.2 # Confidence threshold for points used in angle calculation

    if jump_start_index_hip > 0 :
        # Use the appropriate hip y coordinate list based on orientation
        # Need original coordinates for finding the index
        y_coords_for_lowest_point = all_y_coords # Use original Y coords list
        idx_for_lowest_point = hip_idx if orientation != "front" else -1 # Use side index, or handle front case

        for i in range(jump_start_index_hip):
            # Determine which hip index to check based on orientation/confidence for front view
            if orientation == "front":
                 # Check both hips' confidence
                 conf_l = all_confidences[i][L_HIP_IDX] if len(all_confidences[i]) > L_HIP_IDX else 0
                 conf_r = all_confidences[i][R_HIP_IDX] if len(all_confidences[i]) > R_HIP_IDX else 0
                 # Use the index of the more confident hip, if one is clearly better
                 if conf_l > conf_threshold_angle and conf_l >= conf_r:
                      idx_for_lowest_point = L_HIP_IDX
                 elif conf_r > conf_threshold_angle and conf_r > conf_l:
                      idx_for_lowest_point = R_HIP_IDX
                 else: # Both low confidence or equal, skip frame for max_y check
                      idx_for_lowest_point = -1
            else: # Left/Right view, index already set
                 idx_for_lowest_point = hip_idx

            # Proceed if we have a valid index and data for it
            if idx_for_lowest_point != -1 and \
               len(y_coords_for_lowest_point) > i and \
               len(y_coords_for_lowest_point[i]) > idx_for_lowest_point and \
               len(all_confidences[i]) > idx_for_lowest_point:

                current_y = y_coords_for_lowest_point[i][idx_for_lowest_point]
                current_conf = all_confidences[i][idx_for_lowest_point]

                if current_conf > conf_threshold_angle: # Check confidence of the point being used
                    if current_y > max_y_before_jump:
                         max_y_before_jump = current_y
                         knee_angle_index = i


    # Calculate knee angle at that lowest point (knee_angle_index)
    # We need the original coordinates (all_x_coords, all_y_coords) for angle calculation
    # Use the side indices determined earlier (hip_idx, knee_idx, ankle_idx)
    # Ensure hip_idx, knee_idx, ankle_idx are valid before proceeding
    required_indices_knee = [hip_idx, knee_idx, ankle_idx]
    if -1 not in required_indices_knee and \
       len(all_x_coords) > knee_angle_index and \
       len(all_y_coords) > knee_angle_index and \
       len(all_x_coords[knee_angle_index]) > max(required_indices_knee) and \
       len(all_y_coords[knee_angle_index]) > max(required_indices_knee) and \
       len(all_confidences[knee_angle_index]) > max(required_indices_knee): # Check confidences list length too

        # Check confidences of the three points
        conf_hip = all_confidences[knee_angle_index][hip_idx]
        conf_knee = all_confidences[knee_angle_index][knee_idx]
        conf_ankle = all_confidences[knee_angle_index][ankle_idx]

        if conf_hip > conf_threshold_angle and conf_knee > conf_threshold_angle and conf_ankle > conf_threshold_angle:
            hip_pt = (all_x_coords[knee_angle_index][hip_idx], all_y_coords[knee_angle_index][hip_idx])
            knee_pt = (all_x_coords[knee_angle_index][knee_idx], all_y_coords[knee_angle_index][knee_idx])
            ankle_pt = (all_x_coords[knee_angle_index][ankle_idx], all_y_coords[knee_angle_index][ankle_idx])
            # Angle is at the knee
            knee_angle_deg = find_angle(knee_pt, hip_pt, ankle_pt)
        else:
            print(f"Warning: Low confidence for points needed for knee angle at index {knee_angle_index}. Confs: Hip={conf_hip:.2f}, Knee={conf_knee:.2f}, Ankle={conf_ankle:.2f}")
    else:
         print(f"Warning: Could not calculate knee angle. Index {knee_angle_index} out of bounds, missing keypoints, or orientation ambiguous (required indices: {required_indices_knee}).")


    # Calculate hip angle at jump peak (jump_peak_index_hip)
    # Use the side indices determined earlier (hip_idx, knee_idx, shoulder_idx)
    required_indices_hip = [hip_idx, knee_idx, shoulder_idx]
    if -1 not in required_indices_hip and \
       len(all_x_coords) > jump_peak_index_hip and \
       len(all_y_coords) > jump_peak_index_hip and \
       len(all_x_coords[jump_peak_index_hip]) > max(required_indices_hip) and \
       len(all_y_coords[jump_peak_index_hip]) > max(required_indices_hip) and \
       len(all_confidences[jump_peak_index_hip]) > max(required_indices_hip): # Check confidences list length

         # Check confidences
        conf_hip = all_confidences[jump_peak_index_hip][hip_idx]
        conf_knee = all_confidences[jump_peak_index_hip][knee_idx]
        conf_shoulder = all_confidences[jump_peak_index_hip][shoulder_idx]

        if conf_hip > conf_threshold_angle and conf_knee > conf_threshold_angle and conf_shoulder > conf_threshold_angle:
             hip_pt = (all_x_coords[jump_peak_index_hip][hip_idx], all_y_coords[jump_peak_index_hip][hip_idx])
             knee_pt = (all_x_coords[jump_peak_index_hip][knee_idx], all_y_coords[jump_peak_index_hip][knee_idx])
             shoulder_pt = (all_x_coords[jump_peak_index_hip][shoulder_idx], all_y_coords[jump_peak_index_hip][shoulder_idx])
             # Angle is at the hip
             raw_hip_angle = find_angle(hip_pt, shoulder_pt, knee_pt)
             # Apply Dart's 360 - angle logic (needs verification if this interpretation is correct)
             hip_angle_deg = 360.0 - raw_hip_angle if raw_hip_angle > 0 else 0.0
        else:
            print(f"Warning: Low confidence for points needed for hip angle at index {jump_peak_index_hip}. Confs: Hip={conf_hip:.2f}, Knee={conf_knee:.2f}, Shoulder={conf_shoulder:.2f}")
    else:
         print(f"Warning: Could not calculate hip angle. Index {jump_peak_index_hip} out of bounds, missing keypoints, or orientation ambiguous (required indices: {required_indices_hip}).")


    # 8. Prepare Results Dictionary
    results_dict = {
        "Status": "Success",
        "Orientation Detected": orientation,
        "Calculated Jump Height (cm)": round(jump_height_cm, 1),
        "Calculated Flight Time (s)": round(time_in_air_s, 3),
        "Take-off Velocity (est.) (m/s)": round(velocity_mps, 2),
        "Knee Angle (at lowest point) (degrees)": round(knee_angle_deg, 1),
        "Hip Angle (at peak) (degrees)": round(hip_angle_deg, 1),
        "Error Message": "" # No error for success
    }

    # Print Results to console (optional, but good for progress)
    print("\n--- Jump Analysis Results ---")
    for key, value in results_dict.items():
        if key != "Error Message" or value: # Don't print empty error message
             print(f"{key}: {value}")
    print("-----------------------------")

    return results_dict


# --- Argument Parsing and Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze jump performance from videos (1.mp4, 2.mp4, 3.mp4) in a directory using YOLO Pose.")
    parser.add_argument("--input_dir", help="Path to the root directory containing video files.", required=True)
    parser.add_argument("-m", "--model", default="yolo11m-pose.pt", help="Path to the YOLO11 Pose model file (e.g., yolo11m-pose.pt).")
    parser.add_argument("--save_video", action='store_true', help="Save annotated videos (e.g., 1_annotated.mp4) alongside originals.")
    parser.add_argument("--save_json", action='store_true', help="Save raw detection results as JSON (e.g., 1_results.json) alongside originals.")

    args = parser.parse_args()

    # Check if input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at {args.input_dir}")
        exit()

    # Check if model file exists
    if not os.path.exists(args.model):
         print(f"Error: Model file not found at {args.model}")
         print("Please download it or provide the correct path.")
         exit()

    target_videos = ["1.mp4", "2.mp4", "3.mp4"]
    target_csvs = ["1.csv", "2.csv", "3.csv"]
    processed_count = 0
    failed_count = 0

    print(f"Starting analysis in directory: {args.input_dir}")
    print(f"Looking for videos: {', '.join(target_videos)}")

    for root, dirs, files in os.walk(args.input_dir):
        # Avoid processing .DS_Store and other hidden files/folders if necessary
        files = [f for f in files if not f.startswith('.')]
        dirs[:] = [d for d in dirs if not d.startswith('.')] # Modifies dirs in-place for os.walk

        for video_filename in target_videos:
            if video_filename in files:
                video_path = os.path.join(root, video_filename)
                base_name = os.path.splitext(video_filename)[0]
                csv_filename = f"{base_name}.csv"
                csv_path = os.path.join(root, csv_filename)

                print(f"\nProcessing video: {video_path}")

                # Determine output paths if flags are set
                output_video_path = None
                if args.save_video:
                    output_video_filename = f"{base_name}_annotated.mp4"
                    output_video_path = os.path.join(root, output_video_filename)

                output_json_path = None
                if args.save_json:
                    output_json_filename = f"{base_name}_results.json"
                    output_json_path = os.path.join(root, output_json_filename)

                # Process the video
                results_dict = process_video(video_path, args.model, output_video_path, output_json_path)

                # Write results to CSV
                try:
                    # Ensure the directory exists (though os.walk found it)
                    os.makedirs(root, exist_ok=True)
                    with open(csv_path, 'w', newline='') as csvfile:
                        if results_dict: # Check if dictionary is not empty/None
                            fieldnames = results_dict.keys()
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerow(results_dict)
                            print(f"Results saved to: {csv_path}")
                            if results_dict.get("Status") == "Success":
                                processed_count += 1
                            else:
                                failed_count += 1
                                print(f"Analysis failed for {video_path}. Reason: {results_dict.get('Error Message', 'Unknown')}")

                        else:
                            print(f"Warning: No results dictionary returned for {video_path}. Skipping CSV writing.")
                            failed_count += 1

                except Exception as e:
                    print(f"Error writing CSV file {csv_path}: {e}")
                    failed_count += 1

    print(f"\n--- Analysis Complete ---")
    print(f"Successfully processed videos: {processed_count}")
    print(f"Failed analyses: {failed_count}")
    print(f"Output CSVs generated in respective subdirectories.") 