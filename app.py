import streamlit as st
import tempfile
import cv2
import numpy as np
from ultralytics import YOLO

# -------------------------------------------------------
# Streamlit Page Setup
# -------------------------------------------------------
st.set_page_config(page_title="Player Tracking App", layout="wide")
st.title("Mark My Move ⚽")

# -------------------------------------------------------
# Session State Initialization
# -------------------------------------------------------
if "player_selected" not in st.session_state:
    st.session_state.player_selected = False

if "selected_player_id" not in st.session_state:
    st.session_state.selected_player_id = None

if "first_frame_boxes" not in st.session_state:
    st.session_state.first_frame_boxes = None

if "player_color_signature" not in st.session_state:
    st.session_state.player_color_signature = None

# -------------------------------------------------------
# Load YOLO model
# -------------------------------------------------------
model = YOLO("yolov8n.pt")  # Upgrade for accuracy if needed

# -------------------------------------------------------
# Helper functions
# -------------------------------------------------------
def detect_players(frame):
    """Run YOLO and return bounding boxes for PERSON class."""
    results = model(frame, verbose=False)
    boxes = []
    for r in results:
        for b in r.boxes:
            if int(b.cls) == 0:  # person class
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                boxes.append((x1, y1, x2, y2))
    return boxes

def get_centroid(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def find_nearest_box(target_centroid, boxes, max_distance=120):
    """Return nearest box within distance threshold."""
    if not boxes:
        return None
    tx, ty = target_centroid
    best = min(
        boxes,
        key=lambda b: (get_centroid(b)[0] - tx) ** 2 + (get_centroid(b)[1] - ty) ** 2
    )
    cx, cy = get_centroid(best)
    distance = np.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)
    if distance < max_distance:
        return best
    return None

def get_color_signature(frame, box):
    """Compute average HSV color inside the bounding box."""
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_mean = int(np.mean(hsv[:, :, 0]))
    s_mean = int(np.mean(hsv[:, :, 1]))
    v_mean = int(np.mean(hsv[:, :, 2]))
    return (h_mean, s_mean, v_mean)

def color_match(frame, box, signature, threshold=30):
    """Check if box matches color signature."""
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_mean = int(np.mean(hsv[:, :, 0]))
    s_mean = int(np.mean(hsv[:, :, 1]))
    v_mean = int(np.mean(hsv[:, :, 2]))
    dist = np.sqrt((h_mean - signature[0])**2 + (s_mean - signature[1])**2 + (v_mean - signature[2])**2)
    return dist < threshold

# -------------------------------------------------------
# File Upload
# -------------------------------------------------------
uploaded = st.file_uploader("Upload a sports video", type=["mp4", "mov", "avi", "mkv"])
if not uploaded:
    st.stop()

# Save temp file
input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
input_temp.write(uploaded.read())

cap = cv2.VideoCapture(input_temp.name)
ret, first_frame = cap.read()
if not ret:
    st.error("Could not read video.")
    st.stop()

# -------------------------------------------------------
# Step 1: Player Selection
# -------------------------------------------------------
if not st.session_state.player_selected:
    st.subheader("Step 1: Select the player you want to track")

    if st.session_state.first_frame_boxes is None:
        st.session_state.first_frame_boxes = detect_players(first_frame)

    boxes = st.session_state.first_frame_boxes

    if len(boxes) == 0:
        st.error("No players detected in the first frame.")
        st.stop()

    # Draw numbered boxes
    preview = first_frame.copy()
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(preview, f"ID {idx}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    st.image(preview, caption="Detected players (ID labels)", channels="BGR")

    player_id = st.number_input(
        "Enter the ID of the player to track:",
        min_value=0,
        max_value=len(boxes) - 1,
        step=1
    )

    if st.button("Start Tracking"):
        st.session_state.player_selected = True
        st.session_state.selected_player_id = int(player_id)
        st.session_state.player_color_signature = get_color_signature(first_frame, boxes[player_id])
        st.rerun()

    st.stop()

# -------------------------------------------------------
# Step 2: Tracking
# -------------------------------------------------------
player_id = st.session_state.selected_player_id
target_box = st.session_state.first_frame_boxes[player_id]
target_centroid = get_centroid(target_box)
player_color_signature = st.session_state.player_color_signature

st.subheader(f"Tracking player **ID {player_id}**…")

show_trail = st.checkbox("Show movement trail")
show_zoom = st.checkbox("Add zoom window")
show_speed = st.checkbox("Show estimated speed")

trail_points = []

prev_centroid = target_centroid
predicted_centroid = target_centroid
velocity = (0, 0)
lost_frames = 0
max_lost_frames = 15
alpha = 0.6

BASE_DISTANCE = 80
SPEED_FACTOR = 0.5

output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_temp.name, fourcc, fps, (width, height))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
progress = st.progress(0)
current_frame = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# -------------------------------------------------------
# Main Tracking Loop
# -------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    boxes = detect_players(frame)
    # Filter by color first
    boxes = [b for b in boxes if color_match(frame, b, player_color_signature, threshold=30)]

    # Adaptive max distance
    dx_vel = velocity[0]
    dy_vel = velocity[1]
    speed = np.sqrt(dx_vel**2 + dy_vel**2)
    adaptive_max_distance = int(BASE_DISTANCE + SPEED_FACTOR * speed)

    new_box = find_nearest_box(predicted_centroid, boxes, max_distance=adaptive_max_distance)

    if new_box:
        lost_frames = 0
        new_cx, new_cy = get_centroid(new_box)

        # Update velocity
        dx = new_cx - prev_centroid[0]
        dy = new_cy - prev_centroid[1]
        velocity = (dx, dy)

        # Smooth centroid
        cx = int(alpha * new_cx + (1 - alpha) * prev_centroid[0])
        cy = int(alpha * new_cy + (1 - alpha) * prev_centroid[1])
        predicted_centroid = (cx, cy)
        prev_centroid = predicted_centroid
        target_box = new_box

    else:
        # Temporary loss: predict
        lost_frames += 1
        cx = predicted_centroid[0] + int(velocity[0])
        cy = predicted_centroid[1] + int(velocity[1])
        predicted_centroid = (cx, cy)
        prev_centroid = predicted_centroid
        cv2.putText(frame, f"Player temporarily lost ({lost_frames})", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if lost_frames >= max_lost_frames:
            st.warning("Player lost for too long. Stopping tracking.")
            break

    # Draw circle
    cv2.circle(frame, (cx, cy), 30, (0, 0, 255), 4)

    # Trail
    if show_trail:
        trail_points.append((cx, cy))
        for i in range(1, len(trail_points)):
            cv2.line(frame, trail_points[i - 1], trail_points[i], (0, 255, 255), 3)

    # Speed
    if show_speed and len(trail_points) > 1:
        dx_spd = cx - trail_points[-2][0]
        dy_spd = cy - trail_points[-2][1]
        speed_px = np.sqrt(dx_spd**2 + dy_spd**2) * fps
        cv2.putText(frame, f"Speed: {speed_px:.1f} px/s", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Zoom
    if show_zoom:
        size = 80
        x1 = max(cx - size, 0)
        y1 = max(cy - size, 0)
        x2 = min(cx + size, frame.shape[1])
        y2 = min(cy + size, frame.shape[0])
        zoom = frame[y1:y2, x1:x2]
        zoom = cv2.resize(zoom, (200, 200))
        frame[20:220, 20:220] = zoom

    out.write(frame)
    current_frame += 1
    progress.progress(current_frame / frame_count)

# -------------------------------------------------------
# Final Output
# -------------------------------------------------------
cap.release()
out.release()

st.success("Tracking complete!")
st.video(output_temp.name)

with open(output_temp.name, "rb") as f:
    st.download_button("Download Tracked Video", f, file_name="tracked_output.mp4")
