⚽ Mark My Move — Computer Vision Player Tracking App
--------------------------------------------------

This is a code repository for the CS 366 Final Project.

You can run this code by using the command: streamlit run app.py

Mark My Move is an end-to-end computer vision application for tracking an individual athlete across a sports video. Built with Python, YOLOv8, OpenCV, and Streamlit, the app enables user-guided player selection and produces a fully annotated tracking video with motion analytics.

This project demonstrates applied skills in object detection, video processing, stateful application design, and algorithmic tracking under uncertainty.

--------------------------------------------------
🔍 Project Overview
--------------------------------------------------

• Detects all players in a video using YOLOv8  
• Allows the user to select a target player from the first frame  
• Tracks the selected player across frames using:
  - Centroid distance matching
  - Color-based re-identification (HSV)
  - Velocity-based motion prediction
• Outputs a downloadable video with visual tracking overlays

--------------------------------------------------
🧠 Technical Highlights
--------------------------------------------------

• Deep Learning Integration  
  Integrated a pre-trained YOLOv8 model for real-time person detection in video frames.

• Robust Tracking Logic  
  Implemented adaptive nearest-neighbor tracking with velocity estimation, exponential smoothing, and temporary occlusion handling through motion prediction.

• Feature Engineering for Re-Identification  
  Computed average HSV color signatures to filter detections and improve player consistency across frames.

• Interactive ML Application  
  Designed a multi-step Streamlit interface using session state to manage user-driven workflows.

• Performance-Aware Video Processing  
  Streamed video frames efficiently while generating an annotated MP4 output.

--------------------------------------------------
🛠️ Tech Stack
--------------------------------------------------

Languages:
• Python

Frameworks & Libraries:
• Streamlit
• OpenCV
• NumPy
• Ultralytics YOLOv8

Core Concepts:
• Computer Vision
• Object Detection
• Motion Tracking
• State Management
• Video Analytics

--------------------------------------------------
⚙️ How It Works (High Level)
--------------------------------------------------

1. User uploads a sports video
2. YOLOv8 detects all players in the first frame
3. User selects a player ID to track
4. App tracks the player using color filtering and motion prediction
5. Annotated video is generated and made available for download

--------------------------------------------------
📈 Optional Analytics & Visualizations
--------------------------------------------------

• Movement trail overlay  
• Estimated speed (pixels per second)  
• Zoomed-in tracking window  

All features can be toggled on or off within the app.

--------------------------------------------------
🚧 Known Limitations
--------------------------------------------------

• Tracks a single player at a time  
• Speed is estimated in pixel units (no real-world calibration)  
• Accuracy depends on visibility and color distinctiveness

--------------------------------------------------
🔮 Potential Extensions
--------------------------------------------------

• Multi-player tracking and ID assignment  
• Real-world speed and distance estimation  
• Team-level analysis (formations, spacing)  
• Improved re-identification using deep appearance embeddings

--------------------------------------------------
📌 Why This Project Matters
--------------------------------------------------

This project showcases my ability to:  
• Apply machine learning models to real-world video data  
• Build complete, user-facing ML applications  
• Design algorithms that handle noisy and imperfect inputs  
• Communicate technical work clearly through interactive tools  
