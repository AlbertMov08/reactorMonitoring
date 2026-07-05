#!/usr/bin/env python3
"""
apply_camera_patch.py

Run this in the same folder as your WORKING 1.3k-line cnn_inference.py.
It makes a backup, then ONLY inserts the webcam camera feature into the
existing file. It does not replace the whole file or remove your CSS/JS.
"""

from pathlib import Path
import sys

TARGET = Path("cnn_inference.py")

if not TARGET.exists():
    print("ERROR: cnn_inference.py not found in this folder.")
    print("Run: python apply_camera_patch.py from the project folder.")
    sys.exit(1)

text = TARGET.read_text()

if "const startWebcam = async () =>" in text:
    print("Camera patch already appears to be installed. No changes made.")
    sys.exit(0)

backup = TARGET.with_suffix(".py.before_camera_backup")
backup.write_text(text)
print(f"Backup saved to: {backup}")

# 1) Add webcam refs/state after existing refs.
old_refs = """      const videoRef = useRef(null);
      const canvasRef = useRef(null);
      const tokenClientRef = useRef(null);
      const createdBlobUrlRef = useRef(null);
"""

new_refs = """      const videoRef = useRef(null);
      const canvasRef = useRef(null);
      const webcamStreamRef = useRef(null);
      const tokenClientRef = useRef(null);
      const createdBlobUrlRef = useRef(null);
      const [isWebcam, setIsWebcam] = useState(false);
"""

if old_refs not in text:
    print("ERROR: Could not find the original ref block. Nothing changed.")
    print("Look for this area in App(): videoRef, canvasRef, tokenClientRef, createdBlobUrlRef")
    sys.exit(1)

text = text.replace(old_refs, new_refs, 1)

# 2) Add start/stop webcam helpers right before revokeCurrentBlobUrl.
webcam_helpers = """      const startWebcam = async () => {
        try {
          if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert("Your browser does not support webcam access. Use Chrome/Safari on localhost or HTTPS.");
            return;
          }

          const stream = await navigator.mediaDevices.getUserMedia({
            video: {
              width: { ideal: 1280 },
              height: { ideal: 720 },
              facingMode: "environment"
            },
            audio: false
          });

          webcamStreamRef.current = stream;

          const video = videoRef.current;
          if (video) {
            video.pause();
            video.srcObject = stream;
            video.removeAttribute("src");
            video.load();
            await video.play();
          }

          setSelectedFile(null);
          setVideoSourceName("Live webcam");
          setVideoSourceType("Camera");
          setDriveStatus("Webcam active. Live foam scanning is running.");
          setIsWebcam(true);
        } catch (err) {
          console.error("Webcam error:", err);
          alert("Could not access webcam: " + err.message);
        }
      };

      const stopWebcam = () => {
        if (webcamStreamRef.current) {
          webcamStreamRef.current.getTracks().forEach((track) => track.stop());
          webcamStreamRef.current = null;
        }

        const video = videoRef.current;
        if (video && video.srcObject) {
          video.srcObject = null;
        }

        setIsWebcam(false);
      };

"""

marker = "      const revokeCurrentBlobUrl = () => {"
if marker not in text:
    print("ERROR: Could not find revokeCurrentBlobUrl. Nothing changed.")
    sys.exit(1)

text = text.replace(marker, webcam_helpers + marker, 1)

# 3) Stop webcam when switching back to local video.
old_reset = """      const resetToDefaultVideo = () => {
        revokeCurrentBlobUrl();
"""
new_reset = """      const resetToDefaultVideo = () => {
        stopWebcam();
        revokeCurrentBlobUrl();
"""

if old_reset not in text:
    print("ERROR: Could not patch resetToDefaultVideo. Nothing changed.")
    sys.exit(1)

text = text.replace(old_reset, new_reset, 1)

# Make sure default video clears srcObject before setting src.
old_reset_video = """        if (video) {
          video.pause();
          video.src = "/static/test1.mp4";
          video.load();
        }
"""
new_reset_video = """        if (video) {
          video.pause();
          video.srcObject = null;
          video.src = "/static/test1.mp4";
          video.load();
        }
"""

if old_reset_video in text:
    text = text.replace(old_reset_video, new_reset_video, 1)

# 4) Stop webcam when switching to a Google Drive video.
old_play = """      const playDriveFile = async (file) => {
        try {
"""
new_play = """      const playDriveFile = async (file) => {
        stopWebcam();
        try {
"""

if old_play not in text:
    print("ERROR: Could not patch playDriveFile. Nothing changed.")
    sys.exit(1)

text = text.replace(old_play, new_play, 1)

# Make sure Drive video clears srcObject before setting src.
old_drive_video = """          if (video) {
            video.pause();
            video.src = objectUrl;
            video.load();
            video.play().catch(() => {});
          }
"""
new_drive_video = """          if (video) {
            video.pause();
            video.srcObject = null;
            video.src = objectUrl;
            video.load();
            video.play().catch(() => {});
          }
"""

if old_drive_video in text:
    text = text.replace(old_drive_video, new_drive_video, 1)

# 5) Add webcam button to toolbar before Choose Drive Folder.
toolbar_marker = """            <div className="toolbar">
              <button onClick={chooseDriveFolder} disabled={!driveReady || pickerBusy || !GOOGLE_CLIENT_ID || !GOOGLE_API_KEY}>
"""
toolbar_replacement = """            <div className="toolbar">
              <button onClick={startWebcam} disabled={isWebcam}>
                {isWebcam ? "Webcam Active" : "Use Webcam"}
              </button>
              <button onClick={chooseDriveFolder} disabled={!driveReady || pickerBusy || !GOOGLE_CLIENT_ID || !GOOGLE_API_KEY}>
"""

if toolbar_marker not in text:
    print("ERROR: Could not find toolbar button area. Nothing changed.")
    sys.exit(1)

text = text.replace(toolbar_marker, toolbar_replacement, 1)

# 6) Make the live scan loop wait until a real frame exists.
old_loop = """              if (!video.paused && !video.ended) {
                fetchClassification();
              }
"""
new_loop = """              if (!video.paused && !video.ended && video.readyState >= 2) {
                fetchClassification();
              }
"""

if old_loop in text:
    text = text.replace(old_loop, new_loop, 1)
else:
    print("WARNING: Could not find the exact polling loop; continuing without changing it.")

TARGET.write_text(text)
print("Camera patch installed successfully in cnn_inference.py")
print("Run it the same way as before:")
print("  python cnn_inference.py")
print("Then open:")
print("  http://localhost:8080/")
