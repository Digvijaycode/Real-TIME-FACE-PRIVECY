import os
import argparse
import cv2
import mediapipe as mp
from tqdm import tqdm
import time
import tkinter as tk
from tkinter import filedialog, messagebox

def process_img(img, face_detection):
    """
    Process the image to detect and blur faces using MediaPipe.
    """
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # Increase the blur kernel size for stronger blurring effect
            img[y1:y1 + h, x1:x1 + w] = cv2.blur(img[y1:y1 + h, x1:x1 + w], (75, 75))

    return img

def main(mode, file_path, webcam_index):
    """
    Main function to process image, video or webcam stream based on the selected mode.
    """
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize face detection
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detection:
        if mode == "image":
            if not file_path:
                print("Please provide a valid file path for the image.")
                return

            print(f"Processing image: {file_path}")
            img = cv2.imread(file_path)
            if img is None:
                print(f"Unable to read the image file: {file_path}")
                return

            img = process_img(img, face_detection)
            output_path = os.path.join(output_dir, 'output.png')
            cv2.imwrite(output_path, img)
            print(f"Processed image saved at: {output_path}")

        elif mode == "video":
            if not file_path:
                print("Please provide a valid file path for the video.")
                return

            print(f"Processing video: {file_path}")
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                print(f"Unable to read the video file: {file_path}")
                return

            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read the video file.")
                return

            output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'),
                                           cv2.VideoWriter_fourcc(*'MP4V'),
                                           25,
                                           (frame.shape[1], frame.shape[0]))

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
                while ret:
                    frame = process_img(frame, face_detection)
                    output_video.write(frame)
                    ret, frame = cap.read()
                    pbar.update(1)

            cap.release()
            output_video.release()
            print(f"Processed video saved at: {os.path.join(output_dir, 'output.mp4')}")

        elif mode == "webcam":
            print("Starting webcam. Press 'ESC' to exit.")
            cap = cv2.VideoCapture(webcam_index)
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

            # Countdown before starting
            for i in range(3, 0, -1):
                print(f"Starting in {i}...")
                time.sleep(1)

            ret, frame = cap.read()
            while ret:
                frame = process_img(frame, face_detection)
                cv2.imshow('frame', frame)

                if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key
                    print("Exiting webcam...")
                    break

                # Check if the window is still open
                if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
                    break

                ret, frame = cap.read()

            cap.release()
            cv2.destroyAllWindows()

def get_available_webcams():
    """
    List all available webcams by trying to open them.
    """
    index = 0
    available_webcams = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            available_webcams.append(index)
        cap.release()
        index += 1
    return available_webcams

def start_application():
    """
    Start the application with a simple Tkinter GUI for user-friendly interaction.
    """
    def select_file():
        filetypes = (('All files', '*.*'),)
        file_path = filedialog.askopenfilename(title='Open a file', filetypes=filetypes)
        if file_path:
            file_path_entry.delete(0, tk.END)
            file_path_entry.insert(0, file_path)

    def on_start():
        mode = mode_var.get()
        file_path = file_path_entry.get()
        webcam_index = int(webcam_var.get())
        if mode != "webcam" and not file_path:
            messagebox.showerror("Error", "Please select a file.")
            return
        root.destroy()
        main(mode, file_path, webcam_index)

    root = tk.Tk()
    root.title("Face Blur Application")

    tk.Label(root, text="Select Mode:").pack(pady=10)
    mode_var = tk.StringVar(value="webcam")
    tk.Radiobutton(root, text="Webcam", variable=mode_var, value="webcam").pack()
    tk.Radiobutton(root, text="Image", variable=mode_var, value="image").pack()
    tk.Radiobutton(root, text="Video", variable=mode_var, value="video").pack()

    tk.Label(root, text="File Path:").pack(pady=10)
    file_path_entry = tk.Entry(root, width=50)
    file_path_entry.pack()
    tk.Button(root, text="Browse", command=select_file).pack()

    tk.Label(root, text="Select Webcam:").pack(pady=10)
    webcam_var = tk.StringVar()
    webcam_options = get_available_webcams()
    if not webcam_options:
        messagebox.showerror("Error", "No webcams found.")
        root.destroy()
        return
    webcam_dropdown = tk.OptionMenu(root, webcam_var, *webcam_options)
    webcam_var.set(webcam_options[0])  # Set default to the first webcam
    webcam_dropdown.pack()

    tk.Button(root, text="Start", command=on_start).pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    start_application()