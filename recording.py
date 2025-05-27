import cv2

def record_resized_video(output_path='output.avi',
                         cam_index=0,
                         frame_size=(320, 240),
                         fps=30.0,
                         codec='XVID'):
    # Open the camera
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise IOError(f"Cannot open camera {cam_index}")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    print(f"Recording started. Press 'q' to stop.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Resize frame
        resized = cv2.resize(frame, frame_size)

        # Write the resized frame
        out.write(resized)

        # Optional: show on-screen preview
        cv2.imshow('Resized Preview', resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Recording saved to {output_path}")

if __name__ == '__main__':
    record_resized_video(
        output_path='camera_320x240.avi',
        cam_index=0,
        frame_size=(320, 240),
        fps=30.0,
        codec='XVID'
    )