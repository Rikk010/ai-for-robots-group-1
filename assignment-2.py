import argparse
import cv2
import time
import numpy as np

from transformers import pipeline
from PIL import Image

def main():
    parser = argparse.ArgumentParser(
        description="Live depth estimation from camera input"
    )
    parser.add_argument(
        "-c", "--camera",
        type=int,
        default=0,
        help="Camera index to use (default: 0)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help=(
            "Hugging Face depth-estimation model ID. Available models:\n"
            "- depth-anything/Depth-Anything-V2-Small-hf    (Params: 24.8M)\n"
            "- depth-anything/Depth-Anything-V2-Medium-hf   (Params: 97.5M)\n"
            "- depth-anything/Depth-Anything-V2-Large-hf    (Params: 335.3M)\n"
            "- depth-anything/Depth-Anything-V2-Giant-hf    (Params: 1.3B)"
        )
    )
    args = parser.parse_args()

    # Initialize depth-estimation pipeline
    pipe = pipeline(task="depth-estimation", model=args.model)

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    prev_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # BGR -> RGB -> PIL Image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # Run depth estimation
            depth = pipe(pil_img)["depth"]
            depth_np = np.array(depth, dtype=np.float32)

            # Normalize & convert to 8-bit
            depth_norm = cv2.normalize(
                depth_np, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)

            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time

            # Convert to BGR and overlay FPS in bottom-right
            depth_bgr = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)
            text = f"FPS: {fps:.1f}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            pos = (depth_bgr.shape[1] - text_w - 10, depth_bgr.shape[0] - 10)
            cv2.putText(depth_bgr, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
            cv2.imshow("Depth", depth_bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()