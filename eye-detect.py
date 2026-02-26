import cv2
import time
import math
import sys
import mediapipe as mp
import winsound
import threading
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

LEFT = [33, 160, 158, 133, 153, 144]
RIGHT = [362, 385, 387, 263, 373, 380]

def ear(points, ids):
    a = math.dist(points[ids[1]], points[ids[5]])
    b = math.dist(points[ids[2]], points[ids[4]])
    c = math.dist(points[ids[0]], points[ids[3]])
    if c == 0:
        return 1.0
    return (a + b) / (2.0 * c)

def eye_box(points, ids, pad=8):
    xs = [points[i][0] for i in ids]
    ys = [points[i][1] for i in ids]
    x1 = max(min(xs) - pad, 0)
    y1 = max(min(ys) - pad, 0)
    x2 = max(xs) + pad
    y2 = max(ys) + pad
    return x1, y1, x2, y2

# -----------------------------------------------------------
#  SAME CAMERA OPEN LOGIC FROM AIR OBJECT PROJECT
# -----------------------------------------------------------
def open_camera():
    attempts = [
        ("CAP_DSHOW index=0", 0, cv2.CAP_DSHOW),
        ("CAP_MSMF index=0", 0, cv2.CAP_MSMF),
        ("DEFAULT index=0", 0, None),
        ("CAP_DSHOW index=1", 1, cv2.CAP_DSHOW),
        ("CAP_MSMF index=1", 1, cv2.CAP_MSMF),
        ("DEFAULT index=1", 1, None),
    ]

    for label, index, backend in attempts:
        cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            continue

        ok = False
        for _ in range(6):
            ret, _f = cap.read()
            if ret:
                ok = True
                break
            cv2.waitKey(20)

        if ok:
            print(f"📷 Camera opened using: {label}")
            return cap

        cap.release()

    return None
# -----------------------------------------------------------


def main():
    print("🚀 Starting Eye Blink Detection (Stable Camera Version)")
    print("Interpreter:", sys.executable)
    cap = None
    detector = None

    try:
        # Load model
        model_path = "face_landmarker.task"
        try:
            with open(model_path, "rb"):
                pass
        except OSError:
            print(f"❌ Model file missing: {model_path}")
            return

        base = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1
        )
        detector = vision.FaceLandmarker.create_from_options(options)

        # Open camera using stable logic
        cap = open_camera()
        if cap is None:
            print("❌ Camera failed to open!")
            print("Close Zoom / Teams / Browser")
            return

        print("📷 Camera OK — Starting detection...")

        blink_count = 0
        closed_start = None
        was_closed = False
        long_close_beeped = False
        beep_stop = threading.Event()
        beep_thread = None
        BLINK_THRESHOLD = 0.17
        BLINK_MIN_TIME = 0.18
        LONG_CLOSE_TIME = 1.0

        def beep_loop():
            while not beep_stop.is_set():
                try:
                    winsound.Beep(1000, 200)
                    print("beep sound on")
                except Exception:
                    break
                if beep_stop.wait(0.1):
                    print("beep sound off")
                    break

        def start_beep():
            nonlocal beep_thread
            if beep_thread is None or not beep_thread.is_alive():
                beep_stop.clear()
                beep_thread = threading.Thread(target=beep_loop, daemon=True)
                beep_thread.start()

        def stop_beep():
            nonlocal beep_thread
            if beep_thread is not None and beep_thread.is_alive():
                beep_stop.set()
                beep_thread.join(timeout=1.0)
            beep_thread = None
        timestamp = 0

        while True:
            try:
                ok, frame = cap.read()
                if not ok:
                    print("⚠ Frame read failed — retrying")
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp += 33
                result = detector.detect_for_video(mp_img, timestamp)

                if result.face_landmarks:
                    h, w, _ = frame.shape
                    lm = result.face_landmarks[0]
                    points = [(int(p.x * w), int(p.y * h)) for p in lm]

                    L = ear(points, LEFT)
                    R = ear(points, RIGHT)
                    avg = (L + R) / 2

                    cv2.putText(frame, f"EAR: {avg:.2f}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    now = time.monotonic()
                    if avg < BLINK_THRESHOLD:
                        if not was_closed:
                            closed_start = now
                            was_closed = True
                        if closed_start is not None and not long_close_beeped:
                            if now - closed_start >= LONG_CLOSE_TIME:
                                start_beep()
                                long_close_beeped = True
                    else:
                        if was_closed and closed_start is not None:
                            duration = now - closed_start
                            if duration >= BLINK_MIN_TIME:
                                blink_count += 1
                                print("blink ",blink_count)
                        closed_start = None
                        was_closed = False
                        long_close_beeped = False
                        stop_beep()

                    eye_color = (0, 0, 255) if avg < BLINK_THRESHOLD else (0, 165, 255)
                    lx1, ly1, lx2, ly2 = eye_box(points, LEFT)
                    rx1, ry1, rx2, ry2 = eye_box(points, RIGHT)
                    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), eye_color, 2)
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), eye_color, 2)

                    cv2.putText(frame, f"Blinks: {blink_count}", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Face not detected", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Eye Blink Detector", frame)
                if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
                    break
            except cv2.error as e:
                print("OpenCV runtime error:", e)
                break
            except Exception as e:
                print("Loop error:", repr(e))
                continue
    except Exception as e:
        print("Fatal error:", repr(e))
    finally:
        if detector is not None:
            try:
                detector.close()
            except Exception as e:
                print("Detector close error:", repr(e))

        if cap is not None:
            cap.release()

        try:
            stop_beep()
        except Exception:
            pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
