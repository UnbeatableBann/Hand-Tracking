import time

import cv2
import numpy as np

SKIN_HSV_LOW = np.array([0, 30, 60])
SKIN_HSV_HIGH = np.array([20, 170, 255])

MORPH_KERNEL = np.ones((5, 5), np.uint8)
MIN_CONTOUR_AREA = 3000

BOX_FRAC = (0.35, 0.2, 0.65, 0.6)

WARNING_DIST = 120
DANGER_DIST = 60
WARNING_HYST = 15
DANGER_HYST = 10

DIST_SMOOTH_ALPHA = 0.35
STATE_MIN_HOLD_SEC = 0.6

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

STATE_SAFE = "SAFE"
STATE_WARNING = "WARNING"
STATE_DANGER = "DANGER"

COLOR_SAFE = (0, 255, 0)
COLOR_WARNING = (0, 255, 255)
COLOR_DANGER = (0, 0, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_virtual_box(frame):
    h, w = frame.shape[:2]
    x1 = int(BOX_FRAC[0] * w)
    y1 = int(BOX_FRAC[1] * h)
    x2 = int(BOX_FRAC[2] * w)
    y2 = int(BOX_FRAC[3] * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return (x1, y1, x2, y2)


def preprocess_skin(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, SKIN_HSV_LOW, SKIN_HSV_HIGH)
    mask = cv2.erode(mask, MORPH_KERNEL, iterations=1)
    mask = cv2.dilate(mask, MORPH_KERNEL, iterations=2)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask


def largest_contour(binary_mask):
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_CONTOUR_AREA:
        return None
    return c


def contour_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def convex_hull_and_fingertip(contour, centroid):
    hull = cv2.convexHull(contour)
    if hull is None or len(hull) == 0:
        return None, None
    if centroid is None:
        return hull, None
    cx, cy = centroid
    max_dist = -1
    tip = None
    for p in hull:
        x, y = p[0]
        d = (x - cx) ** 2 + (y - cy) ** 2
        if d > max_dist:
            max_dist = d
            tip = (int(x), int(y))
    return hull, tip


def point_to_box_distance(pt, box):
    x, y = pt
    x1, y1, x2, y2 = box
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    if dx == 0 and dy == 0:
        return 0
    return int(np.hypot(dx, dy))


def classify_state(distance, prev_state):
    if prev_state == STATE_DANGER:
        leave_thresh = DANGER_DIST + DANGER_HYST
        if distance <= DANGER_DIST:
            return STATE_DANGER
        elif distance <= max(WARNING_DIST, leave_thresh):
            return STATE_WARNING
        else:
            return STATE_SAFE
    elif prev_state == STATE_WARNING:
        leave_thresh = WARNING_DIST + WARNING_HYST
        if distance <= DANGER_DIST:
            return STATE_DANGER
        elif distance <= WARNING_DIST:
            return STATE_WARNING
        elif distance <= leave_thresh:
            return STATE_WARNING
        else:
            return STATE_SAFE
    else:
        if distance <= DANGER_DIST:
            return STATE_DANGER
        elif distance <= WARNING_DIST:
            return STATE_WARNING
        else:
            return STATE_SAFE


def render_overlay(frame, state, fps, fingertip, centroid, hull, box):
    x1, y1, x2, y2 = box
    color = (
        COLOR_SAFE
        if state == STATE_SAFE
        else (COLOR_WARNING if state == STATE_WARNING else COLOR_DANGER)
    )
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if hull is not None:
        cv2.polylines(frame, [hull], isClosed=True, color=(200, 200, 0), thickness=2)
    if centroid is not None:
        cv2.circle(frame, centroid, 5, (255, 0, 0), -1)
    if fingertip is not None:
        cv2.circle(frame, fingertip, 8, (0, 0, 255), -1)
        cv2.line(
            frame, fingertip, centroid if centroid else fingertip, (0, 255, 255), 2
        )
    cv2.putText(frame, f"State: {state}", (20, 30), FONT, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (20, 60), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA
    )
    if state == STATE_DANGER:
        msg = "DANGER DANGER"
        (tw, th), _ = cv2.getTextSize(msg, FONT, 1.6, 4)
        cx = frame.shape[1] // 2
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (cx - tw // 2 - 20, 60), (cx + tw // 2 + 20, 120), (0, 0, 255), -1
        )
        frame[:] = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
        cv2.putText(
            frame, msg, (cx - tw // 2, 110), FONT, 1.6, COLOR_DANGER, 4, cv2.LINE_AA
        )


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    prev = time.time()
    fps = 0.0
    smoothed_distance = None
    state = STATE_SAFE
    state_since = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        now = time.time()
        dt = now - prev
        prev = now
        if dt > 0:
            fps = 1.0 / dt
        box = draw_virtual_box(frame)
        mask = preprocess_skin(frame)
        cnt = largest_contour(mask)
        fingertip = None
        centroid = None
        hull = None
        distance = 9999
        if cnt is not None:
            centroid = contour_centroid(cnt)
            hull, fingertip = convex_hull_and_fingertip(cnt, centroid)
            if fingertip is None:
                bx = (box[0] + box[2]) // 2
                by = (box[1] + box[3]) // 2
                min_d = 1e9
                min_pt = None
                for p in cnt:
                    x, y = p[0]
                    d = (x - bx) ** 2 + (y - by) ** 2
                    if d < min_d:
                        min_d = d
                        min_pt = (int(x), int(y))
                fingertip = min_pt
            if fingertip is not None:
                distance = point_to_box_distance(fingertip, box)
        if distance < 9999:
            if smoothed_distance is None:
                smoothed_distance = float(distance)
            else:
                smoothed_distance = (
                    DIST_SMOOTH_ALPHA * float(distance)
                    + (1.0 - DIST_SMOOTH_ALPHA) * smoothed_distance
                )
        else:
            if smoothed_distance is not None:
                smoothed_distance = (
                    1.0 - DIST_SMOOTH_ALPHA
                ) * smoothed_distance + DIST_SMOOTH_ALPHA * 9999.0
            else:
                smoothed_distance = 9999.0
        new_state = classify_state(int(smoothed_distance), state)
        now_ts = time.time()
        if new_state != state:
            if (now_ts - state_since) >= STATE_MIN_HOLD_SEC:
                state = new_state
                state_since = now_ts
        render_overlay(frame, state, fps, fingertip, centroid, hull, box)
        cv2.imshow("Hand-Boundary Warning", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
