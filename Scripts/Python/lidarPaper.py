import cv2, dxcam
import numpy as np
import math
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# dy = np.array([-351, -297, -181]) # -380, -374, -367,
# distances = np.array([48, 10, 5]) # 1216, 192, 96, 
# dy_pos = np.abs(dy)

# log_dy = np.log(dy_pos)
# log_distances = np.log(distances)

# # interp_fn = interp1d(log_dy, distances, kind='linear', fill_value="extrapolate")



# b, log_a = np.polyfit(log_dy, distances, 1)
# a = np.exp(log_a)
# def power_model(x, a, b):
#     return a * np.power(x, b)

# params, _ = curve_fit(power_model, dy_pos, distances, p0=[0.01, 2], maxfev=10000)
# a, b = params
# print(f"Fitted parameters: a = {a:.4f}* y ^ b = {b:.4f}")

def pixel_to_distance(y, angle, image_height=1600, vertical_fov=130, camera_tilt_degree=33, camera_height=1.5):
    pixel_from_bottom = abs(y)
    if pixel_from_bottom < 1:
        pixel_from_bottom = 0
    vertical_ratio = pixel_from_bottom / image_height
    beta_degree = vertical_fov * vertical_ratio
    total_angle = min(camera_tilt_degree + beta_degree, 89.7)
    total_angle_rad = math.radians(total_angle)
    print(f"theta: {angle}, Pixel from bottom: {pixel_from_bottom}, Vertical ratio: {vertical_ratio:.4f}, Beta degree: {beta_degree:.2f}, Total angle: {total_angle:.2f}°")
    distance = math.tan(total_angle_rad) * camera_height
    return distance

def lateral_offset(x, forward_distance, image_width=2560, horizontal_fov=70):
    if x == 0:
        x = 1
    horizontal_ratio = x / image_width
    alpha_degree = horizontal_fov * horizontal_ratio
    alpha_rad = math.radians(alpha_degree)
    lateral_offset = forward_distance * math.tan(alpha_rad)
    # if x < 1260:
    #     return 5.0
    print(f"dx: {x}, Forward distance: {forward_distance:.2f} m, Horizontal ratio: {horizontal_ratio:.4f}, Alpha degree: {alpha_degree:.2f}°, Lateral offset: {lateral_offset:.2f} m")
    return lateral_offset

# def distance_from_dy(dy):
#     dy_abs = np.abs(dy)
#     dy_val = np.clip(np.abs(dy), 1, None)
#     b_val = b
#     return a * np.power(dy_val, b_val)
#     # dy = np.clip(dy, dy_pos.min(), dy_pos.max())
#     # return float(interp_fn(np.abs(np.log(dy))))
# def estimate_total_distance(origin, hit):
#     dx = hit[0] - origin[0]
#     dy = abs(hit[1] - origin[1])
#     print(f"dx: {dx}, dy: {dy}")
#     vertical_dist = distance_from_dy(dy)
#     raw_pixel_length = math.hypot(dx, dy)
#     print(vertical_dist, raw_pixel_length)
#     if abs(dy) < 10:
#         return 5.0
#     scale = raw_pixel_length / abs(dy)
#     return vertical_dist * scale
# dist = distance_from_dy(-380)
# print(f"Distance from dy = -380: {dist:.2f} m")
# dist = distance_from_dy(-100)
# print(f"Distance from dy = -100: {dist:.2f} m")
def black_and_white_filter(image):
    if len(image.shape) == 2 or image.shape[2] == 1:
        mask = image < 50
        result = np.full_like(image, 255)
        result[mask] = 0
        return result
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black = gray < 50
    white = gray > 240
    result = np.full_like(image, 255)
    black_mask = np.repeat(black[:, :, np.newaxis], 3, axis=2)
    white_mask = np.repeat(white[:, :, np.newaxis], 3, axis=2)
    result[black_mask] = 0
    result[white_mask] = 255
    return result

def simulate_lidar_overlay(image, num_rays = 19, max_distance = 1270):
    masked_gray = black_and_white_filter(image)
    edges = cv2.Canny(masked_gray, 50, 150)
    H, W = edges.shape
    # print(W, H)
    overlay = masked_gray
    origin = (W // 2, H - 100)
    angles = np.linspace(-math.radians(90), math.radians(90), num_rays)
    # print(f"Origin: {origin}, Image Size: {W}x{H}, Number of Rays: {num_rays}, Max Distance: {max_distance}")
    for theta in angles:
        for d in range(1, max_distance):
            dx = (d * math.sin(theta))
            dy = (d * math.cos(theta))
            x = int(origin[0] + dx)
            y = int(origin[1] - dy)
            x_part = int(origin[0] + dx*2/3)
            y_part = int(origin[1] - dy*2/3)
            if 0 <= x < W and 0 <= y < H:
                if edges[y, x] > 0:
                    cv2.line(overlay, origin, (x, y), (0, 255, 0), 3)
                    # print(f"Ray at angle {math.degrees(theta):.2f}° hit at distance {d} pixels, coordinates ({x - origin[0]}, {y - origin[1]})")
                    forward_distance = pixel_to_distance(dy, theta, H)
                    lateral_distance = lateral_offset(dx, forward_distance, W)
                    total = math.hypot(forward_distance, lateral_distance)
                    # total = estimate_total_distance(origin, (x, y))
                    # print(f"dx: {dx}, dy: {dy}")
                    # print(f"Ray at angle {math.degrees(theta):.2f} hit at distance {total:.2f} meters")
                    cv2.putText(overlay, f"{total:.2f} m", (x_part, y_part), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    break
            else:
                break
        else:
            dx = (d * math.sin(theta))
            dy = (d * math.cos(theta))
            end_x = int(origin[0] + dx)
            end_y = int(origin[1] - dy)
            x_part = int(origin[0] + dx*2/3)
            y_part = int(origin[1] - dy*2/3)
            if 0 <= end_x < W and 0 <= end_y < H:
                cv2.line(overlay, origin, (end_x, end_y), (0, 0, 255), 3)
                # print(f"Ray at angle {math.degrees(theta):.2f}° hit at distance {d} pixels, coordinates ({end_x}, {end_y})")
                # total = estimate_total_distance(origin, (x, y))
                forward_distance = pixel_to_distance(dy, theta, H)
                lateral_distance = lateral_offset(dx, forward_distance, W)
                total = math.hypot(forward_distance, lateral_distance)
                cv2.putText(overlay, f"{total:.2f} m", (x_part, y_part), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                # print(f"Ray at angle {math.degrees(theta):.2f} hit at distance {total:.2f} meters")

    return overlay

camera = dxcam.create(output_idx=0, output_color="BGR", region=(0, 100, 2560, 1400))
camera.start(target_fps=60)

while True:
    frame = camera.get_latest_frame()
    if frame is None:
        continue

    lidar_overlay = simulate_lidar_overlay(frame)
    cv2.imshow("Lidar Overlay", lidar_overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.stop()
cv2.destroyAllWindows()