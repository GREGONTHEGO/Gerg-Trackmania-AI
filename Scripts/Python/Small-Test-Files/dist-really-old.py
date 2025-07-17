import numpy as np

def camera_distance_from_dy(dy, img_height=1300, camera_height=1.5, vertical_fov_deg=80):
    # Avoid dy = 0 (straight ahead)
    dy = np.clip(dy, 1, img_height)
    beta_deg = (dy / img_height) * vertical_fov_deg
    beta_rad = np.radians(beta_deg)
    return camera_height * np.tan(beta_rad)


for dy in [10, 50, 100, 200, 300, 400, 600]:
    print(f"dy = {dy:3} → distance = {camera_distance_from_dy(dy):.2f} m")