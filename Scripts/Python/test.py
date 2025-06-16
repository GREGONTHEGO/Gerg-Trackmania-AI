import dxcam, cv2, numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

camera = dxcam.create(output_idx=0, output_color='GRAY')
region = (640, 300, 1920, 1200)  # (left, top, width, height)

frame = camera.grab(region=region)
img = frame[:, :, :1]            # already (100,200,3)
img = cv2.resize(img, (200, 100))
img = img.astype(np.float32) / 255.0
plt.figure(figsize=(6,3))
plt.imshow(img)
plt.axis('off')
plt.title("Cropped Preview via DXCam")
plt.show()