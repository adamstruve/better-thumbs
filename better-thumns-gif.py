import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageSequence


def create_thumbnail(input_image_path, output_image_path, thumbnail_size=(500, 280)):
    image_pil = Image.open(input_image_path)
    frames = []
    first_frame = True
    thumbnail_left = 0
    thumbnail_top = 0

    for frame in ImageSequence.Iterator(image_pil):
        image = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        height, width, _ = image.shape
        aspect_ratio = float(width) / float(height)

        target_width, target_height = thumbnail_size
        target_aspect_ratio = float(target_width) / float(target_height)

        if target_aspect_ratio > aspect_ratio:
            new_width = target_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = target_height
            new_width = int(new_height * aspect_ratio)

        image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        if first_frame:
            face_locations = face_recognition.face_locations(image_resized)

            if face_locations:
                top, right, bottom, left = face_locations[0]

                face_center_x = (right + left) // 2
                face_center_y = (top + bottom) // 2

                thumbnail_left = min(max(face_center_x - target_width // 2, 0), new_width - target_width)
                thumbnail_top = min(max(face_center_y - target_height // 2, 0), new_height - target_height)
            else:
                thumbnail_left = (new_width - target_width) // 2
                thumbnail_top = (new_height - target_height) // 2

            first_frame = False

        thumbnail = image_resized[thumbnail_top:thumbnail_top + target_height, thumbnail_left:thumbnail_left + target_width]
        thumbnail_pil = Image.fromarray(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB))
        frames.append(thumbnail_pil)

    if len(frames) > 1:
        frames[0].save(output_image_path, format='GIF', save_all=True, append_images=frames[1:], duration=image_pil.info['duration'], loop=0)
    else:
        frames[0].save(output_image_path)


if __name__ == "__main__":
    input_image_path = "1.jpg"
    output_image_path = "1thumbnail.jpg"

    create_thumbnail(input_image_path, output_image_path)
