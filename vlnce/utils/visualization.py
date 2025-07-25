# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import List

import cv2
import numpy as np

def write_video(video_frames, filename, fps=10):
    '''
    video_frames: list of frames (T, C, H, W)
    '''

    import imageio
    for i in range(len(video_frames)):
        video_frames[i] = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
    imageio.mimwrite(filename, video_frames, fps=fps)


def add_text_to_image(image: np.ndarray, text: str, top: bool = False) -> np.ndarray:
    """
    Adds text to the given image.

    Args:
        image (np.ndarray): Input image.
        text (str): Text to be added.
        top (bool, optional): Whether to add the text to the top or bottom of the image.

    Returns:
        np.ndarray: Image with text added.
    """
    width = image.shape[1]
    text_image = generate_text_image(width, text)
    if top:
        combined_image = np.vstack([text_image, image])
    else:
        combined_image = np.vstack([image, text_image])

    return combined_image


def generate_text_image(width: int, text: str) -> np.ndarray:
    """
    Generates an image of the given text with line breaks, honoring given width.

    Args:
        width (int): Width of the image.
        text (str): Text to be drawn.

    Returns:
        np.ndarray: Text drawn on white image with the given width.
    """
    # Define the parameters for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    line_spacing = 10  # Spacing between lines in pixels

    # Calculate the maximum width and height of the text
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    max_width = width - 20  # Allow some padding
    max_height = text_size[1] + line_spacing

    # Split the text into words
    words = text.split()

    # Initialize variables for text positioning
    x = 10
    y = text_size[1]

    to_draw = []

    # Iterate over the words and add them to the image
    num_rows = 1
    for word in words:
        # Get the size of the word
        word_size, _ = cv2.getTextSize(word, font, font_scale, font_thickness)

        # Check if adding the word exceeds the maximum width
        if x + word_size[0] > max_width:
            # Add a line break before the word
            y += max_height
            x = 10
            num_rows += 1

        # Draw the word on the image
        to_draw.append((word, x, y))

        # Update the position for the next word
        x += word_size[0] + 5  # Add some spacing between words

    # Create a blank white image with the calculated dimensions
    image = 255 * np.ones((max_height * num_rows, width, 3), dtype=np.uint8)
    for word, x, y in to_draw:
        cv2.putText(
            image,
            word,
            (x, y),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA,
        )

    return image


def pad_images(images: List[np.ndarray], pad_from_top: bool = False) -> List[np.ndarray]:
    """
    Pads a list of images with white pixels to make them have the same dimensions.

    Args:
        images (List[np.ndarray]): List of NumPy images.
        pad_from_top (bool): If True, pad the images from the top; if False (default),
            pad from the bottom.

    Returns:
        List[np.ndarray]: List of padded images.

    """
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)

    padded_images = []
    for img in images:
        height_diff = max_height - img.shape[0]
        width_diff = max_width - img.shape[1]

        if pad_from_top:
            pad_top = height_diff
            pad_bottom = 0
        else:
            pad_top = 0
            pad_bottom = height_diff

        padded_img = np.pad(
            img,
            ((pad_top, pad_bottom), (0, width_diff), (0, 0)),
            mode="constant",
            constant_values=255,
        )
        padded_images.append(padded_img)

    return padded_images


if __name__ == "__main__":
    width = 400
    text = (
        "This is a long text that needs to be drawn on an image with a specified "
        "width. The text should wrap around if it exceeds the given width."
    )

    result_image = generate_text_image(width, text)

    cv2.imshow("Text Image", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
