import os
import cv2
import torch
import argparse
import onnxruntime as ort
import numpy as np
from lib.core.general import non_max_suppression
from PIL import Image

import os
import cv2
import torch
import argparse
import onnxruntime as ort
import numpy as np
from lib.core.general import non_max_suppression
from PIL import Image


def resize_image_to_640(image_path, output_path=None):
    """
    Resizes any image to 640x640 pixels while maintaining aspect ratio and adding padding if necessary.
    
    :param image_path: Path to the original image.
    :param output_path: Path to save the resized image. If None, saves with '_resized' suffix.
    :return: Path to the resized image.
    """
    # Open the image
    with Image.open(image_path) as img:
        # Get the original size
        original_width, original_height = img.size
        
        # Determine the aspect ratio and resize while maintaining aspect ratio
        ratio = min(640 / original_width, 640 / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        
        # Resize the image
        resized_image = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Create a new 640x640 background with white padding
        new_image = Image.new("RGB", (640, 640), (255, 255, 255))
        
        # Paste the resized image onto the center of the new 640x640 background
        paste_position = ((640 - new_width) // 2, (640 - new_height) // 2)
        new_image.paste(resized_image, paste_position)
        
        # Define the output path if not provided
        if not output_path:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_resized{ext}"
        
        # Save the resized image
        new_image.save(output_path)
        print(f"✅ Resized image saved at: {output_path}")
        return output_path


def infer_yolop(weight="yolop-640-640.onnx",
                img_path="inference/images/EditedCar/download-resizehood.com (2).jpg"):

    ort.set_default_logger_severity(4)
    onnx_path = f"./weights/{weight}"
    ort_session = ort.InferenceSession(onnx_path)
    print(f"✅ Loaded model: {onnx_path}")

    # Load resized image
    img_bgr = cv2.imread(img_path)
    height, width, _ = img_bgr.shape

    # Convert to RGB for normalization
    img_rgb = img_bgr[:, :, ::-1].copy()

    # Normalize the image (no need to resize again, already done)
    img = img_rgb.astype(np.float32) / 255.0
    img[:, :, 0] -= 0.485
    img[:, :, 1] -= 0.456
    img[:, :, 2] -= 0.406
    img[:, :, 0] /= 0.229
    img[:, :, 1] /= 0.224
    img[:, :, 2] /= 0.225

    img = img.transpose(2, 0, 1)  # Channels first
    img = np.expand_dims(img, 0)  # Add batch dimension

    # Inference: (1, n, 6) (1, 2, 640, 640) (1, 2, 640, 640)
    det_out, da_seg_out, ll_seg_out = ort_session.run(
        ['det_out', 'drive_area_seg', 'lane_line_seg'],
        input_feed={"images": img}
    )

    # Process detection results
    det_out = torch.from_numpy(det_out).float()
    boxes = non_max_suppression(det_out)[0]  # [n,6] -> [x1, y1, x2, y2, conf, cls]
    boxes = boxes.cpu().numpy().astype(np.float32)

    save_det_path = f"./pictures/detect_onnx.jpg"
    save_da_path = f"./pictures/da_onnx.jpg"
    save_ll_path = f"./pictures/ll_onnx.jpg"
    save_merge_path = f"./pictures/output_onnx.jpg"

    has_boxes = boxes.shape[0] > 0

    # Draw bounding boxes if detected
    if has_boxes:
        img_det = img_rgb[:, :, ::-1].copy()
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2, conf, label = boxes[i]
            x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
            img_det = cv2.rectangle(img_det, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
        cv2.imwrite(save_det_path, img_det)
    else:
        cv2.imwrite(save_det_path, img_bgr)
        print("⚠️ No bounding boxes detected.")

    # Process segmentation results
    da_seg_mask = np.argmax(da_seg_out, axis=1)[0]  # Drive area segmentation
    ll_seg_mask = np.argmax(ll_seg_out, axis=1)[0]  # Lane line segmentation

    # Create segmented images
    da_seg_mask = (da_seg_mask * 255).astype(np.uint8)
    ll_seg_mask = (ll_seg_mask * 255).astype(np.uint8)

    # Save segmentation results
    cv2.imwrite(save_da_path, da_seg_mask)
    cv2.imwrite(save_ll_path, ll_seg_mask)

    # Merge segmentation and original image
    color_area = np.zeros((640, 640, 3), dtype=np.uint8)
    color_area[da_seg_mask == 255] = [0, 255, 0]  # Green for drivable area
    color_area[ll_seg_mask == 255] = [255, 0, 0]  # Blue for lane lines
    color_seg = color_area[..., ::-1]  # Convert to BGR

    img_merge = img_rgb[:, :, ::-1].copy()
    mask = np.mean(color_seg, 2) > 0
    img_merge[mask] = img_merge[mask] * 0.5 + color_seg[mask] * 0.5
    img_merge = img_merge.astype(np.uint8)

    # Draw boxes on the merged image if detected
    if has_boxes:
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2, conf, label = boxes[i]
            x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
            img_merge = cv2.rectangle(img_merge, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)

    cv2.imwrite(save_merge_path, img_merge)

    print("✅ Inference and saving results done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default="yolop-640-640.onnx")
    parser.add_argument('--img', type=str, default="inference/images/EditedCar/download-resizehood.com (2).jpg")
    args = parser.parse_args()

    # Resize image to 640x640 before inference
    resized_img_path = resize_image_to_640(args.img)

    # Run inference with resized image
    infer_yolop(weight=args.weight, img_path=resized_img_path)

    import os
import cv2
import torch
import argparse
import onnxruntime as ort
import numpy as np
from lib.core.general import non_max_suppression
from PIL import Image