#!/usr/bin/env python

import argparse
import pathlib
from pathlib import Path
import os
import json
import numpy as np

import rclpy
from rclpy.node import Node
import rospkg
from sensor_msgs.msg import Image
from std_msgs.msg import String
from car_ros_msgs.msg import PLN, Point
from cv_bridge import CvBridge, CvBridgeError
from ament_index_python.packages import get_package_share_directory
from rclpy.clock import Clock, ClockType

import cv2
import torch
import onnxruntime as ort
from PIL import Image as PILImage
from lib.core.general import non_max_suppression


class LaneDetector(Node):

    def __init__(self, subscriber, onnx_model_path, camera):
        super().__init__("lane_detector")
        
        # Publishers
        self.lane_msg_pub = self.create_publisher(PLN, f"{camera}_lane_msg", 1)
        self.da_seg_pub = self.create_publisher(Image, f"{camera}_drivable_area", 1)
        self.ll_seg_pub = self.create_publisher(Image, f"{camera}_lane_lines", 1)
        self.detection_pub = self.create_publisher(Image, f"{camera}_detection", 1)
        self.merged_pub = self.create_publisher(Image, f"{camera}_merged_result", 1)
        
        # Subscriber
        self.sub = self.create_subscription(
            Image, subscriber, self.callback, 1
        )
        
        # Load ONNX model
        ort.set_default_logger_severity(4)
        self.ort_session = ort.InferenceSession(onnx_model_path)
        self.get_logger().info(f"Loaded ONNX model: {onnx_model_path}")
        
        # Bridge for converting between ROS and OpenCV images
        self.bridge = CvBridge()
        
        self.count = 0

    def resize_image_to_640(self, cv_image):
        """
        Resizes image to 640x640 pixels while maintaining aspect ratio and adding padding if necessary.
        This matches the resize_image_to_640 function from test_onnx.py
        
        :param cv_image: OpenCV image
        :return: Resized image (640x640), original image in RGB format, and resize information
        """
        # Convert to PIL Image for consistent resizing behavior with test_onnx.py
        img_bgr = cv_image.copy()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(img_rgb)
        
        # Get the original size
        original_width, original_height = pil_img.size
        
        # Determine the aspect ratio and resize while maintaining aspect ratio
        ratio = min(640 / original_width, 640 / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        
        # Resize the image
        resized_image = pil_img.resize((new_width, new_height), PILImage.LANCZOS)
        
        # Create a new 640x640 background with white padding
        new_image = PILImage.new("RGB", (640, 640), (255, 255, 255))
        
        # Paste the resized image onto the center of the new 640x640 background
        paste_position = ((640 - new_width) // 2, (640 - new_height) // 2)
        new_image.paste(resized_image, paste_position)
        
        # Convert back to numpy array for further processing
        padded_img = np.array(new_image)
        
        # Return the processed image and metadata for later use
        return padded_img, img_rgb, paste_position[0], paste_position[1], new_width, new_height, ratio

    def preprocess_for_inference(self, img_array):
        """
        Prepare the resized image for ONNX model inference by normalizing
        
        :param img_array: Normalized numpy array (640x640x3)
        :return: Preprocessed image ready for model input
        """
        # Normalize the image
        img = img_array.astype(np.float32) / 255.0
        img[:, :, 0] -= 0.485
        img[:, :, 1] -= 0.456
        img[:, :, 2] -= 0.406
        img[:, :, 0] /= 0.229
        img[:, :, 1] /= 0.224
        img[:, :, 2] /= 0.225
        
        # Channel first and add batch dimension
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        
        return img

    def callback(self, data):
        try:
            # Convert ROS Image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
            header = data.header
            
            # Resize image to 640x640 with proper padding (matching test_onnx.py approach)
            padded_img, orig_rgb, offset_x, offset_y, new_w, new_h, ratio = self.resize_image_to_640(cv_image)
            
            # Preprocess image for model inference
            model_input = self.preprocess_for_inference(padded_img)
            
            # Run inference
            det_out, da_seg_out, ll_seg_out = self.ort_session.run(
                ['det_out', 'drive_area_seg', 'lane_line_seg'],
                input_feed={"images": model_input}
            )
            
            # Process detection results
            det_out = torch.from_numpy(det_out).float()
            boxes = non_max_suppression(det_out)[0]  # [n,6] -> [x1, y1, x2, y2, conf, cls]
            boxes = boxes.cpu().numpy().astype(np.float32)
            
            has_boxes = boxes.shape[0] > 0
            
            # Process segmentation results
            da_seg_mask = np.argmax(da_seg_out, axis=1)[0]  # Drive area segmentation
            ll_seg_mask = np.argmax(ll_seg_out, axis=1)[0]  # Lane line segmentation
            
            # Create segmented images
            da_seg_mask = (da_seg_mask * 255).astype(np.uint8)
            ll_seg_mask = (ll_seg_mask * 255).astype(np.uint8)
            
            # Create output images for visualization
            img_det = padded_img.copy()
            img_merge = padded_img.copy()
            
            # Create colored segmentation overlay
            color_area = np.zeros((640, 640, 3), dtype=np.uint8)
            color_area[da_seg_mask == 255] = [0, 255, 0]  # Green for drivable area
            color_area[ll_seg_mask == 255] = [255, 0, 0]  # Blue for lane lines
            
            # Merge segmentation with original image
            mask = np.mean(color_area, 2) > 0
            img_merge[mask] = img_merge[mask] * 0.5 + color_area[mask] * 0.5
            img_merge = img_merge.astype(np.uint8)
            
            # Draw bounding boxes if detected
            if has_boxes:
                for i in range(boxes.shape[0]):
                    x1, y1, x2, y2, conf, label = boxes[i]
                    x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
                    img_det = cv2.rectangle(img_det, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
                    img_merge = cv2.rectangle(img_merge, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
                    
                    # Add label and confidence score
                    label_text = f"Class: {label}, {conf:.2f}"
                    cv2.putText(img_det, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(img_merge, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Extract lane points from lane line segmentation
            lane_points = self.extract_lane_points(ll_seg_mask)
            
            # Create and publish PLN message
            result_msg = PLN(header=header)
            for point in lane_points:
                # Convert coordinates back to original image size
                x = (point[0] - offset_x) / ratio if ratio > 0 else 0
                y = (point[1] - offset_y) / ratio if ratio > 0 else 0
                point_msg = Point()
                point_msg.x = float(x)
                point_msg.y = float(y)
                result_msg.points.append(point_msg)
            
            self.lane_msg_pub.publish(result_msg)
            
            # Convert OpenCV images to ROS messages and publish
            da_seg_msg = self.bridge.cv2_to_imgmsg(da_seg_mask, encoding="mono8")
            da_seg_msg.header = header
            self.da_seg_pub.publish(da_seg_msg)
            
            ll_seg_msg = self.bridge.cv2_to_imgmsg(ll_seg_mask, encoding="mono8")
            ll_seg_msg.header = header
            self.ll_seg_pub.publish(ll_seg_msg)
            
            det_msg = self.bridge.cv2_to_imgmsg(cv2.cvtColor(img_det, cv2.COLOR_RGB2BGR), encoding="bgr8")
            det_msg.header = header
            self.detection_pub.publish(det_msg)
            
            merged_msg = self.bridge.cv2_to_imgmsg(cv2.cvtColor(img_merge, cv2.COLOR_RGB2BGR), encoding="bgr8")
            merged_msg.header = header
            self.merged_pub.publish(merged_msg)
            
            self.get_logger().info(f"Processed frame {self.count}, detected {len(lane_points)} lane points")
            self.count += 1
            
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")
        except Exception as e:
            self.get_logger().error(f"Error in callback: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def extract_lane_points(self, lane_mask, sample_points=20):
        """Extract lane points from the lane line segmentation mask"""
        lane_points = []
        
        # Find contours in the lane mask
        contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Only process contours with enough points
            if len(contour) > 5:
                # Reshape to get usable coordinates
                contour_reshaped = contour.reshape(-1, 2)
                
                # Sample points along the contour
                if len(contour_reshaped) <= sample_points:
                    sampled_points = contour_reshaped
                else:
                    indices = np.linspace(0, len(contour_reshaped) - 1, sample_points, dtype=int)
                    sampled_points = contour_reshaped[indices]
                
                for point in sampled_points:
                    lane_points.append(point)
        
        return lane_points


def main(arg=None):
    # Set up arguments
    rclpy.init(args=arg)

    parser = argparse.ArgumentParser(
        description="Lane detection using YOLOP ONNX model with ROS",
    )

    parser.add_argument(
        "-s", "--Subscriber", 
        help="Image Topic to subscribe to", 
        type=str, 
        required=True
    )

    parser.add_argument(
        "-c", "--Camera",
        help="Name of camera",
        required=True
    )
    
    parser.add_argument(
        "-m", "--Model", 
        help="Path to ONNX model file",
        default="weights/yolop-640-640.onnx",
        type=str
    )

    args, _ = parser.parse_known_args()

    # Create and run the node
    node = LaneDetector(args.Subscriber, args.Model, args.Camera)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
