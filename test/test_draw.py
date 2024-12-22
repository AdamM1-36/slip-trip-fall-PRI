import cv2
import numpy as np

PALETTE = np.array(
                [[255, 128, 0], [255, 153, 51], [255, 178, 102],
                [230, 230, 0], [255, 153, 255], [153, 204, 255],
                [255, 102, 255], [255, 51, 255], [102, 178, 255],
                [51, 153, 255], [255, 153, 153], [255, 102, 102],
                [255, 51, 51], [153, 255, 153], [102, 255, 102],
                [51, 255, 51], [0, 255, 0], [0, 0, 255], 
                [255, 0, 0], [255, 255, 255]]
            )
SKELETON = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
        ]
POSE_LIMB_COLOR = PALETTE[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

def draw_keypoints_and_skeleton(frame, keypoints):
    # Draw keypoints
    for i, keypoint in enumerate(keypoints):
        r, g, b = POSE_LIMB_COLOR[i] if i < len(POSE_LIMB_COLOR) else (255, 255, 255)
        x, y, conf = keypoint
        if conf > 0.5:
            cv2.putText(frame, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (int(x), int(y)), 5, (int(r), int(g), int(b)), -1)

    # Draw skeleton
    for i, sk in enumerate(SKELETON):
        r, g, b = POSE_LIMB_COLOR[i] if i < len(POSE_LIMB_COLOR) else (255, 255, 255)
        pt1 = keypoints[sk[0] - 1]
        pt1x, pt1y, pt1_conf = [int(pt1[0]), int(pt1[1]), pt1[2]]
        pt2 = keypoints[sk[1] - 1]
        pt2x, pt2y, pt2_conf = [int(pt2[0]), int(pt2[1]), pt2[2]]
        if pt1_conf > 0.5 and pt2_conf > 0.5:  # Only draw lines if both keypoints have high confidence
            cv2.line(frame, (pt1x, pt1y), (pt2x, pt2y), (int(r), int(g), int(b)), 2)