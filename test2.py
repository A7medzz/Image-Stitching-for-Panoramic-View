import streamlit as st
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

# ============== Simplified Stitching Functions ==============

def detect_features(image, method="ORB"):
    """Detect keypoints and descriptors"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if method == "SIFT":
        detector = cv2.SIFT_create()
    else:  # ORB
        detector = cv2.ORB_create(3000)  # Fixed number for simplicity
    
    kps, features = detector.detectAndCompute(gray, None)
    kps = np.float32([kp.pt for kp in kps])
    return kps, features

def match_features(kps1, kps2, feats1, feats2, method="ORB"):
    """Match features between two images"""
    if method == "SIFT":
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    raw_matches = matcher.knnMatch(feats1, feats2, k=2)
    good_matches = []
    
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:  # Lowe's ratio test
            good_matches.append(m)
    
    return good_matches

def stitch_two_images(img_left, img_right, method="ORB"):
    """Stitch two images together with simple blending"""
    
    # Step 1: Detect features
    kps_left, feats_left = detect_features(img_left, method)
    kps_right, feats_right = detect_features(img_right, method)
    
    # Step 2: Match features
    matches = match_features(kps_left, kps_right, feats_left, feats_right, method)
    
    if len(matches) < 10:
        raise ValueError(f"Not enough matches found ({len(matches)}). Need at least 10.")
    
    # Step 3: Compute homography
    pts_left = np.float32([kps_left[m.queryIdx] for m in matches])
    pts_right = np.float32([kps_right[m.trainIdx] for m in matches])
    
    H, _ = cv2.findHomography(pts_right, pts_left, cv2.RANSAC, 4.0)
    
    if H is None:
        raise ValueError("Could not compute homography")
    
    # Step 4: Warp right image
    h1, w1 = img_left.shape[:2]
    h2, w2 = img_right.shape[:2]
    
    # Calculate output size
    corners_right = np.float32([[0,0], [w2,0], [w2,h2], [0,h2]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners_right, H)
    
    all_corners = np.concatenate((warped_corners, 
                                  np.float32([[0,0], [w1,0], [w1,h1], [0,h1]]).reshape(-1,1,2)))
    
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # Translation to keep positive coordinates
    tx, ty = -xmin, -ymin
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]])
    
    # Warp right image
    output_w = xmax - xmin
    output_h = ymax - ymin
    warped = cv2.warpPerspective(img_right, T @ H, (output_w, output_h))
    
    # Step 5: Simple blending
    result = warped.copy()
    
    # Place left image
    left_x1, left_y1 = tx, ty
    left_x2, left_y2 = tx + w1, ty + h1
    
    # Create mask for left image
    mask = np.zeros((output_h, output_w), dtype=np.uint8)
    mask[left_y1:left_y2, left_x1:left_x2] = 255
    
    # Blend overlap region (simple feather)
    overlap = cv2.bitwise_and(warped[:,:,0] > 0, mask > 0)
    
    if np.any(overlap):
        # Find overlap bounds
        y_idx, x_idx = np.where(overlap)
        if len(x_idx) > 0:
            x_min, x_max = np.min(x_idx), np.max(x_idx)
            
            # Create alpha blend
            for x in range(x_min, x_max + 1):
                if x_max > x_min:
                    alpha = (x - x_min) / (x_max - x_min)  # 0 to 1
                    mask_col = mask[:, x]
                    warp_col = warped[:, x]
                    left_col = img_left[:, x - tx] if 0 <= (x - tx) < w1 else np.zeros_like(warp_col)
                    
                    # Blend
                    result[:, x] = warp_col * (1 - alpha) + left_col * alpha
    
    # Place non-overlapping left image parts
    result[ty:ty+h1, tx:tx+w1] = img_left
    
    return result

def crop_black_borders(image):
    """Remove black borders from image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find non-black pixels
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    # Get bounding box of largest contour
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    # Add small margin
    margin = 5
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(image.shape[1] - x, w + 2 * margin)
    h = min(image.shape[0] - y, h + 2 * margin)
    
    return image[y:y+h, x:x+w]

def stitch_all_images(images, method="ORB"):
    """Stitch multiple images using center reference"""
    if len(images) < 2:
        raise ValueError("Need at least 2 images")
    
    # Use middle image as starting point
    center_idx = len(images) // 2
    panorama = images[center_idx]
    
    # Stitch images to the left of center
    for i in range(center_idx - 1, -1, -1):
        panorama = stitch_two_images(images[i], panorama, method)
    
    # Stitch images to the right of center
    for i in range(center_idx + 1, len(images)):
        panorama = stitch_two_images(panorama, images[i], method)
    
    # Remove black borders
    panorama = crop_black_borders(panorama)
    
    return panorama

# ============== Simple Streamlit GUI ==============

def main():
    st.set_page_config(page_title="Panorama Stitcher", layout="wide")
    
    st.title("🌄 Simple Panorama Stitcher")
    st.write("Upload overlapping images to create a panoramic view.")
    
    # Simple sidebar
    with st.sidebar:
        st.header("Settings")
        method = st.selectbox("Feature Method", ["ORB", "SIFT"])
        st.info("ORB: Fast | SIFT: More accurate")
        
        if st.checkbox("Resize large images", value=True):
            max_width = st.slider("Max width", 400, 1200, 800)
        else:
            max_width = None
    
    # Upload section
    uploaded_files = st.file_uploader(
        "Choose images (2 or more)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files and len(uploaded_files) >= 2:
        st.success(f"Loaded {len(uploaded_files)} images")
        
        # Show thumbnails
        cols = st.columns(min(3, len(uploaded_files)))
        for i, file in enumerate(uploaded_files):
            with cols[i % 3]:
                st.image(Image.open(file), caption=f"Image {i+1}", use_column_width=True)
        
        if st.button("✨ Create Panorama", type="primary"):
            try:
                # Load and resize images
                images = []
                for file in uploaded_files:
                    pil_img = Image.open(file)
                    img_array = np.array(pil_img)
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    if max_width:
                        h, w = img_bgr.shape[:2]
                        if w > max_width:
                            scale = max_width / w
                            new_h = int(h * scale)
                            img_bgr = cv2.resize(img_bgr, (max_width, new_h))
                    
                    images.append(img_bgr)
                
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Stitch with progress updates
                status_text.text("Starting stitching...")
                progress_bar.progress(10)
                
                panorama = images[len(images)//2]  # Start with middle image
                
                # Stitch left side
                center_idx = len(images) // 2
                for i in range(center_idx - 1, -1, -1):
                    progress = 10 + ((center_idx - i) / len(images)) * 40
                    progress_bar.progress(int(progress))
                    status_text.text(f"Stitching image {i+1} from left...")
                    panorama = stitch_two_images(images[i], panorama, method)
                
                # Stitch right side
                for i in range(center_idx + 1, len(images)):
                    progress = 50 + ((i - center_idx) / len(images)) * 40
                    progress_bar.progress(int(progress))
                    status_text.text(f"Stitching image {i+1} from right...")
                    panorama = stitch_two_images(panorama, images[i], method)
                
                progress_bar.progress(95)
                status_text.text("Cropping borders...")
                panorama = crop_black_borders(panorama)
                
                progress_bar.progress(100)
                status_text.text("Done!")
                
                # Display result
                st.subheader("🎯 Panorama Result")
                panorama_rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
                st.image(panorama_rgb, use_column_width=True)
                
                # Stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Width", f"{panorama.shape[1]} px")
                with col2:
                    st.metric("Height", f"{panorama.shape[0]} px")
                with col3:
                    st.metric("Images", len(images))
                
                # Download
                buf = BytesIO()
                Image.fromarray(panorama_rgb).save(buf, format="PNG")
                st.download_button(
                    "⬇️ Download Panorama",
                    data=buf.getvalue(),
                    file_name="panorama.png",
                    mime="image/png"
                )
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("""
                **Tips for better results:**
                - Ensure 30-50% overlap between images
                - Keep images in left-to-right order
                - Use consistent lighting
                - Try SIFT if ORB fails
                """)
    
    elif uploaded_files and len(uploaded_files) == 1:
        st.warning("Please upload at least 2 images for stitching.")
    
    else:
        st.info("👈 Upload 2 or more overlapping images to begin.")
        
        # Simple instructions
        with st.expander("How to take good photos for panoramas"):
            st.markdown("""
            1. **Overlap**: Each photo should overlap with the next by 30-50%
            2. **Consistency**: Keep same camera height and angle
            3. **Rotation**: Rotate around the lens, not your body
            4. **Lighting**: Avoid drastic lighting changes
            5. **Order**: Take photos left-to-right or right-to-left
            """)
    
    # Footer
    st.divider()
    st.caption("Panorama Stitcher | Feature Matching • Homography • Simple Blending")

if __name__ == "__main__":
    main()