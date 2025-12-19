import streamlit as st
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

# ============== Stitching Functions (from stitcher.py) ==============

def detect_and_describe(image, method="ORB"):
    """Detect and compute keypoints and descriptors"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if method == "SIFT":
        sift = cv2.SIFT_create()
        kps, features = sift.detectAndCompute(gray, None)
    else:  # ORB (free and built-in)
        orb = cv2.ORB_create(5000)
        kps, features = orb.detectAndCompute(gray, None)
    
    kps = np.float32([kp.pt for kp in kps])
    return kps, features


def match_keypoints(kpsA, kpsB, featsA, featsB, ratio=0.75, reproj_thresh=4.0, method="ORB"):
    """Match keypoints between two images"""
    if method == "SIFT":
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    raw_matches = matcher.knnMatch(featsA, featsB, k=2)
    good = []
    
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    
    if len(good) < 4:
        return None, None, None
    
    ptsA = np.float32([kpsA[m.queryIdx] for m in good])
    ptsB = np.float32([kpsB[m.trainIdx] for m in good])
    
    H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reproj_thresh)
    return good, H, status


def stitch_pair(img_left, img_right, method="ORB", progress_bar=None):
    """Stitch two images together"""
    kpsA, featsA = detect_and_describe(img_left, method)
    kpsB, featsB = detect_and_describe(img_right, method)
    
    matches, H, status = match_keypoints(kpsB, kpsA, featsB, featsA, method=method)
    
    if H is None:
        raise ValueError("Not enough feature matches to compute homography")
    
    h1, w1 = img_left.shape[:2]
    h2, w2 = img_right.shape[:2]
    
    corners_right = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_right, H)
    
    corners = np.concatenate((
        warped_corners,
        np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    ), axis=0)
    
    [xmin, ymin] = np.int32(corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(corners.max(axis=0).ravel())
    
    translation = [-xmin, -ymin]
    T = np.array([[1, 0, translation[0]],
                  [0, 1, translation[1]],
                  [0, 0, 1]])
    
    result = cv2.warpPerspective(img_right, T @ H, (xmax - xmin, ymax - ymin))
    result[translation[1]:translation[1]+h1, translation[0]:translation[0]+w1] = img_left
    
    if progress_bar:
        progress_bar.progress(100)
    
    return result


def stitch_images(images, method="ORB", progress_callback=None):
    """Stitch multiple images together"""
    if len(images) < 2:
        raise ValueError("Need at least 2 images")
    
    panorama = images[0]
    total = len(images) - 1
    
    for i in range(1, len(images)):
        progress = int((i / total) * 100)
        if progress_callback:
            progress_callback(progress, f"Stitching image {i}/{total}...")
        panorama = stitch_pair(panorama, images[i], method=method)
    
    return panorama


# ============== Streamlit App ==============

st.set_page_config(page_title="Panorama Stitcher", layout="wide")
st.title("🖼️ Panorama Image Stitcher")
st.write("Upload multiple images to create a panoramic view using feature matching and homography!")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    method = st.selectbox("Feature Detection Method", ["ORB", "SIFT"], help="ORB is free, SIFT may need opencv-contrib")
    st.info("**Note:** SIFT requires `opencv-contrib-python`")
    
    resize_enabled = st.checkbox("Resize images (for speed)", value=True)
    max_width = st.slider("Max width (pixels)", 400, 2000, 800) if resize_enabled else 2000

# Main upload section
st.subheader("📤 Upload Images")
uploaded_files = st.file_uploader(
    "Select multiple images (JPG, PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Upload at least 2 overlapping images"
)

if uploaded_files:
    st.success(f"✅ Loaded {len(uploaded_files)} image(s)")
    
    # Display thumbnails
    with st.expander("📸 Preview Uploaded Images"):
        cols = st.columns(min(3, len(uploaded_files)))
        for idx, file in enumerate(uploaded_files):
            with cols[idx % 3]:
                img = Image.open(file)
                st.image(img, caption=f"Image {idx+1}: {file.name}", use_column_width=True)
    
    # Process images
    if st.button("🚀 Stitch Panorama", type="primary"):
        try:
            # Load images
            images = []
            for file in uploaded_files:
                img_array = np.array(Image.open(file))
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Resize if enabled
                if resize_enabled:
                    h, w = img_bgr.shape[:2]
                    if w > max_width:
                        scale = max_width / w
                        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
                
                images.append(img_bgr)
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(percent, message):
                progress_bar.progress(percent)
                status_text.text(message)
            
            # Stitch
            with st.spinner("🔄 Stitching images..."):
                panorama = stitch_images(images, method=method, progress_callback=update_progress)
            
            status_text.text("✅ Stitching complete!")
            
            # Display result
            st.subheader("📍 Result")
            pano_rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
            st.image(pano_rgb, use_column_width=True, caption="Panoramic View")
            
            # Download button
            pil_img = Image.fromarray(pano_rgb)
            buf = BytesIO()
            pil_img.save(buf, format="PNG")
            buf.seek(0)
            
            st.download_button(
                label="⬇️ Download Panorama",
                data=buf,
                file_name="panorama.png",
                mime="image/png",
                type="primary"
            )
            
            # Stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Panorama Width", f"{panorama.shape[1]} px")
            with col2:
                st.metric("Panorama Height", f"{panorama.shape[0]} px")
            with col3:
                st.metric("Images Stitched", len(images))
        
        except Exception as e:
            st.error(f"❌ Error during stitching:\n\n{str(e)}")
            st.info("💡 **Tips for better results:**\n"
                    "- Ensure images overlap by 30-50%\n"
                    "- Keep images in sequence (left-to-right order)\n"
                    "- Use images from same camera/phone\n"
                    "- Avoid very blurry images")
else:
    st.info("👈 Upload at least 2 overlapping images to get started!")

# Footer
st.divider()
st.caption("**How it works:** Features → Match → Homography → Warp → Blend → Panorama ✨")