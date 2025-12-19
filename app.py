import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ==========================
# Config
# ==========================
MAX_INPUT_WIDTH = 800
MAX_PANO_WIDTH = 2500  # Increased to allow wider panoramas like your example


# ==========================
# Utility functions
# ==========================
def resize_to_max_width(img, max_width=MAX_INPUT_WIDTH):
    """Resize image keeping aspect ratio if its width is larger than max_width."""
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / float(w)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def to_gray(img):
    """Convert BGR/BGRA/RGB to grayscale."""
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def crop_black_borders(image, threshold=10):
    """
    Crop black borders by scanning from edges inward.
    Finds the largest clean rectangle with no black pixels.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Scan from top
    top = 0
    for i in range(h):
        if np.min(gray[i, :]) > threshold:
            top = i
            break
    
    # Scan from bottom
    bottom = h - 1
    for i in range(h-1, -1, -1):
        if np.min(gray[i, :]) > threshold:
            bottom = i
            break
    
    # Scan from left
    left = 0
    for j in range(w):
        if np.min(gray[top:bottom+1, j]) > threshold:
            left = j
            break
    
    # Scan from right
    right = w - 1
    for j in range(w-1, -1, -1):
        if np.min(gray[top:bottom+1, j]) > threshold:
            right = j
            break
    
    # Safety check
    if bottom <= top or right <= left:
        return image
    
    # Add small safety margin
    margin = 2
    top = min(top + margin, h-1)
    bottom = max(bottom - margin, 0)
    left = min(left + margin, w-1)
    right = max(right - margin, 0)
    
    return image[top:bottom+1, left:right+1]


# ==========================
# Core stitching
# ==========================
def stitch_pair(img1, img2, ratio=0.75, reproj_thresh=4.0):
    """
    Stitch two images using SIFT + Homography.
    """
    # 1) Detect & describe features with SIFT
    sift = cv2.SIFT_create()
    gray1, gray2 = to_gray(img1), to_gray(img2)
    kps1, des1 = sift.detectAndCompute(gray1, None)
    kps2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        raise ValueError("Could not find enough SIFT features in one of the images")

    # 2) Match with BFMatcher + Lowe ratio test
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)

    if len(good) < 4:
        raise ValueError(f"Not enough matches (found {len(good)}, need at least 4)")

    pts1 = np.float32([kps1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in good])

    # 3) Homography with RANSAC
    H, status = cv2.findHomography(pts2, pts1, cv2.RANSAC, reproj_thresh)
    if H is None:
        raise ValueError("Homography estimation failed")

    # 4) Calculate output size
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    warped_corners2 = cv2.perspectiveTransform(corners2, H)
    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    all_corners = np.concatenate((corners1, warped_corners2), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation = [-xmin, -ymin]
    T = np.array([[1, 0, translation[0]],
                  [0, 1, translation[1]],
                  [0, 0, 1]])

    # 5) Warp img2 and blend with img1
    output_size = (xmax - xmin, ymax - ymin)
    result = cv2.warpPerspective(img2, T @ H, output_size)
    
    # Simple overwrite blending
    result[translation[1]:translation[1] + h1,
           translation[0]:translation[0] + w1] = img1

    return result


def stitch_images(images):
    """
    Stitch multiple images sequentially to create a wide panorama.
    """
    if len(images) == 0:
        raise ValueError("No images to stitch")
    
    if len(images) == 1:
        return images[0]
    
    # Start with first image
    pano = images[0].copy()
    
    # Stitch each subsequent image
    for i in range(1, len(images)):
        st.write(f"Stitching image {i+1}/{len(images)}...")
        pano = stitch_pair(pano, images[i])
        
        # Intermediate crop after each stitch to keep size manageable
        if i < len(images) - 1:
            pano = crop_black_borders(pano, threshold=5)
        
        # Memory safety
        h, w = pano.shape[:2]
        if w > MAX_PANO_WIDTH * 2:
            scale = (MAX_PANO_WIDTH * 2) / float(w)
            new_size = (int(w * scale), int(h * scale))
            pano = cv2.resize(pano, new_size, interpolation=cv2.INTER_AREA)
    
    # Final aggressive crop to remove ALL black borders
    pano = crop_black_borders(pano, threshold=5)
    
    return pano


# ==========================
# Streamlit app
# ==========================
st.title("🌄 Panoramic Image Stitching")

st.write(
    "Upload 2 or more overlapping images taken from left to right "
    "to create a clean wide panoramic view."
)

# Settings in sidebar
with st.sidebar:
    st.header("Settings")
    custom_width = st.slider(
        "Input Image Max Width (pixels)",
        min_value=400,
        max_value=1500,
        value=800,
        step=100,
        help="Lower values use less memory but reduce quality"
    )
    MAX_INPUT_WIDTH = custom_width
    
    st.info(
        "💡 **Tips:**\n"
        "- Upload images in left-to-right order\n"
        "- Images should have 30-50% overlap\n"
        "- Lower the max width if you get memory errors\n"
        "- For 4+ images, try 600-800px max width"
    )

uploaded_files = st.file_uploader(
    "Choose images (JPG, PNG, TIFF, BMP, ...)",
    type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"],
    accept_multiple_files=True,
)

if uploaded_files:
    if len(uploaded_files) < 2:
        st.warning("⚠️ Please upload at least 2 images.")
    else:
        st.success(f"✅ {len(uploaded_files)} images uploaded")
        
        # Show uploaded images
        with st.expander("Preview Uploaded Images", expanded=False):
            cols = st.columns(min(len(uploaded_files), 4))
            for idx, uploaded in enumerate(uploaded_files):
                col_idx = idx % 4
                pil_img = Image.open(uploaded).convert("RGB")
                cols[col_idx].image(pil_img, caption=f"Image {idx+1}", use_container_width=True)

        if st.button("🚀 Stitch Panorama", type="primary"):
            cv_images = []
            
            with st.spinner("Loading and resizing images..."):
                for idx, uploaded in enumerate(uploaded_files):
                    pil_img = Image.open(uploaded).convert("RGB")
                    cv_img = np.array(pil_img)
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
                    cv_img = resize_to_max_width(cv_img, MAX_INPUT_WIDTH)
                    cv_images.append(cv_img)
            
            try:
                with st.spinner(f"Creating panorama from {len(cv_images)} images..."):
                    panorama = stitch_images(cv_images)
                    panorama_rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
                
                st.success("✨ Clean panorama created successfully!")
                st.image(panorama_rgb, caption="Final Panoramic View", use_container_width=True)
                
                # Download option
                pil_result = Image.fromarray(panorama_rgb)
                from io import BytesIO
                buf = BytesIO()
                pil_result.save(buf, format="PNG")
                st.download_button(
                    label="⬇️ Download Panorama",
                    data=buf.getvalue(),
                    file_name="panorama.png",
                    mime="image/png"
                )
                
            except Exception as e:
                st.error(f"❌ Stitching failed: {e}")
                st.write(
                    "**Troubleshooting:**\n"
                    "- Try reducing the 'Input Image Max Width' in the sidebar\n"
                    "- Make sure images overlap by 30-50%\n"
                    "- Upload images in the correct order (left to right)\n"
                    "- Try with fewer images first (2-3) to test"
                )
else:
    st.info("👆 Upload your images to begin.")
