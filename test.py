import streamlit as st
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage

# ============== Enhanced Stitching Functions ==============

def validate_images(images):
    """Check if images are suitable for stitching"""
    if len(images) < 2:
        return False, "Need at least 2 images"
    
    # Check dimensions
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    
    if max(heights) / min(heights) > 3 or max(widths) / min(widths) > 3:
        return False, "Images vary too much in size"
    
    return True, "Images validated"

def detect_and_describe(image, method="ORB", max_features=5000):
    """Detect and compute keypoints and descriptors"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if method == "SIFT":
        sift = cv2.SIFT_create(nfeatures=max_features)
        kps, features = sift.detectAndCompute(gray, None)
    elif method == "SURF":
        # Try to create SURF detector (requires opencv-contrib)
        try:
            surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
            kps, features = surf.detectAndCompute(gray, None)
        except:
            # Fall back to SIFT if SURF not available
            sift = cv2.SIFT_create(nfeatures=max_features)
            kps, features = sift.detectAndCompute(gray, None)
    else:  # ORB
        orb = cv2.ORB_create(nfeatures=max_features)
        kps, features = orb.detectAndCompute(gray, None)
    
    kps = np.float32([kp.pt for kp in kps])
    return kps, features

def match_keypoints(kpsA, kpsB, featsA, featsB, ratio=0.75, method="ORB"):
    """Match keypoints between two images using Lowe's ratio test"""
    if method == "SIFT" or method == "SURF":
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:  # ORB
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    raw_matches = matcher.knnMatch(featsA, featsB, k=2)
    good_matches = []
    
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good_matches.append(m)
    
    return good_matches

def compute_homography(kpsA, kpsB, matches, reproj_thresh=4.0):
    """Compute homography from matched keypoints"""
    if len(matches) < 4:
        return None, None
    
    ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
    ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
    
    H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reproj_thresh)
    return H, mask

def create_mask(image):
    """Create a mask for the image (non-black pixels)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.uint8(gray > 10) * 255
    return mask

def feather_blending(img1, img2, mask1, mask2):
    """Feather (linear) blending for overlapping regions"""
    # Find overlap region
    overlap = cv2.bitwise_and(mask1, mask2)
    
    if np.sum(overlap) == 0:
        # No overlap, just combine
        result = cv2.add(img1, img2)
        return result
    
    # Create weight maps
    y_indices, x_indices = np.where(overlap > 0)
    if len(x_indices) == 0:
        return cv2.add(img1, img2)
    
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    
    # Create linear gradient weights
    weights = np.zeros_like(img1, dtype=np.float32)
    
    # For pixels in overlap region
    for x in range(x_min, x_max + 1):
        if x < x_min or x > x_max:
            continue
        
        # Linear weighting from left to right
        weight = (x - x_min) / (x_max - x_min) if x_max > x_min else 0.5
        weights[:, x] = weight
    
    # Apply blending
    result = np.zeros_like(img1, dtype=np.float32)
    result = img1.astype(np.float32) * (1 - weights) + img2.astype(np.float32) * weights
    
    # For non-overlap regions
    non_overlap1 = cv2.bitwise_and(mask1, cv2.bitwise_not(overlap))
    non_overlap2 = cv2.bitwise_and(mask2, cv2.bitwise_not(overlap))
    
    result[non_overlap1 > 0] = img1[non_overlap1 > 0]
    result[non_overlap2 > 0] = img2[non_overlap2 > 0]
    
    return np.clip(result, 0, 255).astype(np.uint8)

def multi_band_blending(img1, img2, mask1, mask2, levels=5):
    """Multi-band blending for smoother transitions"""
    # Find overlap region
    overlap = cv2.bitwise_and(mask1, mask2)
    
    if np.sum(overlap) == 0:
        return cv2.add(img1, img2)
    
    # Create Gaussian pyramids
    gauss_pyramid1 = [img1.astype(np.float32)]
    gauss_pyramid2 = [img2.astype(np.float32)]
    mask_pyramid = [overlap.astype(np.float32) / 255.0]
    
    for i in range(1, levels):
        gauss_pyramid1.append(cv2.pyrDown(gauss_pyramid1[-1]))
        gauss_pyramid2.append(cv2.pyrDown(gauss_pyramid2[-1]))
        mask_pyramid.append(cv2.pyrDown(mask_pyramid[-1]))
    
    # Create Laplacian pyramids
    laplacian_pyramid1 = [gauss_pyramid1[levels-1]]
    laplacian_pyramid2 = [gauss_pyramid2[levels-1]]
    
    for i in range(levels-2, -1, -1):
        size = (gauss_pyramid1[i].shape[1], gauss_pyramid1[i].shape[0])
        expanded1 = cv2.pyrUp(gauss_pyramid1[i+1], dstsize=size)
        expanded2 = cv2.pyrUp(gauss_pyramid2[i+1], dstsize=size)
        
        laplacian1 = gauss_pyramid1[i] - expanded1
        laplacian2 = gauss_pyramid2[i] - expanded2
        
        laplacian_pyramid1.append(laplacian1)
        laplacian_pyramid2.append(laplacian2)
    
    laplacian_pyramid1.reverse()
    laplacian_pyramid2.reverse()
    
    # Blend pyramids
    blended_pyramid = []
    for lap1, lap2, mask in zip(laplacian_pyramid1, laplacian_pyramid2, mask_pyramid):
        mask_3channel = np.stack([mask, mask, mask], axis=2)
        blended = lap1 * (1 - mask_3channel) + lap2 * mask_3channel
        blended_pyramid.append(blended)
    
    # Reconstruct
    result = blended_pyramid[0]
    for i in range(1, levels):
        size = (blended_pyramid[i].shape[1], blended_pyramid[i].shape[0])
        result = cv2.pyrUp(result, dstsize=size) + blended_pyramid[i]
    
    # Handle non-overlap regions
    non_overlap1 = cv2.bitwise_and(mask1, cv2.bitwise_not(overlap))
    non_overlap2 = cv2.bitwise_and(mask2, cv2.bitwise_not(overlap))
    
    result[non_overlap1 > 0] = img1[non_overlap1 > 0]
    result[non_overlap2 > 0] = img2[non_overlap2 > 0]
    
    return np.clip(result, 0, 255).astype(np.uint8)

def stitch_pair_with_blending(img_left, img_right, method="ORB", blending="feather", max_features=5000):
    """Stitch two images together with blending"""
    # Detect features
    kpsA, featsA = detect_and_describe(img_left, method, max_features)
    kpsB, featsB = detect_and_describe(img_right, method, max_features)
    
    # Match features
    matches = match_keypoints(kpsA, kpsB, featsA, featsB, method=method)
    
    if len(matches) < 10:
        raise ValueError(f"Not enough feature matches ({len(matches)}). Need at least 10.")
    
    # Compute homography (right image projected onto left image's plane)
    H, mask = compute_homography(kpsB, kpsA, matches)
    
    if H is None:
        raise ValueError("Could not compute homography")
    
    # Get image dimensions
    h1, w1 = img_left.shape[:2]
    h2, w2 = img_right.shape[:2]
    
    # Calculate the size of the output panorama
    corners_right = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    corners_right_transformed = cv2.perspectiveTransform(corners_right, H)
    
    all_corners = np.concatenate((corners_right_transformed, 
                                  np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)), axis=0)
    
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # Translation matrix to keep all pixels positive
    translation = [-xmin, -ymin]
    T = np.array([[1, 0, translation[0]],
                  [0, 1, translation[1]],
                  [0, 0, 1]])
    
    # Warp the right image
    output_width = xmax - xmin
    output_height = ymax - ymin
    warped_right = cv2.warpPerspective(img_right, T @ H, (output_width, output_height))
    
    # Create panorama canvas
    panorama = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # Place left image
    left_x_start = translation[0]
    left_y_start = translation[1]
    left_x_end = left_x_start + w1
    left_y_end = left_y_start + h1
    
    panorama[left_y_start:left_y_end, left_x_start:left_x_end] = img_left
    
    # Create masks for blending
    mask_left = create_mask(panorama)
    mask_right = create_mask(warped_right)
    
    # Apply blending
    if blending == "multi-band":
        panorama = multi_band_blending(panorama, warped_right, mask_left, mask_right)
    elif blending == "feather":
        panorama = feather_blending(panorama, warped_right, mask_left, mask_right)
    else:  # no blending
        # Simple overlay (right image overwrites left in overlap)
        overlap = cv2.bitwise_and(mask_left, mask_right)
        non_overlap_right = cv2.bitwise_and(mask_right, cv2.bitwise_not(overlap))
        panorama[non_overlap_right > 0] = warped_right[non_overlap_right > 0]
    
    return panorama

def estimate_pairwise_overlap(img1, img2, method="ORB"):
    """Estimate overlap percentage between two images"""
    kpsA, featsA = detect_and_describe(img1, method)
    kpsB, featsB = detect_and_describe(img2, method)
    
    matches = match_keypoints(kpsA, kpsB, featsA, featsB, method=method)
    
    if len(matches) < 4:
        return 0
    
    # Rough estimate: more matches = more overlap
    max_matches = min(len(kpsA), len(kpsB))
    overlap_ratio = len(matches) / max_matches if max_matches > 0 else 0
    
    return overlap_ratio

def order_images_by_overlap(images, method="ORB"):
    """Arrange images by their pairwise overlap"""
    n = len(images)
    if n <= 2:
        return list(range(n))
    
    # Compute overlap matrix
    overlap_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            overlap = estimate_pairwise_overlap(images[i], images[j], method)
            overlap_matrix[i, j] = overlap
            overlap_matrix[j, i] = overlap
    
    # Find image with most connections (most overlaps)
    total_overlaps = np.sum(overlap_matrix > 0.1, axis=1)  # Threshold 10% overlap
    center_idx = np.argmax(total_overlaps)
    
    # Order images by distance from center
    ordered = [center_idx]
    visited = set([center_idx])
    
    # Simple BFS-like ordering
    while len(ordered) < n:
        max_overlap = -1
        next_idx = -1
        
        for i in range(n):
            if i not in visited:
                # Find overlap with any visited image
                for j in visited:
                    if overlap_matrix[i, j] > max_overlap:
                        max_overlap = overlap_matrix[i, j]
                        next_idx = i
        
        if next_idx == -1:
            # No more connected images, add remaining in order
            for i in range(n):
                if i not in visited:
                    ordered.append(i)
                    visited.add(i)
            break
        
        ordered.append(next_idx)
        visited.add(next_idx)
    
    return ordered

def crop_black_borders(panorama):
    """Remove black borders from panorama"""
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find non-black pixels
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return panorama
    
    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add small padding
    padding = 5
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(panorama.shape[1] - x, w + 2 * padding)
    h = min(panorama.shape[0] - y, h + 2 * padding)
    
    return panorama[y:y+h, x:x+w]

def exposure_compensation(images):
    """Simple exposure compensation by matching histograms"""
    if len(images) < 2:
        return images
    
    # Convert to LAB color space for better exposure matching
    lab_images = []
    for img in images:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_images.append(lab)
    
    # Use middle image as reference
    ref_idx = len(lab_images) // 2
    ref_lab = lab_images[ref_idx]
    
    compensated_images = []
    for i, lab in enumerate(lab_images):
        if i == ref_idx:
            compensated_images.append(images[i])
            continue
        
        # Match L channel histogram
        ref_l = ref_lab[:,:,0]
        src_l = lab[:,:,0]
        
        # Compute histograms
        ref_hist = cv2.calcHist([ref_l], [0], None, [256], [0,256])
        src_hist = cv2.calcHist([src_l], [0], None, [256], [0,256])
        
        # Compute CDFs
        ref_cdf = ref_hist.cumsum()
        src_cdf = src_hist.cumsum()
        
        # Normalize CDFs
        ref_cdf = ref_cdf / ref_cdf[-1]
        src_cdf = src_cdf / src_cdf[-1]
        
        # Create lookup table
        lut = np.interp(src_cdf, ref_cdf, np.arange(256))
        lut = np.clip(lut, 0, 255).astype(np.uint8)
        
        # Apply lookup table
        matched_l = cv2.LUT(src_l, lut)
        
        # Reconstruct LAB image
        matched_lab = lab.copy()
        matched_lab[:,:,0] = matched_l
        
        # Convert back to BGR
        matched_bgr = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)
        compensated_images.append(matched_bgr)
    
    return compensated_images

def stitch_images_improved(images, method="ORB", blending="feather", 
                          auto_order=True, crop=True, exposure_comp=True,
                          max_features=5000, progress_callback=None):
    """Enhanced stitching with multiple improvements"""
    
    # Validate images
    is_valid, msg = validate_images(images)
    if not is_valid:
        raise ValueError(msg)
    
    # Exposure compensation
    if exposure_comp:
        images = exposure_compensation(images)
    
    # Order images if requested
    if auto_order and len(images) > 2:
        ordered_indices = order_images_by_overlap(images, method)
        images = [images[i] for i in ordered_indices]
    
    # Use center reference strategy for better accuracy
    center_idx = len(images) // 2
    panorama = images[center_idx]
    
    total_steps = len(images) - 1
    current_step = 0
    
    # Stitch left side (from center to left)
    for i in range(center_idx-1, -1, -1):
        if progress_callback:
            current_step += 1
            progress = int((current_step / total_steps) * 100)
            progress_callback(progress, f"Stitching image {i+1} from left...")
        
        panorama = stitch_pair_with_blending(
            images[i], panorama, method, blending, max_features
        )
    
    # Stitch right side (from center to right)
    for i in range(center_idx+1, len(images)):
        if progress_callback:
            current_step += 1
            progress = int((current_step / total_steps) * 100)
            progress_callback(progress, f"Stitching image {i+1} from right...")
        
        panorama = stitch_pair_with_blending(
            panorama, images[i], method, blending, max_features
        )
    
    # Post-processing
    if crop:
        panorama = crop_black_borders(panorama)
    
    return panorama

def visualize_matches(img1, img2, method="ORB", max_features=5000):
    """Visualize feature matches between two images"""
    kps1, feats1 = detect_and_describe(img1, method, max_features)
    kps2, feats2 = detect_and_describe(img2, method, max_features)
    
    matches = match_keypoints(kps1, kps2, feats1, feats2, method=method)
    
    # Convert images to RGB for display
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Draw matches
    match_img = cv2.drawMatches(
        img1_rgb, [cv2.KeyPoint(x[0], x[1], 1) for x in kps1.astype(int)],
        img2_rgb, [cv2.KeyPoint(x[0], x[1], 1) for x in kps2.astype(int)],
        matches[:50],  # Show first 50 matches
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    return match_img

# ============== Streamlit GUI ==============

def main():
    st.set_page_config(page_title="Enhanced Panorama Stitcher", layout="wide")
    
    st.title("🌅 Enhanced Panorama Image Stitcher")
    st.write("Create seamless panoramic views with advanced feature matching, blending, and alignment!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Feature detection
        st.subheader("🔍 Feature Detection")
        method = st.selectbox("Feature Method", ["ORB", "SIFT", "SURF"], 
                             help="ORB: Fast and free | SIFT: More accurate but patented | SURF: Fast alternative to SIFT")
        max_features = st.slider("Max Features", 100, 10000, 5000, 100,
                                help="More features = more accurate but slower")
        match_ratio = st.slider("Match Ratio (Lowe's)", 0.5, 0.9, 0.75, 0.05,
                               help="Lower = stricter matching")
        
        # Stitching options
        st.subheader("🔄 Stitching Options")
        auto_order = st.checkbox("Auto-detect image order", value=True,
                                help="Automatically arrange images based on overlap")
        exposure_comp = st.checkbox("Exposure Compensation", value=True,
                                   help="Match exposure between images")
        crop_result = st.checkbox("Crop black borders", value=True,
                                 help="Remove unused black areas")
        
        # Blending options
        st.subheader("🎨 Blending Options")
        blending_method = st.selectbox("Blending Method", 
                                      ["feather", "multi-band", "none"],
                                      help="Feather: Linear blending | Multi-band: Smoother but slower | None: Simple overlay")
        
        # Performance
        st.subheader("⚡ Performance")
        resize_enabled = st.checkbox("Resize large images", value=True)
        max_width = st.slider("Max width (pixels)", 400, 2000, 1200) if resize_enabled else 2000
        
        # Advanced
        with st.expander("Advanced Options"):
            reproj_thresh = st.slider("RANSAC Threshold", 1.0, 10.0, 4.0, 0.5,
                                     help="Higher = more tolerant to outliers")
            show_matches = st.checkbox("Show feature matches", value=False)
    
    # Main upload section
    st.subheader("📤 Upload Images")
    uploaded_files = st.file_uploader(
        "Select multiple overlapping images (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload 2-10 overlapping images for best results"
    )
    
    if uploaded_files and len(uploaded_files) > 0:
        st.success(f"✅ Loaded {len(uploaded_files)} image(s)")
        
        # Display thumbnails
        with st.expander("📸 Preview Uploaded Images", expanded=True):
            cols = st.columns(min(4, len(uploaded_files)))
            for idx, file in enumerate(uploaded_files):
                with cols[idx % 4]:
                    img = Image.open(file)
                    st.image(img, caption=f"Image {idx+1}: {file.name}", use_column_width=True)
        
        # Process images button
        if st.button("🚀 Stitch Panorama", type="primary", use_container_width=True):
            try:
                # Load and process images
                images = []
                original_sizes = []
                
                with st.spinner("📥 Loading images..."):
                    for file in uploaded_files:
                        img_array = np.array(Image.open(file))
                        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        
                        # Store original size
                        original_sizes.append(img_bgr.shape[:2])
                        
                        # Resize if enabled
                        if resize_enabled:
                            h, w = img_bgr.shape[:2]
                            if w > max_width:
                                scale = max_width / w
                                new_w = max_width
                                new_h = int(h * scale)
                                img_bgr = cv2.resize(img_bgr, (new_w, new_h))
                        
                        images.append(img_bgr)
                
                # Validate
                is_valid, msg = validate_images(images)
                if not is_valid:
                    st.error(f"❌ Validation failed: {msg}")
                    st.stop()
                
                # Show feature matches if requested
                if show_matches and len(images) >= 2:
                    st.subheader("🔗 Feature Matches")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        img_idx1 = st.number_input("First image for matching", 1, len(images), 1) - 1
                    with col2:
                        img_idx2 = st.number_input("Second image for matching", 1, len(images), 2) - 1
                    
                    if img_idx1 != img_idx2:
                        match_img = visualize_matches(images[img_idx1], images[img_idx2], method, max_features)
                        st.image(match_img, caption=f"Feature matches between image {img_idx1+1} and {img_idx2+1}", 
                                use_column_width=True)
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(percent, message):
                    progress_bar.progress(percent)
                    status_text.text(message)
                
                # Stitch images
                with st.spinner("🔄 Creating panorama..."):
                    panorama = stitch_images_improved(
                        images=images,
                        method=method,
                        blending=blending_method,
                        auto_order=auto_order,
                        crop=crop_result,
                        exposure_comp=exposure_comp,
                        max_features=max_features,
                        progress_callback=update_progress
                    )
                
                status_text.text("✅ Stitching complete!")
                
                # Display results
                st.subheader("📍 Panorama Result")
                
                # Convert to RGB for display
                panorama_rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.image(panorama_rgb, use_column_width=True, caption="Final Panorama")
                
                with col2:
                    st.metric("Final Width", f"{panorama.shape[1]} px")
                    st.metric("Final Height", f"{panorama.shape[0]} px")
                    st.metric("Images Used", len(images))
                    st.metric("Scale Factor", f"{panorama.shape[1] / sum(s[1] for s in original_sizes):.2f}")
                
                # Download section
                st.subheader("💾 Download Result")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # PNG download
                    png_buf = BytesIO()
                    Image.fromarray(panorama_rgb).save(png_buf, format='PNG', quality=95)
                    st.download_button(
                        label="⬇️ Download PNG",
                        data=png_buf.getvalue(),
                        file_name="panorama.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                with col2:
                    # JPEG download
                    jpg_buf = BytesIO()
                    Image.fromarray(panorama_rgb).save(jpg_buf, format='JPEG', quality=95)
                    st.download_button(
                        label="⬇️ Download JPEG",
                        data=jpg_buf.getvalue(),
                        file_name="panorama.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
                
                with col3:
                    # Detailed info
                    with st.expander("📊 Details"):
                        st.write(f"**Processing Details:**")
                        st.write(f"- Feature method: {method}")
                        st.write(f"- Blending: {blending_method}")
                        st.write(f"- Images: {len(images)}")
                        st.write(f"- Output size: {panorama.shape[1]} × {panorama.shape[0]}")
                        if resize_enabled:
                            st.write(f"- Resized to max width: {max_width}px")
                
                # Tips section
                st.divider()
                with st.expander("💡 Tips for Better Results"):
                    st.markdown("""
                    **For best panorama quality:**
                    1. **Overlap**: Ensure 30-50% overlap between consecutive images
                    2. **Consistency**: Use same camera settings for all images
                    3. **Stability**: Keep camera level and rotate around lens center
                    4. **Lighting**: Avoid extreme exposure changes
                    5. **Order**: Images don't need to be in perfect order (enable auto-order)
                    
                    **Troubleshooting:**
                    - **Blurry results**: Try increasing max features
                    - **Visible seams**: Use multi-band blending
                    - **Alignment issues**: Ensure sufficient overlap
                    - **Failed stitching**: Try SIFT instead of ORB
                    """)
            
            except Exception as e:
                st.error(f"❌ Error during stitching: {str(e)}")
                
                with st.expander("🔧 Debug Information"):
                    st.code(f"Error type: {type(e).__name__}")
                    st.code(f"Error details: {str(e)}")
                    
                    if "OpenCV" in str(e):
                        st.info("**OpenCV issues:** Try installing opencv-contrib-python: `pip install opencv-contrib-python`")
                    elif "homography" in str(e).lower():
                        st.info("**Homography issues:** Images may not have enough overlap or may be too different")
                
                st.info("🔄 **Quick fixes to try:**\n"
                       "1. Upload images with more overlap (30-50%)\n"
                       "2. Try the SIFT feature method\n"
                       "3. Disable auto-order if images are already in correct sequence\n"
                       "4. Increase max features to 8000\n"
                       "5. Use similar images (same lighting, same camera)")
    
    else:
        # Welcome/instructions
        st.info("👈 **Upload at least 2 overlapping images to get started!**")
        
        with st.expander("📚 How to use this tool"):
            st.markdown("""
            **Step-by-Step Guide:**
            1. **Upload** 2 or more overlapping images using the uploader above
            2. **Configure** settings in the sidebar (defaults work for most cases)
            3. **Click** "Stitch Panorama" button
            4. **Download** your panoramic image
            
            **Example Workflow:**
            - Take multiple photos by panning your camera horizontally
            - Ensure each photo overlaps with the previous one by 30-50%
            - Upload all photos here
            - Use default settings for first attempt
            - Adjust settings if needed for better results
            """)
        
        # Example images (optional)
        st.divider()
        st.subheader("🖼️ Example Image Layout")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("https://via.placeholder.com/400x300/4A90E2/FFFFFF?text=Image+1", 
                    caption="First image (left)")
        with col2:
            st.image("https://via.placeholder.com/400x300/50E3C2/FFFFFF?text=Image+2", 
                    caption="Second image (30-50% overlap)")
        with col3:
            st.image("https://via.placeholder.com/400x300/9013FE/FFFFFF?text=Image+3", 
                    caption="Third image (continue pattern)")
        
        st.caption("*Ideal image sequence with consistent overlap*")
    
    # Footer
    st.divider()
    st.caption("""
    **Enhanced Panorama Stitcher** | Features: ORB/SIFT/SURF • Homography • Multi-band Blending • Exposure Compensation • Auto-ordering
    """)

# ============== Installation Instructions ==============

def show_installation():
    st.sidebar.divider()
    with st.sidebar.expander("📦 Installation"):
        st.code("""
# Required packages:
pip install streamlit opencv-python 
pip install opencv-contrib-python  # For SIFT/SURF
pip install numpy pillow scipy matplotlib
        """, language="bash")
        
        st.info("""
        **Note for SIFT/SURF:**
        - SIFT is now free in OpenCV 4.4.0+
        - SURF requires opencv-contrib-python
        - ORB works with basic opencv-python
        """)

# Run the app
if __name__ == "__main__":
    show_installation()
    main()