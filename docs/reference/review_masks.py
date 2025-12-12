#!/usr/bin/env python3
"""
Batch Mask Review Tool v2
=========================
Loops through images in the review folder and allows interactive editing.
Saves corrected masks back to the masks folder.

Features:
- Brush painting (add/remove)
- Click-to-segment (flood fill for connected regions)
- Help overlay with all controls
"""
import cv2
import numpy as np
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MaskReviewer:
    def __init__(self):
        self.current_image = None
        self.current_mask = None
        self.original_mask = None
        self.history = []
        self.drawing = False
        self.brush_size = 20
        self.mode = 1  # 1: add, 0: remove
        self.show_mask = True
        self.show_help = True  # Initially visible, press 'h' to toggle
        self.flood_tolerance = 15  # Color tolerance for flood fill (lower = more precise)
        self.brightness_mode = False  # Toggle with 'b' - uses brightness only (good for shadows)
        self.mouse_x = 0
        self.mouse_y = 0
        # Zoom and pan
        self.zoom_level = 1.0
        self.pan_x = 0  # Pan offset in original image coordinates
        self.pan_y = 0
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        # Lasso selection
        self.lasso_points = []  # Points being drawn for lasso
        self.drawing_lasso = False
        
    def _mouse_callback(self, event, x, y, flags, param):
        # Ctrl + drag = lasso selection (draw shape, then Otsu threshold inside)
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing_lasso = True
                self.lasso_points = [(x, y)]
                self.mode = 1  # Add mode for lasso
                return
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.drawing_lasso = True
                self.lasso_points = [(x, y)]
                self.mode = 0  # Remove mode for lasso
                return
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing_lasso:
                self.lasso_points.append((x, y))
                self.mouse_x, self.mouse_y = x, y
                return
            elif (event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP) and self.drawing_lasso:
                if len(self.lasso_points) > 2:
                    self._apply_lasso_selection()
                    self.history.append(self.current_mask.copy())
                self.drawing_lasso = False
                self.lasso_points = []
                return
        
        # Cancel lasso if ctrl released during draw
        if self.drawing_lasso and not (flags & cv2.EVENT_FLAG_CTRLKEY):
            self.drawing_lasso = False
            self.lasso_points = []
        
        # Shift + click = flood fill segment
        if event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_SHIFTKEY):
            self._flood_fill_add(x, y)
            self.history.append(self.current_mask.copy())
            return
        elif event == cv2.EVENT_RBUTTONDOWN and (flags & cv2.EVENT_FLAG_SHIFTKEY):
            self._flood_fill_remove(x, y)
            self.history.append(self.current_mask.copy())
            return
        
        # Middle click = flood fill
        if event == cv2.EVENT_MBUTTONDOWN:
            if self.mode == 1:
                self._flood_fill_add(x, y)
            else:
                self._flood_fill_remove(x, y)
            self.history.append(self.current_mask.copy())
            return
        
        # Regular brush
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.mode = 1  # Add mode
            self._draw_at(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = True
            self.mode = 0  # Remove mode
            self._draw_at(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x, self.mouse_y = x, y
            if self.drawing:
                self._draw_at(x, y)
        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            if self.drawing:
                self.history.append(self.current_mask.copy())
            self.drawing = False
        
        # Scroll wheel = zoom
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:  # Scroll up = zoom in
                self.zoom_level = min(4.0, self.zoom_level * 1.2)
            else:  # Scroll down = zoom out
                self.zoom_level = max(0.25, self.zoom_level / 1.2)
            # Adjust pan to zoom toward mouse position
            self.pan_x = x - (x - self.pan_x) * (self.zoom_level / (self.zoom_level / 1.2 if flags > 0 else self.zoom_level * 1.2))
            self.pan_y = y - (y - self.pan_y) * (self.zoom_level / (self.zoom_level / 1.2 if flags > 0 else self.zoom_level * 1.2))
    
    def _draw_at(self, x, y):
        cv2.circle(self.current_mask, (x, y), self.brush_size, self.mode, -1)
    
    def _apply_lasso_selection(self):
        """Apply Otsu thresholding within the drawn lasso polygon."""
        if len(self.lasso_points) < 3:
            return
        
        h, w = self.current_image.shape[:2]
        
        # Create polygon mask from lasso points
        polygon = np.array(self.lasso_points, dtype=np.int32)
        lasso_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(lasso_mask, [polygon], 1)
        
        # Convert image to grayscale
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        # Extract pixels within the lasso region
        lasso_pixels = gray[lasso_mask > 0]
        
        if len(lasso_pixels) < 10:
            return
        
        # Use Otsu's method to find optimal threshold between dark (shadow) and light (surface)
        otsu_threshold, _ = cv2.threshold(lasso_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Adjust threshold based on tolerance setting (lower tolerance = stricter, higher = more permissive)
        # At tolerance=50 (default-ish), use Otsu directly
        # At tolerance=0, use threshold - 20 (stricter)
        # At tolerance=100, use threshold + 20 (more permissive)
        adjustment = (self.flood_tolerance - 50) * 0.4
        threshold = otsu_threshold + adjustment
        
        # Create selection: pixels darker than threshold within the lasso
        dark_pixels = ((gray < threshold) & (lasso_mask > 0)).astype(np.uint8)
        
        # Morphological opening to remove small isolated spots (grass texture)
        kernel = np.ones((5, 5), np.uint8)
        dark_pixels = cv2.morphologyEx(dark_pixels, cv2.MORPH_OPEN, kernel)
        
        # Keep only the largest connected component (the main shadow body)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_pixels)
        
        if num_labels > 1:  # More than just background
            # Find the largest non-background component
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            dark_pixels = (labels == largest_label).astype(np.uint8)
        
        # Apply to mask based on mode
        if self.mode == 1:  # Add
            self.current_mask = np.maximum(self.current_mask, dark_pixels)
        else:  # Remove
            self.current_mask = np.where(dark_pixels > 0, 0, self.current_mask).astype(np.uint8)
        
        logger.info(f"Lasso: Otsu={otsu_threshold:.0f}, adjusted={threshold:.0f}, selected {dark_pixels.sum()} pixels")
    
    def _flood_fill_add(self, x, y):
        """Flood fill to add a region. Uses brightness-only or LAB color based on mode."""
        h, w = self.current_image.shape[:2]
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        if self.brightness_mode:
            # RELATIVE darkness mode: compare each pixel to its local neighborhood
            # Shadow = darker than surroundings. Dark texture = similar to surroundings.
            
            # Compute local brightness using a blur (approximation of local context)
            # 51x51 kernel considers ~50px neighborhood
            local_median = cv2.blur(gray, (51, 51))
            
            # A pixel is "in shadow" if it's significantly darker than its local neighborhood
            darkness_ratio = gray.astype(np.float32) / (local_median.astype(np.float32) + 1)
            
            # Threshold: tolerance=30 means pixel must be 30% darker than surroundings
            # At tolerance=100, threshold=0 meaning any darker pixel qualifies
            threshold = 1.0 - (self.flood_tolerance / 100.0)
            shadow_mask = (darkness_ratio < threshold).astype(np.uint8)
            
            # Morphological closing to bridge small gaps (gravel texture, grass blades)
            # This connects nearby shadow regions that are separated by bright texture
            kernel_size = max(5, self.flood_tolerance // 5)  # Larger tolerance = more closing
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find connected components
            num_labels, labels = cv2.connectedComponents(shadow_mask)
            
            # Get the label at click point
            click_label = labels[y, x]
            
            if click_label > 0:
                filled_region = (labels == click_label).astype(np.uint8)
            else:
                filled_region = np.zeros((h, w), np.uint8)
        else:
            # LAB color mode: use edge detection + flood fill
            flood_mask = np.zeros((h + 2, w + 2), np.uint8)
            edges = cv2.Canny(gray, 30, 100)
            edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
            flood_mask[1:-1, 1:-1] = (edges > 0).astype(np.uint8)
            
            lab_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2LAB)
            lo_diff = (self.flood_tolerance, self.flood_tolerance // 2, self.flood_tolerance // 2)
            hi_diff = (self.flood_tolerance, self.flood_tolerance // 2, self.flood_tolerance // 2)
            lab_copy = lab_image.copy()
            
            cv2.floodFill(
                lab_copy,
                flood_mask,
                (x, y),
                (255, 128, 128),
                lo_diff,
                hi_diff,
                cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
            )
            
            filled_region = flood_mask[1:-1, 1:-1]
            filled_region = np.where(edges > 0, 0, filled_region).astype(np.uint8)
        
        self.current_mask = np.maximum(self.current_mask, filled_region)
        
    def _flood_fill_remove(self, x, y):
        """Flood fill to remove a region. Uses brightness-only or LAB color based on mode."""
        h, w = self.current_image.shape[:2]
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        if self.brightness_mode:
            # RELATIVE darkness mode: compare each pixel to its local neighborhood
            local_median = cv2.blur(gray, (51, 51))
            darkness_ratio = gray.astype(np.float32) / (local_median.astype(np.float32) + 1)
            threshold = 1.0 - (self.flood_tolerance / 100.0)
            shadow_mask = (darkness_ratio < threshold).astype(np.uint8)
            
            # Morphological closing to bridge small gaps
            kernel_size = max(5, self.flood_tolerance // 5)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
            
            num_labels, labels = cv2.connectedComponents(shadow_mask)
            click_label = labels[y, x]
            
            if click_label > 0:
                filled_region = (labels == click_label).astype(np.uint8)
            else:
                filled_region = np.zeros((h, w), np.uint8)
        else:
            # LAB color mode: use edge detection + flood fill
            flood_mask = np.zeros((h + 2, w + 2), np.uint8)
            edges = cv2.Canny(gray, 30, 100)
            edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
            flood_mask[1:-1, 1:-1] = (edges > 0).astype(np.uint8)
            
            lab_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2LAB)
            lo_diff = (self.flood_tolerance, self.flood_tolerance // 2, self.flood_tolerance // 2)
            hi_diff = (self.flood_tolerance, self.flood_tolerance // 2, self.flood_tolerance // 2)
            lab_copy = lab_image.copy()
            
            cv2.floodFill(
                lab_copy,
                flood_mask,
                (x, y),
                (255, 128, 128),
                lo_diff,
                hi_diff,
                cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
            )
            
            filled_region = flood_mask[1:-1, 1:-1]
            filled_region = np.where(edges > 0, 0, filled_region).astype(np.uint8)
        
        self.current_mask = np.where(filled_region > 0, 0, self.current_mask).astype(np.uint8)

    def _create_display(self):
        display = self.current_image.copy()
        
        if self.show_mask:
            # Red overlay for mask
            mask_overlay = np.zeros_like(display)
            mask_overlay[:, :, 2] = self.current_mask * 200  # Red channel
            display = cv2.addWeighted(display, 0.7, mask_overlay, 0.3, 0)
            
            # Add contour
            contours, _ = cv2.findContours(self.current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, (0, 255, 0), 2)
        
        # Draw lasso path while drawing
        if self.drawing_lasso and len(self.lasso_points) > 1:
            # Draw the lasso path as a polyline
            pts = np.array(self.lasso_points, dtype=np.int32)
            lasso_color = (255, 255, 0) if self.mode == 1 else (0, 165, 255)  # Cyan for add, Orange for remove
            cv2.polylines(display, [pts], isClosed=False, color=lasso_color, thickness=2)
            # Draw dotted line back to start to show what will be closed
            if len(self.lasso_points) > 2:
                cv2.line(display, self.lasso_points[-1], self.lasso_points[0], lasso_color, 1, cv2.LINE_AA)
        
        # Apply zoom by cropping and resizing
        if self.zoom_level != 1.0:
            h, w = display.shape[:2]
            # Calculate visible area size
            view_w = int(w / self.zoom_level)
            view_h = int(h / self.zoom_level)
            
            # Allow mouse to reach all corners - don't clamp to center
            # Instead, let the view extend to edges
            x1 = int(self.mouse_x - view_w // 2)
            y1 = int(self.mouse_y - view_h // 2)
            
            # Clamp to image bounds
            x1 = max(0, min(w - view_w, x1))
            y1 = max(0, min(h - view_h, y1))
            x2 = x1 + view_w
            y2 = y1 + view_h
            
            cropped = display[y1:y2, x1:x2]
            
            # Resize back to original size
            display = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Help overlay
        if self.show_help:
            self._draw_help_overlay(display)
        
        # Draw brush cursor at mouse position (adjusted for zoom)
        cursor_color = (0, 255, 0) if self.mode == 1 else (0, 0, 255)
        # When zoomed, cursor stays at center-ish
        cursor_x = self.current_image.shape[1] // 2 if self.zoom_level > 1.0 else self.mouse_x
        cursor_y = self.current_image.shape[0] // 2 if self.zoom_level > 1.0 else self.mouse_y
        brush_display_size = int(self.brush_size * self.zoom_level)
        cv2.circle(display, (cursor_x, cursor_y), brush_display_size, cursor_color, 2)
        cv2.circle(display, (cursor_x, cursor_y), 2, cursor_color, -1)
        
        # Status bar at bottom
        h, w = display.shape[:2]
        cv2.rectangle(display, (0, h-30), (w, h), (40, 40, 40), -1)
        zoom_pct = int(self.zoom_level * 100)
        fill_mode = "BRIGHTNESS" if self.brightness_mode else "COLOR"
        status = f"Brush: {self.brush_size} | Tol: {self.flood_tolerance} | Zoom: {zoom_pct}% | Fill: {fill_mode} | 'h' = help"
        cv2.putText(display, status, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display
    
    def _draw_help_overlay(self, display):
        """Draw a semi-transparent help panel."""
        h, w = display.shape[:2]
        
        # Help panel dimensions
        panel_w, panel_h = 340, 320
        panel_x, panel_y = 10, 10
        
        # Semi-transparent background
        overlay = display.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, display, 0.15, 0, display)
        
        # Border
        cv2.rectangle(display, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (100, 100, 100), 2)
        
        # Title
        cv2.putText(display, "CONTROLS", (panel_x + 110, panel_y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Help text
        controls = [
            ("Left Click", "Brush ADD to mask"),
            ("Right Click", "Brush REMOVE from mask"),
            ("Ctrl + Drag", "LASSO select (Otsu threshold)"),
            ("Shift + Click", "Flood fill add/remove"),
            ("Scroll Wheel", "Zoom in/out"),
            ("", ""),
            ("s", "Save and go to next"),
            ("n", "Skip (don't save)"),
            ("q", "Quit review"),
            ("r", "Reset to original"),
            ("u", "Undo last change"),
            ("+/-", "Adjust brush size"),
            ("[/]", "Adjust flood tolerance"),
            ("0", "Reset zoom to 100%"),
            ("m", "Toggle mask visibility"),
            ("h", "Toggle this help"),
        ]
        
        y_offset = panel_y + 50
        for key, desc in controls:
            if key == "":
                y_offset += 5
                continue
            cv2.putText(display, key, (panel_x + 10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
            cv2.putText(display, desc, (panel_x + 100, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 18
    
    def review_mask(self, image_path: Path, mask_path: Path, review_path: Path = None, window_name: str = "Mask Review"):
        """
        Review and edit a single mask.
        
        Returns:
            'save': saved changes
            'skip': skipped without saving
            'quit': user wants to quit
        """
        # Load image and mask
        self.current_image = cv2.imread(str(image_path))
        if self.current_image is None:
            logger.error(f"Failed to load image: {image_path}")
            return 'skip'
        
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.error(f"Failed to load mask: {mask_path}")
            return 'skip'
        
        # Store original size
        orig_h, orig_w = mask.shape[:2]
        
        # Normalize mask to 0-1
        self.current_mask = (mask > 127).astype(np.uint8)
        self.original_mask = self.current_mask.copy()
        self.history = [self.current_mask.copy()]
        
        # Resize for display if too large
        h, w = self.current_image.shape[:2]
        self.scale = min(1920 / w, 1080 / h, 1.0)
        if self.scale < 1.0:
            new_w, new_h = int(w * self.scale), int(h * self.scale)
            self.current_image = cv2.resize(self.current_image, (new_w, new_h))
            self.current_mask = cv2.resize(self.current_mask, (new_w, new_h))
            self.original_mask = cv2.resize(self.original_mask, (new_w, new_h))
        
        # Create window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.current_image.shape[1], self.current_image.shape[0])
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        logger.info(f"Reviewing: {image_path.name}")
        
        result = 'skip'
        while True:
            display = self._create_display()
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                # Save and continue - resize back to original
                if self.scale < 1.0:
                    final_mask = cv2.resize(self.current_mask, (orig_w, orig_h))
                else:
                    final_mask = self.current_mask
                cv2.imwrite(str(mask_path), final_mask * 255)
                logger.info(f"Saved: {mask_path.name}")
                
                # Delete review image if it exists
                if review_path and review_path.exists():
                    review_path.unlink()
                    logger.info(f"Removed from review: {review_path.name}")
                
                result = 'save'
                break
            elif key == ord('n'):
                logger.info("Skipped")
                result = 'skip'
                break
            elif key == ord('q'):
                result = 'quit'
                break
            elif key == ord('r'):
                self.current_mask = self.original_mask.copy()
                self.history = [self.original_mask.copy()]
            elif key == ord('u') and len(self.history) > 1:
                self.history.pop()
                self.current_mask = self.history[-1].copy()
            elif key == ord('+') or key == ord('='):
                self.brush_size = min(100, self.brush_size + 5)
            elif key == ord('-'):
                self.brush_size = max(1, self.brush_size - 5)
            elif key == ord(']'):
                self.flood_tolerance = min(100, self.flood_tolerance + 5)
            elif key == ord('['):
                self.flood_tolerance = max(5, self.flood_tolerance - 5)
            elif key == ord('m'):
                self.show_mask = not self.show_mask
            elif key == ord('h'):
                self.show_help = not self.show_help
            elif key == ord('b'):
                self.brightness_mode = not self.brightness_mode
                mode_name = "BRIGHTNESS" if self.brightness_mode else "COLOR"
                logger.info(f"Fill mode: {mode_name}")
            elif key == ord('0'):
                # Reset zoom
                self.zoom_level = 1.0
                self.pan_x = 0
                self.pan_y = 0
        
        cv2.destroyWindow(window_name)
        return result


def main():
    parser = argparse.ArgumentParser(description="Batch mask review tool with click-to-segment")
    parser.add_argument("masks_dir", type=Path, help="Directory containing masks to review")
    parser.add_argument("--images-dir", type=Path, help="Directory containing original images")
    parser.add_argument("--review-only", action="store_true", help="Only review masks flagged for review")
    
    args = parser.parse_args()
    
    masks_dir = Path(args.masks_dir)
    
    # Find masks folder
    if (masks_dir / "masks").exists():
        actual_masks_dir = masks_dir / "masks"
        review_dir = masks_dir / "review"
    else:
        actual_masks_dir = masks_dir
        review_dir = masks_dir.parent / "review"
    
    # Find original images
    if args.images_dir:
        images_dir = args.images_dir
    else:
        possible_paths = [
            masks_dir.parent / "frames",
            masks_dir.parent.parent / "frames",
        ]
        images_dir = None
        for p in possible_paths:
            if p.exists():
                images_dir = p
                break
        if images_dir is None:
            logger.error("Could not find images directory. Please specify with --images-dir")
            return
    
    logger.info(f"Masks dir: {actual_masks_dir}")
    logger.info(f"Images dir: {images_dir}")
    
    # Get list of masks to review
    if args.review_only and review_dir.exists():
        review_files = list(review_dir.glob("review_*.jpg")) + list(review_dir.glob("review_*.png"))
        mask_stems = [f.stem.replace("review_", "") for f in review_files]
    else:
        mask_files = list(actual_masks_dir.glob("mask_*.png")) + list(actual_masks_dir.glob("*_mask.png"))
        mask_stems = [f.stem.replace("mask_", "").replace("_mask", "") for f in mask_files]
    
    logger.info(f"Found {len(mask_stems)} masks to review")
    
    reviewer = MaskReviewer()
    stats = {'saved': 0, 'skipped': 0, 'total': len(mask_stems)}
    
    for i, stem in enumerate(sorted(mask_stems)):
        logger.info(f"\n[{i+1}/{len(mask_stems)}] {stem}")
        
        # Find image
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        
        if image_path is None:
            logger.warning(f"Image not found for: {stem}")
            continue
        
        # Find mask
        mask_path = None
        for pattern in [f"mask_{stem}.png", f"{stem}_mask.png"]:
            candidate = actual_masks_dir / pattern
            if candidate.exists():
                mask_path = candidate
                break
        
        if mask_path is None:
            logger.warning(f"Mask not found for: {stem}")
            continue
        
        # Find review file (to delete after save)
        review_path = None
        if review_dir.exists():
            for ext in ['.jpg', '.png']:
                candidate = review_dir / f"review_{stem}{ext}"
                if candidate.exists():
                    review_path = candidate
                    break
        
        result = reviewer.review_mask(image_path, mask_path, review_path)
        
        if result == 'save':
            stats['saved'] += 1
        elif result == 'skip':
            stats['skipped'] += 1
        elif result == 'quit':
            break
    
    logger.info(f"\nReview complete: {stats['saved']} saved, {stats['skipped']} skipped out of {stats['total']}")


if __name__ == "__main__":
    main()
