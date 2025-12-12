#!/usr/bin/env python3
"""
Gallery Review Interface for mask quality control.

Features:
- Thumbnail grid gallery view
- Single image view with navigation
- Jump to any image by clicking thumbnail
- Accept/Reject workflow
- Modes: --flagged (only below-threshold) or --all (full review folder)

Controls:
  H         : Show/Hide this help
  ← → / A D : Navigate prev/next
  G         : Toggle gallery/single view
  B         : Toggle BRUSH mode (paint freehand - best for curved sticks)
  E         : Toggle ERASE mode (remove from mask)
  L         : Toggle LINE mode (draw straight lines)
  U         : Undo last change
  R         : Reset mask to original
  +/-       : Increase/Decrease brush size
  Enter/Space : Accept (remove from flagged)
  R         : Reject (mark for re-mask)
  Q / Esc   : Quit
  1-9       : Jump to image N in current page
  
Brush Mode (recommended for fisheye):
  Click and drag to paint along the stick shadow.
  Automatically follows curves - perfect for fisheye distortion!
  Press B again to exit.
  
Erase Mode:
  Click and drag to remove from mask.
  Press E again to exit.
  
Line Mode:
  Click and drag to draw a straight line.
  Press L again to exit.
"""

import sys
from pathlib import Path
from typing import List, Optional
import argparse
import cv2
import numpy as np


class GalleryReviewer:
    """Interactive gallery for reviewing mask quality."""
    
    def __init__(
        self,
        image_paths: List[Path],
        masks_dir: Optional[Path] = None,
        flagged_dir: Optional[Path] = None,
        window_name: str = "Mask Review Gallery"
    ):
        self.image_paths = list(image_paths)
        self.masks_dir = masks_dir
        self.flagged_dir = flagged_dir
        self.window_name = window_name
        
        self.current_idx = 0
        self.gallery_mode = False  # Start in single view (first frame)
        self.thumb_size = 300  # Increased from 200 for larger gallery
        self.cols = 5
        
        self.accepted = set()
        self.rejected = set()
        
        # Drawing state
        self.draw_mode = False  # Line mode
        self.brush_mode = False  # Brush/paint mode
        self.erase_mode = False  # Erase mode
        self.drawing = False
        self.draw_start = None
        self.draw_end = None
        self.temp_line = None
        self.brush_points = []  # For brush strokes
        self.brush_size = 12  # Default brush size
        self.show_help = True  # Help overlay visible by default (press 'h' to toggle)
        self.show_mask = True  # Toggle mask visibility with 'm'
        
        # Undo history (per-image)
        self.mask_history = {}  # {image_index: [mask_arrays]}
        self.current_mask_cache = None

        # Interactive Segmentation State (Ported from review_masks.py)
        self.flood_tolerance = 15
        self.lasso_points = []
        self.drawing_lasso = False
        self.mouse_x = 0
        self.mouse_y = 0
        
        if not self.image_paths:
            print("No images to review!")
            return
        
        # Initialize layout state
        self.pad_top = 0
        self.pad_left = 0
        self.view_img_h = 0
        self.view_img_w = 0
        
        # Internal render resolution (High Quality)
        # We render to this size, and Qt scales this to fit the actual window
        self.render_w = 1920
        self.render_h = 1920
        
        # Custom zoom/pan state
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
    
    def run(self):
        """Run the interactive review session."""
        if not self.image_paths:
            return
        
        # WINDOW_GUI_NORMAL = no toolbar, resizable
        cv2.namedWindow(self.window_name, cv2.WINDOW_GUI_NORMAL)
        cv2.moveWindow(self.window_name, 0, 0)
        cv2.resizeWindow(self.window_name, 1920, 1920)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        print("\n=== Review Gallery ===")
        print(f"Total images: {len(self.image_paths)}")
        print("Controls: ← → Navigate | G Gallery | S Save | Space Skip | Q Quit")
        print()
        
        while True:
            if self.gallery_mode:
                display = self._create_gallery_view()
            else:
                display = self._create_single_view()
            
            cv2.imshow(self.window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or Esc
                break
            elif key == ord('g'):  # Toggle gallery/single
                self.gallery_mode = not self.gallery_mode
            elif key == ord('h'):  # Toggle help
                self.show_help = not self.show_help
            elif key == ord('m'):  # Toggle mask visibility
                self.show_mask = not self.show_mask
                print(f"Mask visibility: {self.show_mask}")
            elif key == ord('e'):  # Toggle erase mode
                if not self.gallery_mode:
                    self.erase_mode = not self.erase_mode
                    if self.erase_mode:
                        self.brush_mode = False
                        self.draw_mode = False
                    self.brush_points = []
            elif key == ord('b'):  # Toggle brush mode
                if not self.gallery_mode:
                    self.brush_mode = not self.brush_mode
                    if self.brush_mode:
                        self.draw_mode = False
                        self.erase_mode = False
                    self.brush_points = []
            elif key == ord(']'): # Increase tolerance
                 self.flood_tolerance = min(100, self.flood_tolerance + 5)
                 print(f"Flood tolerance: {self.flood_tolerance}")
            elif key == ord('['): # Decrease tolerance
                 self.flood_tolerance = max(5, self.flood_tolerance - 5) 
                 print(f"Flood tolerance: {self.flood_tolerance}")
            elif key == ord('l'):  # Toggle line draw mode
                if not self.gallery_mode:
                    self.draw_mode = not self.draw_mode
                    if self.draw_mode:
                        self.brush_mode = False
                        self.erase_mode = False
                    self.temp_line = None
            elif key == ord('u'):  # Undo
                if not self.gallery_mode:
                    self._undo_last_change()
            elif key == ord('x'):  # Increase brush size
                self.brush_size = min(100, self.brush_size + 3)
                print(f"Brush size: {self.brush_size}px")
            elif key == ord('z'):  # Decrease brush size
                self.brush_size = max(3, self.brush_size - 3)
                print(f"Brush size: {self.brush_size}px")
            elif key == 81:  # Left arrow only
                self.current_idx = (self.current_idx - 1) % len(self.image_paths)
                self.current_mask_cache = None  # Reset cache for new image
            elif key == 83:  # Right arrow only
                self.current_idx = (self.current_idx + 1) % len(self.image_paths)
                self.current_mask_cache = None  # Reset cache for new image
            elif key == 32:  # Space - Skip (next without saving)
                self.current_idx = (self.current_idx + 1) % len(self.image_paths)
                self.current_mask_cache = None
                print(f"Skipped: moving to next")
            elif key == ord('s'):  # S - Save and go to next
                self._save_mask_to_disk()
                self._accept_current()
            elif key == ord('r'):  # Reject
                self._reject_current()
            elif ord('1') <= key <= ord('9'):  # Jump to N
                jump_idx = key - ord('1')
                page_start = (self.current_idx // (self.cols * 3)) * (self.cols * 3)
                target = page_start + jump_idx
                if target < len(self.image_paths):
                    self.current_idx = target
                    self.gallery_mode = False
        
        cv2.destroyWindow(self.window_name)
        
        # Print summary
        print("\n=== Review Complete ===")
        print(f"Accepted: {len(self.accepted)}")
        print(f"Rejected: {len(self.rejected)}")
        print(f"Remaining: {len(self.image_paths) - len(self.accepted) - len(self.rejected)}")
    
    def _skip_processed(self):
        """Skip to next unprocessed image."""
        start = self.current_idx
        while True:
            path = self.image_paths[self.current_idx]
            if path not in self.accepted and path not in self.rejected:
                break
            self.current_idx = (self.current_idx + 1) % len(self.image_paths)
            if self.current_idx == start:
                break  # All processed
    
    def _accept_current(self):
        """Accept current image (remove from flagged)."""
        # Save modifications before accepting
        if self.current_mask_cache is not None:
             self._save_mask_to_disk()

        path = self.image_paths[self.current_idx]
        self.accepted.add(path)
        
        # Remove symlink from flagged if it exists
        if self.flagged_dir and path.is_symlink():
            try:
                path.unlink()
                print(f"✓ Accepted: {path.name}")
            except:
                pass
        
        # Move to next
        self.current_idx = (self.current_idx + 1) % len(self.image_paths)
        self.current_mask_cache = None  # CRITICAL: Clear cache so new mask loads
        # Note: mask_history is a dict keyed by index, no need to clear entirely
        self._skip_processed()
    
    def _reject_current(self):
        """Reject current image (mark for re-mask)."""
        # Save modifications even if rejecting
        if self.current_mask_cache is not None:
             self._save_mask_to_disk()

        path = self.image_paths[self.current_idx]
        self.rejected.add(path)
        print(f"✗ Rejected: {path.name}")
        
        # Move to next
        self.current_idx = (self.current_idx + 1) % len(self.image_paths)
        self.current_mask_cache = None  # Clear cache so new mask loads
        # Note: mask_history is a dict keyed by index, no need to clear entirely
        self._skip_processed()
    
    def _create_gallery_view(self) -> np.ndarray:
        """Create thumbnail grid gallery view."""
        self.pad_top = 0
        self.pad_left = 0
        
        rows = 3
        cols = self.cols
        
        # Calculate page
        page_size = rows * cols
        page_start = (self.current_idx // page_size) * page_size
        
        # Create canvas
        canvas_h = rows * self.thumb_size + 60  # Extra for status bar
        canvas_w = cols * self.thumb_size
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:] = (40, 40, 40)  # Dark gray background
        
        # Draw thumbnails
        for i in range(page_size):
            idx = page_start + i
            if idx >= len(self.image_paths):
                break
            
            row = i // cols
            col = i % cols
            
            x = col * self.thumb_size
            y = row * self.thumb_size
            
            # Load and resize image
            path = self.image_paths[idx]
            img = cv2.imread(str(path))
            if img is None:
                continue
            
            thumb = cv2.resize(img, (self.thumb_size - 10, self.thumb_size - 10))
            
            # Add border based on status
            if path in self.accepted:
                border_color = (0, 255, 0)  # Green
            elif path in self.rejected:
                border_color = (0, 0, 255)  # Red
            elif idx == self.current_idx:
                border_color = (255, 255, 0)  # Cyan (selected)
            else:
                border_color = (100, 100, 100)  # Gray
            
            cv2.rectangle(thumb, (0, 0), (thumb.shape[1]-1, thumb.shape[0]-1), 
                         border_color, 3)
            
            # Add number label
            cv2.putText(thumb, str(i + 1), (5, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Place on canvas
            canvas[y+5:y+5+thumb.shape[0], x+5:x+5+thumb.shape[1]] = thumb
        
        # Status bar (simplified)
        status_y = rows * self.thumb_size + 40
        cv2.putText(canvas, 
                   f"Image {self.current_idx + 1}/{len(self.image_paths)} | G: Single view | Q: Quit",
                   (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        return canvas
    
    def _load_current_mask(self):
        """Load mask for current image into cache."""
        path = self.image_paths[self.current_idx]
        mask_path = self._get_mask_path(path)
        
        if mask_path and mask_path.exists():
            self.current_mask_cache = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # Create empty mask matching image size
            img = cv2.imread(str(path))
            if img is not None:
                self.current_mask_cache = np.zeros(img.shape[:2], dtype=np.uint8)
            else:
                self.current_mask_cache = None

    def _create_single_view(self) -> np.ndarray:
        """Create single image view matching original masking_v2 style."""
        path = self.image_paths[self.current_idx]
        img = cv2.imread(str(path))
        
        if img is None:
            img = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(img, "Could not load image", (100, 300),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.current_mask_cache = None
        
        # Load mask if not cached or size mismatch
        if self.current_mask_cache is None:
            self._load_current_mask()
        
        if self.current_mask_cache is not None and img is not None:
             if self.current_mask_cache.shape != img.shape[:2]:
                 # Resize mask if needed
                 self.current_mask_cache = cv2.resize(self.current_mask_cache, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Apply mask overlay to image (RED like review_masks.py)
        display = img.copy()
        if self.show_mask and self.current_mask_cache is not None:
            # Red overlay for mask (matching original)
            mask_overlay = np.zeros_like(display)
            mask_overlay[:, :, 2] = (self.current_mask_cache > 0).astype(np.uint8) * 200  # Red channel
            display = cv2.addWeighted(display, 0.7, mask_overlay, 0.3, 0)
            
            # Draw green contour around mask
            contours, _ = cv2.findContours(self.current_mask_cache, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, (0, 255, 0), 2)
        
        self.original_h, self.original_w = img.shape[:2]
        
        self.original_h, self.original_w = img.shape[:2]
        
        # Use internal render resolution
        render_w = getattr(self, 'render_w', 1920)
        render_h = getattr(self, 'render_h', 1920)
        
        # Calculate base scale to fit entire image in render buffer with correct aspect ratio
        h, w = img.shape[:2]
        self.base_scale = min(render_w / w, render_h / h)
        
        # Apply custom zoom
        effective_scale = self.base_scale * self.zoom_level
        
        # Viewport size in original coordinates
        view_w_orig = render_w / effective_scale
        view_h_orig = render_h / effective_scale
        
        # Pan handling
        # Ensure pan center is valid or default to center of image
        max_pan_x = max(0, w - view_w_orig)
        max_pan_y = max(0, h - view_h_orig)
        
        self.pan_x = max(0, min(self.pan_x, max_pan_x))
        self.pan_y = max(0, min(self.pan_y, max_pan_y))
        
        # Coordinates of top-left corner of viewport
        x1 = self.pan_x
        y1 = self.pan_y
        
        # Define source rectangle (float)
        # We need integer coordinates for slicing
        src_x = int(x1)
        src_y = int(y1)
        src_w = int(view_w_orig)
        src_h = int(view_h_orig)
        
        # Clamp to image boundaries safely
        # Note: If zoomed out far enough (zoom < 1.0), viewport might be larger than image
        # We handle this by creating a canvas and centering the image
        
        if effective_scale < self.base_scale: # Zoomed out further than fit
             # Just show fit-to-screen logic similar to original, or clamp zoom min
             pass 

        # Extract source region
        # Handle boundary conditions by padding if necessary (or just clamping)
        # For simplicity, we clamp viewport to be within image if zoom >= 1 (vs fit)
        
        # Let's refine: x2, y2
        src_x2 = min(src_x + src_w, w)
        src_y2 = min(src_y + src_h, h)
        src_x = max(0, src_x)
        src_y = max(0, src_y)
        
        viewport = display[src_y:src_y2, src_x:src_x2].copy()
        
        # Resize viewport to fill render buffer (preserving aspect if we want? No, fill buffer)
        # Actually, we want to maintain aspect ratio of the SOURCE pixels.
        # If we resize viewport (w,h) to render (W,H), we might stretch.
        # We should ensure view_w_orig / view_h_orig matches render_w / render_h.
        # Since we calculated view_w/h from render_w/h, aspect ratio is preserved automatically!
        
        if viewport.size > 0:
            display = cv2.resize(viewport, (render_w, render_h), interpolation=cv2.INTER_LINEAR)
        else:
            display = np.zeros((render_h, render_w, 3), dtype=np.uint8)

        # Store transform info for mouse coordinate conversion
        self.display_scale = effective_scale
        self.viewport_x = src_x
        self.viewport_y = src_y
        # Note: display_scale here is (render_w / src_w). 
        # Wait, effective_scale = render_w / view_w_orig. Yes.
        # But since we cast to int for src_w, there's a tiny discretization error. 
        # Better to recompute actual scale from the extracted crop.
        if viewport.shape[1] > 0:
             self.display_scale = render_w / viewport.shape[1]
        
        self.view_img_h, self.view_img_w = render_h, render_w
        self.pad_top = 0
        self.content_offset_x = 0
        self.content_offset_y = 0
        
        # Draw active line preview
        if self.drawing and self.draw_mode and self.temp_line:
            cv2.line(display, self.temp_line[0], self.temp_line[1], (0, 255, 255), 3)

        # Draw Lasso
        if self.drawing_lasso and len(self.lasso_points) > 1:
            pts = np.array(self.lasso_points, dtype=np.int32)
            lasso_color = (255, 255, 0) if self.mode == 1 else (0, 165, 255)
            cv2.polylines(display, [pts], isClosed=False, color=lasso_color, thickness=2)
            # Preview closure
            if len(self.lasso_points) > 2:
                cv2.line(display, self.lasso_points[-1], self.lasso_points[0], lasso_color, 1, cv2.LINE_AA)

        # --- Overlay UI ---
        
        # Helper for shadow text
        def draw_text(img, text, pos, scale=0.6, color=(255, 255, 255)):
            cv2.putText(img, text, (pos[0]+1, pos[1]+1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 2)
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)

        # Footer bar with all info
        h, w = display.shape[:2]
        cv2.rectangle(display, (0, h-45), (w, h), (30, 30, 30), -1)  # Taller dark footer
        
        # Left: Filename only
        footer_left = f"{path.name} ({self.current_idx + 1}/{len(self.image_paths)})"
        # Shadow + white text for readability
        cv2.putText(display, footer_left, (12, h-14), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 2)
        cv2.putText(display, footer_left, (10, h-15), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)
        
        # Right: Status, Brush, and tolerance
        status = "PENDING"
        status_col = (200, 200, 200)
        if path in self.accepted:
            status = "ACCEPTED"; status_col = (0, 255, 0)
        elif path in self.rejected:
            status = "REJECTED"; status_col = (0, 0, 255)
        
        footer_right = f"{status}  |  Brush: {self.brush_size}  |  Tol: {self.flood_tolerance}"
        text_x = w - 400
        cv2.putText(display, footer_right, (text_x+2, h-14), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 2)
        cv2.putText(display, footer_right, (text_x, h-15), cv2.FONT_HERSHEY_DUPLEX, 0.65, status_col, 1)
        
        # Help Overlay Panel
        if self.show_help:
            self._draw_help_overlay(display)
        
        # Draw Brush Cursor at mouse position
        cursor_color = (0, 255, 0) if getattr(self, 'mode', 1) == 1 else (0, 0, 255)
        brush_display_radius = int(self.brush_size / 2)  # In display coords
        cv2.circle(display, (self.mouse_x, self.mouse_y), brush_display_radius, cursor_color, 2)
        cv2.circle(display, (self.mouse_x, self.mouse_y), 2, cursor_color, -1)
        
        return display

    def _apply_brush(self, x, y, is_erase):
        """Apply brush to current mask at x,y."""
        if self.current_mask_cache is None: return
        
        # Convert display (x,y) to original image (x,y) accounting for zoom/pan viewport
        viewport_x = getattr(self, 'viewport_x', 0)
        viewport_y = getattr(self, 'viewport_y', 0)
        
        ox = int(x / self.display_scale) + viewport_x
        oy = int(y / self.display_scale) + viewport_y
        
        # Clamp to image bounds
        ox = max(0, min(ox, self.original_w - 1))
        oy = max(0, min(oy, self.original_h - 1))
        
        color = 0 if is_erase else 255
        radius = int(self.brush_size / self.display_scale / 2)
        if radius < 1: radius = 1
        
        cv2.circle(self.current_mask_cache, (ox, oy), radius, color, -1)
    
    def _draw_help_overlay(self, display):
        """Draw a semi-transparent help panel."""
        h, w = display.shape[:2]
        
        # Help panel dimensions - wider for long key names
        panel_w, panel_h = 420, 400
        panel_x, panel_y = 10, 40
        
        # Semi-transparent background
        overlay = display.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, display, 0.15, 0, display)
        
        # Border
        cv2.rectangle(display, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (100, 100, 100), 2)
        
        # Title
        cv2.putText(display, "CONTROLS", (panel_x + 120, panel_y + 25), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1)
        
        # Help text
        controls = [
            ("Left Click", "Brush ADD to mask"),
            ("Right Click", "Brush REMOVE from mask"),
            ("Ctrl + Drag", "LASSO select (Otsu threshold)"),
            ("Shift + Click", "Flood fill add/remove"),
            ("", ""),
            ("S", "Save"),
            ("Space", "Skip"),
            ("Arrow keys", "Prev / Next"),
            ("G", "Toggle gallery view"),
            ("Q / Esc", "Quit"),
            ("", ""),
            ("Z / X", "Decrease / Increase brush"),
            ("[/]", "Adjust flood tolerance"),
            ("m", "Toggle mask visibility"),
            ("u", "Undo last change"),
            ("h", "Toggle this help"),
        ]
        
        y_offset = panel_y + 50
        for key, desc in controls:
            if key == "":
                y_offset += 10
                continue
            # Shadow for readability
            cv2.putText(display, key, (panel_x + 17, y_offset+1), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 2)
            cv2.putText(display, key, (panel_x + 15, y_offset), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (100, 255, 100), 1)
            cv2.putText(display, desc, (panel_x + 152, y_offset+1), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 2)
            cv2.putText(display, desc, (panel_x + 150, y_offset), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (230, 230, 230), 1)
            y_offset += 24
        
    def _apply_line_preview(self, start, end):
        """Apply line to current mask."""
        if self.current_mask_cache is None: return
        
        x1 = int(start[0] / self.display_scale)
        y1 = int(start[1] / self.display_scale)
        x2 = int(end[0] / self.display_scale)
        y2 = int(end[1] / self.display_scale)
        
        thickness = int(self.brush_size / self.display_scale)
        cv2.line(self.current_mask_cache, (x1, y1), (x2, y2), 255, thickness)

    def _apply_lasso_selection(self):
        """Apply Otsu thresholding within the drawn lasso polygon."""
        if len(self.lasso_points) < 3:
            return
        
        # Load current image for processing
        path = self.image_paths[self.current_idx]
        img = cv2.imread(str(path))
        if img is None: return

        h, w = img.shape[:2]
        
        # Convert lasso points from Display -> Image coordinates
        scale = self.display_scale
        image_polygon = []
        for p in self.lasso_points:
            ix = int(p[0] / scale)
            iy = int(p[1] / scale)
            image_polygon.append((ix, iy))
        
        polygon = np.array(image_polygon, dtype=np.int32)
        lasso_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(lasso_mask, [polygon], 1)
        
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Extract pixels within the lasso region
        lasso_pixels = gray[lasso_mask > 0]
        
        if len(lasso_pixels) < 10:
            return
        
        # Use Otsu's method to find optimal threshold between dark (shadow) and light (surface)
        otsu_threshold, _ = cv2.threshold(lasso_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Adjust threshold based on tolerance
        adjustment = (self.flood_tolerance - 50) * 0.4
        threshold = otsu_threshold + adjustment
        
        # Create selection: pixels darker than threshold within the lasso
        dark_pixels = ((gray < threshold) & (lasso_mask > 0)).astype(np.uint8)
        
        # Morphological opening to remove small isolated spots
        kernel = np.ones((5, 5), np.uint8)
        dark_pixels = cv2.morphologyEx(dark_pixels, cv2.MORPH_OPEN, kernel)
        
        # Keep only the largest connected component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_pixels)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            dark_pixels = (labels == largest_label).astype(np.uint8)
        
        # Apply to mask
        if self.current_mask_cache is None:
             self.current_mask_cache = np.zeros((h, w), dtype=np.uint8)

        if self.mode == 1:  # Add
            self.current_mask_cache = np.maximum(self.current_mask_cache, dark_pixels)
        else:  # Remove
            self.current_mask_cache = np.where(dark_pixels > 0, 0, self.current_mask_cache).astype(np.uint8)
        
        print(f"Lasso: Otsu={otsu_threshold:.0f}, Adj={threshold:.0f}, Px={dark_pixels.sum()}")

    def _flood_fill_op(self, x, y, is_add):
        """Flood fill helper."""
        path = self.image_paths[self.current_idx]
        img = cv2.imread(str(path))
        if img is None: return

        if self.current_mask_cache is None:
             self.current_mask_cache = np.zeros(img.shape[:2], dtype=np.uint8)
             
        h, w = img.shape[:2]
        
        # Map coordinates
        ox = int(x / self.display_scale)
        oy = int(y / self.display_scale)
        
        if not (0 <= ox < w and 0 <= oy < h):
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # LAB color mode with edge detection
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        edges = cv2.Canny(gray, 30, 100)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        flood_mask[1:-1, 1:-1] = (edges > 0).astype(np.uint8)
        
        lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lo_diff = (self.flood_tolerance, self.flood_tolerance // 2, self.flood_tolerance // 2)
        hi_diff = (self.flood_tolerance, self.flood_tolerance // 2, self.flood_tolerance // 2)
        lab_copy = lab_image.copy()
        
        cv2.floodFill(
            lab_copy,
            flood_mask,
            (ox, oy),
            (255, 128, 128),
            lo_diff,
            hi_diff,
            cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
        )
        
        filled_region = flood_mask[1:-1, 1:-1]
        filled_region = np.where(edges > 0, 0, filled_region).astype(np.uint8)
            
        if is_add:
            self.current_mask_cache = np.maximum(self.current_mask_cache, filled_region)
        else:
            self.current_mask_cache = np.where(filled_region > 0, 0, self.current_mask_cache).astype(np.uint8)

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks."""
        self.mouse_x, self.mouse_y = x, y # Track for zoom/help

        if self.gallery_mode:
            # Gallery view
            if event == cv2.EVENT_LBUTTONDOWN:
                col = x // self.thumb_size
                row = y // self.thumb_size
                
                if row < 3:
                    page_size = 3 * self.cols
                    page_start = (self.current_idx // page_size) * page_size
                    clicked_idx = page_start + row * self.cols + col
                    
                    if clicked_idx < len(self.image_paths):
                        self.current_idx = clicked_idx
                        self.current_mask_cache = None
                        self.gallery_mode = False
        
        else:
            # Single view - Interactive Tools
            
            # 1. Lasso Selection (Ctrl + Drag)
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.drawing_lasso = True
                    self.lasso_points = [(x, y)]
                    self.mode = 1 # Add
                    return
                elif event == cv2.EVENT_RBUTTONDOWN:
                    self.drawing_lasso = True
                    self.lasso_points = [(x, y)]
                    self.mode = 0 # Remove
                    return
                elif event == cv2.EVENT_MOUSEMOVE and self.drawing_lasso:
                    self.lasso_points.append((x, y))
                    return
                elif (event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP) and self.drawing_lasso:
                    if len(self.lasso_points) > 2:
                        # Save history before apply
                        if self.current_mask_cache is not None:
                             self._save_to_history(self.current_mask_cache)
                        self._apply_lasso_selection()
                    self.drawing_lasso = False
                    self.lasso_points = []
                    return
            
            # Cancel lasso if ctrl released
            if self.drawing_lasso and not (flags & cv2.EVENT_FLAG_CTRLKEY):
                self.drawing_lasso = False
                self.lasso_points = []
            
            # Middle mouse button for panning
            if event == cv2.EVENT_MBUTTONDOWN:
                self.panning = True
                self.pan_start_x = x
                self.pan_start_y = y
                return
            elif event == cv2.EVENT_MBUTTONUP:
                self.panning = False
                return
            elif event == cv2.EVENT_MOUSEMOVE and self.panning:
                # Calculate pan delta in original image coords
                dx = int((self.pan_start_x - x) / self.display_scale)
                dy = int((self.pan_start_y - y) / self.display_scale)
                self.pan_x += dx
                self.pan_y += dy
                self.pan_start_x = x
                self.pan_start_y = y
                return
            
            # Scroll wheel handling
            if event == cv2.EVENT_MOUSEWHEEL:
                # Extract scroll delta
                try:
                    delta = cv2.getMouseWheelDelta(flags)
                except:
                    delta = flags >> 16
                    if delta > 32767:
                        delta = delta - 65536
                
                if flags & cv2.EVENT_FLAG_SHIFTKEY:
                    # Shift+Scroll = Brush size adjustment
                    if delta > 0:
                        self.brush_size = min(100, self.brush_size + 3)
                    elif delta < 0:
                        self.brush_size = max(3, self.brush_size - 3)
                else:
                    # Plain scroll = Custom zoom (centered on mouse)
                    old_zoom = self.zoom_level
                    
                    if delta > 0:
                        self.zoom_level = min(10.0, self.zoom_level * 1.2)
                    elif delta < 0:
                        self.zoom_level = max(0.5, self.zoom_level / 1.2)
                    
                    # Adjust pan to keep mouse position stable (zoom toward cursor)
                    if old_zoom != self.zoom_level:
                        # 1. Calculate pointer position in original image coordinates BEFORE zoom
                        # Mouse x,y are in render coords (0..1920)
                        # orig_x = (mouse_x / old_scale) + old_pan_x
                        old_scale = getattr(self, 'display_scale', 1.0)
                        old_pan_x = getattr(self, 'viewport_x', 0)
                        old_pan_y = getattr(self, 'viewport_y', 0)
                        
                        ptr_x_orig = (x / old_scale) + old_pan_x
                        ptr_y_orig = (y / old_scale) + old_pan_y
                        
                        # 2. Calculate new scale
                        base_scale = getattr(self, 'base_scale', 1.0)
                        new_scale = base_scale * self.zoom_level
                        
                        # 3. We want: (x / new_scale) + new_pan_x = ptr_x_orig
                        # => new_pan_x = ptr_x_orig - (x / new_scale)
                        
                        self.pan_x = ptr_x_orig - (x / new_scale)
                        self.pan_y = ptr_y_orig - (y / new_scale)
                        
                return

            # 2. Flood Fill (Shift + Click)
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                if event == cv2.EVENT_LBUTTONDOWN:
                    if self.current_mask_cache is not None:
                             self._save_to_history(self.current_mask_cache)
                    self._flood_fill_op(x, y, is_add=True)
                    return
                elif event == cv2.EVENT_RBUTTONDOWN:
                    if self.current_mask_cache is not None:
                             self._save_to_history(self.current_mask_cache)
                    self._flood_fill_op(x, y, is_add=False)
                    return

            # 3. Standard Brush
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.mode = 1  # Add
                self.brush_points = [(x, y)] # Keep for line mode start point
                
                # Save state for Undo BEFORE modifying
                if self.current_mask_cache is not None:
                    self._save_to_history(self.current_mask_cache)
                    
                if self.draw_mode:
                    self.draw_start = (x, y)
                    self.temp_line = None
                else:
                    # Immediate paint
                    self._apply_brush(x, y, is_erase=False)
            
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.drawing = True
                self.mode = 0  # Remove
                
                if self.current_mask_cache is not None:
                    self._save_to_history(self.current_mask_cache)
                    
                # Immediate erase
                self._apply_brush(x, y, is_erase=True)
            
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                if self.draw_mode and self.mode == 1: 
                     self.temp_line = (self.draw_start, (x, y))
                else:
                    # Continuous paint
                    self._apply_brush(x, y, is_erase=(self.mode == 0))
            
            elif event in [cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP]:
                if self.drawing:
                    self.drawing = False
                    
                    if self.draw_mode and self.mode == 1 and self.draw_start:
                        # Commit line
                        self._apply_line_preview(self.draw_start, (x, y))
                        self.draw_start = None
                        self.temp_line = None
                    
                    # Auto-save disabled as per user request.
                    # self._save_mask_to_disk()
    
    def _save_mask_to_disk(self):
        """Save current cached mask to disk."""
        if self.masks_dir and self.current_mask_cache is not None:
            path = self.image_paths[self.current_idx]
            mask_path = self._get_mask_path(path)
            cv2.imwrite(str(mask_path), self.current_mask_cache)
            # print(f"Saved to {mask_path.name}") # Optional log to avoid spam
    
    def _undo_last_change(self):
        """Undo the last change."""
        if self.current_idx not in self.mask_history:
            print("No undo history")
            return
        
        history = self.mask_history[self.current_idx]
        if not history: return
        
        # Restore previous state
        previous_mask = history.pop()
        self.current_mask_cache = previous_mask.copy() # Update cache
        self._save_mask_to_disk()
        print(f"Undone ({len(history)} states left)")
    
    def _get_mask_path(self, review_path):
        """Get the mask file path for a given review image."""
        if not self.masks_dir:
            return None
        
        mask_filename = review_path.name.replace('review_', 'mask_')
        mask_filename = mask_filename.replace('.jpg', '.png').replace('.JPG', '.png')
        mask_path = self.masks_dir / mask_filename
        
        if not mask_path.exists():
            # Try recursive search if masks_dir has subfolders? 
            # Or just standard structure
            if (self.masks_dir / mask_filename).exists():
                return self.masks_dir / mask_filename
            
            # Legacy/Alternative naming?
            frame_name = review_path.name.replace('review_', '').replace('_debug.jpg', '')
            alt_filename = f"{frame_name}_hybrid_mask.png"
            if (self.masks_dir / alt_filename).exists():
                return self.masks_dir / alt_filename
                
        return self.masks_dir / mask_filename
    
    def _save_to_history(self, mask):
        """Save current mask state to history."""
        if self.current_idx not in self.mask_history:
            self.mask_history[self.current_idx] = []
        
        history = self.mask_history[self.current_idx]
        # Store COPY
        history.append(mask.copy())
        
        if len(history) > 20:
            history.pop(0)



def main():
    parser = argparse.ArgumentParser(
        description="Gallery review interface for mask quality control"
    )
    parser.add_argument('review_dir', type=Path,
                        help='Directory containing review images')
    parser.add_argument('--flagged', action='store_true',
                        help='Only show flagged images (from flagged/ subfolder)')
    parser.add_argument('--all', action='store_true',
                        help='Show all images in review folder')
    
    args = parser.parse_args()
    
    review_dir = Path(args.review_dir)
    
    if args.flagged:
        # Look for flagged subfolder
        flagged_dir = review_dir / 'flagged'
        if not flagged_dir.exists():
            flagged_dir = review_dir  # Use as-is if no subfolder
    else:
        flagged_dir = None
    
    # Find images
    search_dir = flagged_dir if args.flagged else (review_dir / 'review' if (review_dir / 'review').exists() else review_dir)
    
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_paths.extend(search_dir.glob(ext))
    
    image_paths = sorted(image_paths)
    
    if not image_paths:
        print(f"No images found in {search_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_paths)} images in {search_dir}")
    
    # Determine masks directory (sibling to review dir)
    if (review_dir / 'masks').exists():
        masks_dir = review_dir / 'masks'
    else:
        masks_dir = review_dir.parent / 'masks' if review_dir.name == 'review' else None
    
    if masks_dir and not masks_dir.exists():
        masks_dir = None
    
    reviewer = GalleryReviewer(
        image_paths,
        masks_dir=masks_dir,
        flagged_dir=flagged_dir if args.flagged else None
    )
    reviewer.run()


if __name__ == '__main__':
    main()
