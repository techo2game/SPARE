"""
PyVista Visualization Library for Anomaly Detection
"""

import numpy as np
import torch
import os

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("Warning: PyVista not available. Install with: pip install pyvista")


class PyVistaAnomalyVisualizer:
    """
    Interactive visualizer for anomaly detection results on point clouds.
    Provides real-time interactive point picking with instant value display
    and interactive colormap switching.
    """
    def __init__(self,
                 colormap='jet',
                 interactive=True,
                 save_screenshots=False,
                 output_dir=None,
                 point_size=9.0,
                 show_scalar_bar=True,
                 window_size=(1200, 900),
                 show_boundary=True,
                 show_reference_frame=True,
                 camera_position=None):
        """
        Initialize the PyVista anomaly visualizer.

        Args:
            colormap: Initial colormap for visualization (default: 'jet')
            interactive: Show interactive visualization (default: True)
            save_screenshots: Save visualization screenshots (default: False)
            output_dir: Directory to save screenshots (default: None)
            point_size: Size of points in visualization (default: 9.0)
            show_scalar_bar: Show color scale bar (default: True)
            window_size: Window size tuple (width, height) (default: (1200, 900))
            show_boundary: Show boundary box (default: True)
            show_reference_frame: Show XYZ reference axes (default: True)
            camera_position: Camera position (x, y, z) tuple/list or 'iso'/'xy'/'xz'/'yz' (default: None -> 'iso')
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista is required. Install with: pip install pyvista")

        # Initialize scaling variables
        self.current_scale = 1.0
        self.scale_increment = 1.15
        self.initial_colormap = colormap
        self.interactive = interactive
        self.save_screenshots = save_screenshots
        self.output_dir = output_dir
        self.point_size = point_size
        self.show_scalar_bar = show_scalar_bar
        self.window_size = window_size
        self.show_boundary = show_boundary
        self.show_reference_frame = show_reference_frame
        self.camera_position = camera_position if camera_position is not None else 'iso'

        # Available colormaps - organized by category for easier navigation
        self.colormaps = {
            # Perceptually uniform
            'viridis': 'viridis',
            'plasma': 'plasma',
            'inferno': 'inferno',
            'magma': 'magma',
            'cividis': 'cividis',

            # Sequential
            'jet': 'jet',
            'turbo': 'turbo',
            'hot': 'hot',
            'cool': 'cool',
            'spring': 'spring',
            'summer': 'summer',
            'autumn': 'autumn',
            'winter': 'winter',
            'bone': 'bone',
            'copper': 'copper',
            'gray': 'gray',

            # Diverging
            'coolwarm': 'coolwarm',
            'bwr': 'bwr',
            'seismic': 'seismic',
            'RdYlBu': 'RdYlBu',
            'RdBu': 'RdBu',
            'spectral': 'spectral',

            # Qualitative
            'hsv': 'hsv',
            'rainbow': 'rainbow',
            'tab10': 'tab10',
            'tab20': 'tab20',
        }

        self.colormap_names = list(self.colormaps.keys())
        self.current_colormap_idx = 0

        # Set initial colormap index
        if colormap in self.colormap_names:
            self.current_colormap_idx = self.colormap_names.index(colormap)

        self.current_colormap = self.colormap_names[self.current_colormap_idx]

        if self.save_screenshots and self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

    def create_border_reference_frame(self, points):
        """
        Create border points at data boundaries and orthonormal reference frame.

        Args:
            points: Point cloud numpy array (N, 3)

        Returns:
            border_cloud, x_axis, y_axis, z_axis: PyVista objects
        """
        # Compute bounding box
        min_bounds = points.min(axis=0)
        max_bounds = points.max(axis=0)

        # Extend bounds slightly for visualization
        margin = 0.1 * (max_bounds - min_bounds)
        min_bounds -= margin
        max_bounds += margin

        # Create border points on all 6 faces of the bounding box
        border_points = []
        n_points_per_face = 20  # Adjust density as needed

        # Generate coordinates for each axis
        x_coords = np.linspace(min_bounds[0], max_bounds[0], n_points_per_face)
        y_coords = np.linspace(min_bounds[1], max_bounds[1], n_points_per_face)
        z_coords = np.linspace(min_bounds[2], max_bounds[2], n_points_per_face)

        # XY faces (z = min/max)
        for z in [min_bounds[2], max_bounds[2]]:
            for x in x_coords:
                for y in y_coords:
                    border_points.append([x, y, z])

        # XZ faces (y = min/max)
        for y in [min_bounds[1], max_bounds[1]]:
            for x in x_coords:
                for z in z_coords:
                    if not (np.isclose(z, min_bounds[2]) or np.isclose(z, max_bounds[2])):
                        border_points.append([x, y, z])

        # YZ faces (x = min/max)
        for x in [min_bounds[0], max_bounds[0]]:
            for y in y_coords:
                for z in z_coords:
                    if not ((np.isclose(y, min_bounds[1]) or np.isclose(y, max_bounds[1])) or
                            (np.isclose(z, min_bounds[2]) or np.isclose(z, max_bounds[2]))):
                        border_points.append([x, y, z])

        border_cloud = pv.PolyData(np.array(border_points))

        # Create orthonormal reference frame (axes) at centroid
        centroid = points.mean(axis=0)
        axis_length = np.linalg.norm(max_bounds - min_bounds) * 0.3

        x_axis = pv.Line(centroid, centroid + [axis_length, 0, 0])
        y_axis = pv.Line(centroid, centroid + [0, axis_length, 0])
        z_axis = pv.Line(centroid, centroid + [0, 0, axis_length])

        return border_cloud, x_axis, y_axis, z_axis

    def visualize(self, points, anomaly_scores, sample_name="Sample", gt_labels=None):
        """
        Visualize point cloud with anomaly scores.

        Args:
            points: Point cloud numpy array (N, 3)
            anomaly_scores: Anomaly score per point (N,)
            sample_name: Name/identifier for this sample (default: "Sample")
            gt_labels: Optional ground truth labels for comparison (N,)
        """
        # Ensure numpy arrays
        if not isinstance(points, np.ndarray):
            points = np.asarray(points)
        if not isinstance(anomaly_scores, np.ndarray):
            anomaly_scores = np.asarray(anomaly_scores)

        # Store original scores for value inspection
        original_scores = anomaly_scores.copy()

        # Create PyVista point cloud
        cloud = pv.PolyData(points)

        # Add original anomaly scores as scalar data
        cloud['original_score'] = original_scores

        # Add normalized scores for better color visualization
        scores_normalized = (original_scores - original_scores.min()) / (
                original_scores.max() - original_scores.min() + 1e-6)
        cloud['normalized_score'] = scores_normalized
        cloud['scaled_score'] = scores_normalized.copy()  # For real-time scaling

        # Add ground truth if provided
        if gt_labels is not None:
            if not isinstance(gt_labels, np.ndarray):
                gt_labels = np.asarray(gt_labels)
            cloud['gt_label'] = gt_labels

        # Set the active scalars for coloring
        cloud.set_active_scalars('scaled_score')

        if self.interactive:
            self._show_interactive_visualization(cloud, sample_name, original_scores)

        if self.save_screenshots and self.output_dir is not None:
            self._save_screenshot(cloud, sample_name)

    def _show_interactive_visualization(self, cloud, sample_name, original_scores):
        """Show interactive PyVista visualization with point picking and colormap switching."""

        # Create plotter
        plotter = pv.Plotter(window_size=self.window_size,
                             title=f"{sample_name} - Interactive Colormap & Point Picking")

        # Add point cloud to plotter
        actor = plotter.add_mesh(
            cloud,
            cmap=self.current_colormap,
            point_size=self.point_size,
            render_points_as_spheres=True,
            scalars='scaled_score',
            scalar_bar_args={'title': 'Scaled Anomaly Score (0-1)'}
        )

        # Create reference frame data (always create, but conditionally display)
        centroid = cloud.points.mean(axis=0)
        min_bounds = cloud.points.min(axis=0)
        max_bounds = cloud.points.max(axis=0)
        axis_length = np.linalg.norm(max_bounds - min_bounds) * 0.3

        # Store reference frame actors for toggling (lines and labels)
        ref_frame_actors = {'x_axis': None, 'y_axis': None, 'z_axis': None,
                           'x_label': None, 'y_label': None, 'z_label': None}
        reference_frame_visible = self.show_reference_frame

        # Helper function to add reference frame
        def add_reference_frame():
            # Create and add axis lines
            x_axis = pv.Line(centroid, centroid + [axis_length, 0, 0])
            y_axis = pv.Line(centroid, centroid + [0, axis_length, 0])
            z_axis = pv.Line(centroid, centroid + [0, 0, axis_length])

            ref_frame_actors['x_axis'] = plotter.add_mesh(x_axis, color='red', line_width=3, name='x_axis')
            ref_frame_actors['y_axis'] = plotter.add_mesh(y_axis, color='green', line_width=3, name='y_axis')
            ref_frame_actors['z_axis'] = plotter.add_mesh(z_axis, color='blue', line_width=3, name='z_axis')

            # Add text labels with show_points=False to hide the colored point markers
            ref_frame_actors['x_label'] = plotter.add_point_labels(
                centroid + [axis_length * 1.1, 0, 0], ['X'],
                font_size=20, show_points=False, text_color='red')
            ref_frame_actors['y_label'] = plotter.add_point_labels(
                centroid + [0, axis_length * 1.1, 0], ['Y'],
                font_size=20, show_points=False, text_color='green')
            ref_frame_actors['z_label'] = plotter.add_point_labels(
                centroid + [0, 0, axis_length * 1.1], ['Z'],
                font_size=20, show_points=False, text_color='blue')

        # Helper function to remove reference frame
        def remove_reference_frame():
            for actor_name, actor in ref_frame_actors.items():
                if actor is not None:
                    try:
                        # Handle both single actors and tuples (actor, mapper) from add_point_labels
                        if isinstance(actor, tuple):
                            for a in actor:
                                if a is not None:
                                    plotter.remove_actor(a)
                        else:
                            plotter.remove_actor(actor)
                    except Exception as e:
                        print(f"Warning: Could not remove {actor_name}: {e}")
                    ref_frame_actors[actor_name] = None

        # Add reference frame if enabled
        if self.show_reference_frame:
            add_reference_frame()

        # Add scalar bar if requested
        scalar_bar = None
        if self.show_scalar_bar:
            scalar_bar = plotter.add_scalar_bar(
                title="Anomaly Score",
                n_labels=5,
                italic=False,
                fmt="%.3f"
            )

        # Helper function to generate info text
        def generate_info_text():
            has_gt = 'gt_label' in cloud.array_names
            gt_info = ""
            if has_gt:
                gt_labels = cloud['gt_label']
                n_anomalous = np.sum(gt_labels > 0)
                gt_info = f"\nGT Anomalies: {n_anomalous}/{len(gt_labels)} points"

            return (f"{sample_name}\n"
                    f"Score range: {original_scores.min():.6f} to {original_scores.max():.6f}\n"
                    f"Points: {len(original_scores)}{gt_info}\n"
                    f"Colormap: {self.current_colormap}")

        # Add initial info text
        info_actor = plotter.add_text(generate_info_text(), position='upper_left', font_size=12)

        # Add camera position display in top-right
        def get_camera_info():
            cam_pos = plotter.camera.position
            focal_point = plotter.camera.focal_point
            return (f"Camera Position:\n"
                    f"Pos: ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f})\n"
                    f"Focus: ({focal_point[0]:.2f}, {focal_point[1]:.2f}, {focal_point[2]:.2f})")

        # Set camera position (use custom or default 'iso')
        plotter.camera_position = self.camera_position

        # Add initial camera info
        camera_info_actor = plotter.add_text(get_camera_info(), position='lower_right', font_size=10, color='black')

        # Camera position update callback
        def update_camera_info():
            """Update camera position display when camera moves."""
            nonlocal camera_info_actor
            plotter.remove_actor(camera_info_actor)
            camera_info_actor = plotter.add_text(get_camera_info(), position='lower_right', font_size=10, color='black')

        # Add camera interaction callback
        plotter.iren.add_observer('EndInteractionEvent', lambda obj, event: update_camera_info())

        # Point picking callback
        def point_pick_callback(mesh, point_id):
            if hasattr(point_id, '__len__') and len(point_id) > 0:
                point_id = point_id[0]

            try:
                point_id = int(point_id)
                if 0 <= point_id < len(original_scores):
                    coord = cloud.points[point_id]
                    original_val = original_scores[point_id]
                    normalized_val = cloud['normalized_score'][point_id]
                    scaled_val = cloud['scaled_score'][point_id]

                    print(f">>> Picked Point {point_id}: coord=({coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f})")
                    print(f"    original={original_val:.6f}, normalized={normalized_val:.3f}, scaled={scaled_val:.3f}")

                    pick_text = (f"Last Picked Point:\n"
                                 f"ID: {point_id}\n"
                                 f"Coord: ({coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f})\n"
                                 f"Original: {original_val:.6f}\n"
                                 f"Normalized: {normalized_val:.3f}\n"
                                 f"Scaled: {scaled_val:.3f}")

                    # Add GT info if available
                    if 'gt_label' in cloud.array_names:
                        gt_val = cloud['gt_label'][point_id]
                        pick_text += f"\nGT Label: {'Anomaly' if gt_val > 0 else 'Normal'}"
                        print(f"    GT: {'Anomaly' if gt_val > 0 else 'Normal'}")

                    # Update pick text
                    if hasattr(plotter, '_pick_text_actor'):
                        plotter.remove_actor(plotter._pick_text_actor)

                    plotter._pick_text_actor = plotter.add_text(pick_text, position='upper_right',
                                                                font_size=12, color='red')
                    plotter.render()

            except (ValueError, TypeError, IndexError) as e:
                print(f"Error processing picked point: {e}")

        # Enable point picking
        plotter.enable_point_picking(
            callback=point_pick_callback,
            show_message=False,
            color='red',
            point_size=15,
            use_mesh=True,
            show_point=True
        )

        # Helper functions
        def update_scaling():
            """Update the scaling and refresh the visualization."""
            scaled_values = np.clip(cloud['normalized_score'] * self.current_scale, 0.0, 1.0)
            cloud['scaled_score'] = scaled_values
            actor.mapper.scalar_range = (0.0, 1.0)
            plotter.render()
            print(f">>> Scale factor: {self.current_scale:.3f}x")

        def update_info_text():
            """Update the info text with current scale."""
            nonlocal info_actor
            plotter.remove_actor(info_actor)
            info_actor = plotter.add_text(generate_info_text(), position='upper_left', font_size=12)

        def update_colormap():
            """Update the colormap efficiently without recreating mesh."""
            actor.mapper.lookup_table.cmap = self.current_colormap
            if scalar_bar:
                scalar_bar.SetTitle(f"Anomaly Score ({self.current_colormap})")
            plotter.render()
            print(f">>> Colormap changed to: {self.current_colormap}")

        # Point size tracking
        current_point_size = self.point_size

        # Keyboard handling
        def on_key_press(key):
            nonlocal current_point_size
            key = key.lower()

            # Point size controls
            if key == 'm':
                current_point_size *= 1.5
                actor.GetProperty().SetPointSize(current_point_size)
                plotter.render()
                print(f">>> Point size increased to: {current_point_size:.1f}")

            elif key == 'l':
                current_point_size = max(0.5, current_point_size / 1.5)
                actor.GetProperty().SetPointSize(current_point_size)
                plotter.render()
                print(f">>> Point size decreased to: {current_point_size:.1f}")

            elif key == 'r':
                current_point_size = self.point_size
                actor.GetProperty().SetPointSize(current_point_size)
                plotter.render()
                print(f">>> Point size reset to: {current_point_size:.1f}")

            # Colormap controls
            elif key == 'n':
                self.current_colormap_idx = (self.current_colormap_idx + 1) % len(self.colormap_names)
                self.current_colormap = self.colormap_names[self.current_colormap_idx]
                update_colormap()
                update_info_text()

            elif key == 'p':
                self.current_colormap_idx = (self.current_colormap_idx - 1) % len(self.colormap_names)
                self.current_colormap = self.colormap_names[self.current_colormap_idx]
                update_colormap()
                update_info_text()

            # Scale controls
            elif key == 'y':
                self.current_scale *= self.scale_increment
                update_scaling()
                update_info_text()

            elif key == 't':
                self.current_scale = max(0.1, self.current_scale / self.scale_increment)
                update_scaling()
                update_info_text()

            elif key == 'u':
                self.current_scale = 1.0
                update_scaling()
                update_info_text()

            # Reference frame toggle
            elif key == 'x':
                nonlocal reference_frame_visible
                reference_frame_visible = not reference_frame_visible
                if reference_frame_visible:
                    add_reference_frame()
                    print(">>> Reference frame (XYZ axes) shown")
                else:
                    remove_reference_frame()
                    print(">>> Reference frame (XYZ axes) hidden")
                plotter.render()

            # Statistics and help
            elif key == 's':
                scaled_values = cloud['scaled_score']
                print(f"\n>>> Score Statistics:")
                print(f"Current scale factor: {self.current_scale:.3f}x")
                print(f"Scaled values: min={scaled_values.min():.3f}, max={scaled_values.max():.3f}")
                print(f"Values clipped to 1.0: {np.sum(scaled_values >= 1.0)} points")
                print(f"Percentage clipped: {100.0 * np.sum(scaled_values >= 1.0) / len(scaled_values):.1f}%")

            elif key == 'h':
                print("\n>>> Keyboard shortcuts:")
                print("  l/m : Decrease/increase point size")
                print("  n/p : Next/previous colormap")
                print("  t/y : Decrease/increase scale factor")
                print("  u   : Reset scale to 1.0x")
                print("  x   : Toggle reference frame (XYZ axes)")
                print("  s   : Show scale statistics")
                print("  r   : Reset point size")
                print("  h   : Show this help")
                print("  q   : Quit")

        # Register keyboard events
        key_mappings = ['l', 'm', 'r', 'n', 'p', 't', 'y', 'u', 'x', 's', 'h']
        try:
            for key in key_mappings:
                plotter.add_key_event(key, lambda k=key: on_key_press(k))
        except Exception as e:
            print(f"Could not register some keyboard shortcuts: {e}")

        # Add instructions
        instructions = ("CONTROLS:\n"
                        "• Click: Pick points for values\n"
                        "• n/p: Next/Previous colormap\n"
                        "• t/y: Scale down/up\n"
                        "• u: Reset scale • s: Scale stats\n"
                        "• l/m: Point size • h: Help")
        plotter.add_text(instructions, position='lower_left', font_size=10)

        # Show the plot
        plotter.show()

    def _save_screenshot(self, cloud, sample_name):
        """Save screenshot of the visualization."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Create offscreen plotter for screenshot
        plotter = pv.Plotter(off_screen=True, window_size=self.window_size)

        # Add point cloud
        plotter.add_mesh(
            cloud,
            cmap=self.current_colormap,
            point_size=self.point_size,
            render_points_as_spheres=True,
            scalars='scaled_score'
        )

        # Add scalar bar
        if self.show_scalar_bar:
            plotter.add_scalar_bar(
                title="Anomaly Score",
                n_labels=5,
                italic=False,
                fmt="%.3f"
            )

        # Set camera position
        plotter.camera_position = 'iso'

        # Save screenshot
        safe_name = sample_name.replace('/', '_').replace('\\', '_')
        filename = os.path.join(self.output_dir,
                               f"pyvista_{safe_name}_{self.current_colormap}.png")
        plotter.screenshot(filename)
        plotter.close()
        print(f"Screenshot saved to: {filename}")


def visualize_anomaly_scores(points, scores, gt_mask=None, title="Anomaly Detection", show=True):
    """
    Convenience function for visualizing anomaly scores on point clouds.

    Args:
        points: Point cloud numpy array (N, 3)
        scores: Anomaly scores per point (N,)
        gt_mask: Optional ground truth labels (N,)
        title: Window title (default: "Anomaly Detection")
        show: Whether to show interactive visualization (default: True)
    """
    if not PYVISTA_AVAILABLE:
        print("Warning: PyVista not available. Install with: pip install pyvista")
        return

    visualizer = PyVistaAnomalyVisualizer(
        colormap='jet',
        interactive=show,
        save_screenshots=False,
        point_size=9.0,
        show_scalar_bar=True,
        window_size=(1200, 900),
        show_boundary=True,
        show_reference_frame=True
    )

    visualizer.visualize(
        points=points,
        anomaly_scores=scores,
        sample_name=title,
        gt_labels=gt_mask
    )

