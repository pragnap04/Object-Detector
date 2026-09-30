import customtkinter as ctk
from PIL import Image
from ultralytics import YOLO
import cv2
import threading
import time


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = 416
CAMERA_INDEX = 0


# ============================================================
# APPEARANCE
# ============================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ============================================================
# APPLICATION
# ============================================================

class ObjectDetectorApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Object Detector")
        self.geometry("1250x760")
        self.minsize(950, 650)

        self.protocol("WM_DELETE_WINDOW", self.close_app)

        # YOLO model
        self.model = YOLO(MODEL_PATH)

        # Camera
        self.cap = None
        self.camera_running = False
        self.camera_thread = None

        # FPS
        self.prev_time = time.time()
        self.fps = 0

        # Detected objects
        self.detected_objects = []

        # Current GUI image
        self.current_image = None

        self.create_ui()

    # ========================================================
    # CREATE UI
    # ========================================================

    def create_ui(self):

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ====================================================
        # SIDEBAR
        # ====================================================

        self.sidebar = ctk.CTkFrame(
            self,
            width=230,
            corner_radius=0
        )

        self.sidebar.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        self.sidebar.grid_propagate(False)

        self.logo = ctk.CTkLabel(
            self.sidebar,
            text="VISION",
            font=ctk.CTkFont(
                size=28,
                weight="bold"
            )
        )

        self.logo.pack(
            padx=25,
            pady=(40, 0)
        )

        self.logo_subtitle = ctk.CTkLabel(
            self.sidebar,
            text="OBJECT DETECTOR",
            font=ctk.CTkFont(
                size=12,
                weight="bold"
            ),
            text_color="gray"
        )

        self.logo_subtitle.pack(
            padx=25,
            pady=(2, 40)
        )

        self.nav_title = ctk.CTkLabel(
            self.sidebar,
            text="DETECTION",
            font=ctk.CTkFont(
                size=11,
                weight="bold"
            ),
            text_color="gray"
        )

        self.nav_title.pack(
            padx=25,
            anchor="w"
        )

        self.live_button = ctk.CTkButton(
            self.sidebar,
            text="●  Live Camera",
            height=45,
            corner_radius=8,
            font=ctk.CTkFont(
                size=14,
                weight="bold"
            ),
            command=self.scroll_to_camera
        )

        self.live_button.pack(
            padx=20,
            pady=(10, 20),
            fill="x"
        )

        # Sidebar status
        self.sidebar_bottom = ctk.CTkFrame(
            self.sidebar,
            fg_color="transparent"
        )

        self.sidebar_bottom.pack(
            side="bottom",
            padx=20,
            pady=25,
            fill="x"
        )

        self.status_dot = ctk.CTkLabel(
            self.sidebar_bottom,
            text="●",
            text_color="#777777",
            font=ctk.CTkFont(size=14)
        )

        self.status_dot.pack(
            side="left"
        )

        self.status_text = ctk.CTkLabel(
            self.sidebar_bottom,
            text="Camera Stopped",
            text_color="gray",
            font=ctk.CTkFont(size=12)
        )

        self.status_text.pack(
            side="left",
            padx=8
        )

        # ====================================================
        # MAIN
        # ====================================================

        self.main = ctk.CTkFrame(
            self,
            corner_radius=0,
            fg_color="#111111"
        )

        self.main.grid(
            row=0,
            column=1,
            sticky="nsew"
        )

        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_rowconfigure(2, weight=1)

        # ====================================================
        # HEADER
        # ====================================================

        self.header = ctk.CTkFrame(
            self.main,
            fg_color="transparent"
        )

        self.header.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=35,
            pady=(30, 10)
        )

        self.title_label = ctk.CTkLabel(
            self.header,
            text="Live Object Detection",
            font=ctk.CTkFont(
                size=26,
                weight="bold"
            )
        )

        self.title_label.pack(
            anchor="w"
        )

        self.description = ctk.CTkLabel(
            self.header,
            text="Real-time object detection using YOLO",
            font=ctk.CTkFont(size=13),
            text_color="gray"
        )

        self.description.pack(
            anchor="w",
            pady=(3, 0)
        )

        # ====================================================
        # CONTENT
        # ====================================================

        self.content = ctk.CTkFrame(
            self.main,
            fg_color="transparent"
        )

        self.content.grid(
            row=2,
            column=0,
            sticky="nsew",
            padx=35,
            pady=15
        )

        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_columnconfigure(1, weight=0)
        self.content.grid_rowconfigure(0, weight=1)

        # ====================================================
        # CAMERA
        # ====================================================

        self.camera_container = ctk.CTkFrame(
            self.content,
            corner_radius=12,
            fg_color="#181818"
        )

        self.camera_container.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 12)
        )

        self.camera_container.grid_columnconfigure(
            0,
            weight=1
        )

        self.camera_container.grid_rowconfigure(
            0,
            weight=1
        )

        self.camera_label = ctk.CTkLabel(
            self.camera_container,
            text="Camera is not active\n\nClick  Start Live Camera  to begin",
            font=ctk.CTkFont(
                size=18,
                weight="bold"
            ),
            text_color="#777777",
            fg_color="#0d0d0d",
            corner_radius=10
        )

        self.camera_label.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=15,
            pady=15
        )

        # ====================================================
        # DETECTED OBJECTS
        # ====================================================

        self.objects_panel = ctk.CTkFrame(
            self.content,
            width=230,
            corner_radius=12,
            fg_color="#181818"
        )

        self.objects_panel.grid(
            row=0,
            column=1,
            sticky="ns"
        )

        self.objects_panel.grid_propagate(False)

        self.objects_title = ctk.CTkLabel(
            self.objects_panel,
            text="Detected Objects",
            font=ctk.CTkFont(
                size=18,
                weight="bold"
            )
        )

        self.objects_title.pack(
            anchor="w",
            padx=20,
            pady=(25, 5)
        )

        self.objects_subtitle = ctk.CTkLabel(
            self.objects_panel,
            text="Objects currently visible",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )

        self.objects_subtitle.pack(
            anchor="w",
            padx=20
        )

        self.objects_count = ctk.CTkLabel(
            self.objects_panel,
            text="0",
            font=ctk.CTkFont(
                size=34,
                weight="bold"
            ),
            text_color="#3b82f6"
        )

        self.objects_count.pack(
            anchor="w",
            padx=20,
            pady=(20, 15)
        )

        self.object_list_frame = ctk.CTkScrollableFrame(
            self.objects_panel,
            fg_color="transparent"
        )

        self.object_list_frame.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=(0, 15)
        )

        self.show_no_objects()

        # ====================================================
        # CONTROLS
        # ====================================================

        self.controls = ctk.CTkFrame(
            self.main,
            fg_color="transparent"
        )

        self.controls.grid(
            row=3,
            column=0,
            sticky="ew",
            padx=35,
            pady=(5, 30)
        )

        self.controls.grid_columnconfigure(
            0,
            weight=1
        )

        self.control_status = ctk.CTkLabel(
            self.controls,
            text="Ready to start camera",
            font=ctk.CTkFont(size=13),
            text_color="gray"
        )

        self.control_status.grid(
            row=0,
            column=0,
            sticky="w"
        )

        self.start_button = ctk.CTkButton(
            self.controls,
            text="▶  Start Live Camera",
            width=170,
            height=42,
            corner_radius=8,
            font=ctk.CTkFont(
                size=13,
                weight="bold"
            ),
            command=self.start_camera
        )

        self.start_button.grid(
            row=0,
            column=1,
            padx=(10, 0)
        )

        self.stop_button = ctk.CTkButton(
            self.controls,
            text="■  Stop Camera",
            width=140,
            height=42,
            corner_radius=8,
            fg_color="#333333",
            hover_color="#444444",
            font=ctk.CTkFont(
                size=13,
                weight="bold"
            ),
            command=self.stop_camera,
            state="disabled"
        )

        self.stop_button.grid(
            row=0,
            column=2,
            padx=(10, 0)
        )

    # ========================================================
    # NO OBJECTS
    # ========================================================

    def show_no_objects(self):

        for widget in self.object_list_frame.winfo_children():
            widget.destroy()

        label = ctk.CTkLabel(
            self.object_list_frame,
            text="No objects detected",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )

        label.pack(
            pady=15
        )

    # ========================================================
    # START CAMERA
    # ========================================================

    def start_camera(self):

        if self.camera_running:
            return

        self.cap = cv2.VideoCapture(CAMERA_INDEX)

        if not self.cap.isOpened():

            self.control_status.configure(
                text="Unable to access camera",
                text_color="#ff5555"
            )

            self.cap.release()
            self.cap = None

            return

        self.camera_running = True

        self.start_button.configure(
            state="disabled",
            text="Camera Running"
        )

        self.stop_button.configure(
            state="normal"
        )

        self.prev_time = time.time()

        self.status_dot.configure(
            text_color="#22c55e"
        )

        self.status_text.configure(
            text="Camera Active",
            text_color="#22c55e"
        )

        self.control_status.configure(
            text="Live detection running",
            text_color="#22c55e"
        )

        self.camera_thread = threading.Thread(
            target=self.detection_loop,
            daemon=True
        )

        self.camera_thread.start()

    # ========================================================
    # DETECTION LOOP
    # ========================================================

    def detection_loop(self):

        while self.camera_running:

            ret, frame = self.cap.read()

            if not ret:
                break

            results = self.model.track(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                imgsz=IMAGE_SIZE,
                persist=True,
                verbose=False
            )

            annotated_frame = results[0].plot()

            # -----------------------------------------------
            # OBJECT NAMES
            # -----------------------------------------------

            detected = []

            for box in results[0].boxes:

                class_id = int(box.cls[0])

                object_name = self.model.names[class_id]

                if object_name not in detected:
                    detected.append(object_name)

            detected.sort()

            object_count = len(results[0].boxes)

            # -----------------------------------------------
            # FPS
            # -----------------------------------------------

            current_time = time.time()

            elapsed = current_time - self.prev_time

            if elapsed > 0:
                self.fps = 1 / elapsed

            self.prev_time = current_time

            # -----------------------------------------------
            # CAMERA INFO
            # -----------------------------------------------

            cv2.rectangle(
                annotated_frame,
                (10, 10),
                (250, 70),
                (25, 25, 25),
                -1
            )

            cv2.putText(
                annotated_frame,
                f"Objects: {object_count}",
                (22, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                annotated_frame,
                f"FPS: {self.fps:.1f}",
                (22, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

            # -----------------------------------------------
            # CONVERT IMAGE
            # -----------------------------------------------

            frame_rgb = cv2.cvtColor(
                annotated_frame,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(frame_rgb)

            # -----------------------------------------------
            # GUI UPDATE
            # -----------------------------------------------

            if self.camera_running:

                self.after(
                    0,
                    self.update_camera,
                    image
                )

                self.after(
                    0,
                    self.update_object_list,
                    detected,
                    object_count
                )

        # Camera loop ended
        if self.winfo_exists():

            self.after(
                0,
                self.camera_stopped
            )

    # ========================================================
    # UPDATE CAMERA
    # ========================================================

    def update_camera(self, image):

        if not self.camera_running:
            return

        if not self.winfo_exists():
            return

        width = self.camera_label.winfo_width()
        height = self.camera_label.winfo_height()

        if width < 100:
            width = 800

        if height < 100:
            height = 500

        image.thumbnail(
            (width - 20, height - 20),
            Image.Resampling.LANCZOS
        )

        new_image = ctk.CTkImage(
            light_image=image,
            dark_image=image,
            size=image.size
        )

        self.current_image = new_image

        self.camera_label.configure(
            image=self.current_image,
            text=""
        )

        # ========================================================
    # UPDATE OBJECT LIST
    # ========================================================

    def update_object_list(self, objects, object_count):

        if not self.camera_running:
            return

        if not self.winfo_exists():
            return

        self.detected_objects = objects

        # Total number of detected object instances
        self.objects_count.configure(
            text=str(object_count)
        )

        # Clear previous list
        for widget in self.object_list_frame.winfo_children():
            widget.destroy()

        # Nothing detected
        if not objects:

            self.show_no_objects()

            return

        # Display unique object types
        for object_name in objects:

            row = ctk.CTkFrame(
                self.object_list_frame,
                fg_color="#222222",
                corner_radius=8
            )

            row.pack(
                fill="x",
                padx=5,
                pady=4
            )

            label = ctk.CTkLabel(
                row,
                text=f"●  {object_name.title()}",
                font=ctk.CTkFont(
                    size=13,
                    weight="bold"
                ),
                anchor="w"
            )

            label.pack(
                padx=12,
                pady=9,
                fill="x"
            )

    

    # ========================================================
    # STOP CAMERA
    # ========================================================

    def stop_camera(self):

        if not self.camera_running:
            return

        self.camera_running = False

        if self.cap is not None:

            self.cap.release()
            self.cap = None

        self.camera_stopped()

    # ========================================================
    # CAMERA STOPPED
    # ========================================================

    def camera_stopped(self):

        if not self.winfo_exists():
            return

        self.camera_running = False

        self.start_button.configure(
            state="normal",
            text="▶  Start Live Camera"
        )

        self.stop_button.configure(
            state="disabled"
        )

        self.status_dot.configure(
            text_color="#777777"
        )

        self.status_text.configure(
            text="Camera Stopped",
            text_color="gray"
        )

        self.control_status.configure(
            text="Ready to start camera",
            text_color="gray"
        )

        self.current_image = None

        self.camera_label.configure(
            image=None,
            text="Camera is not active\n\nClick  Start Live Camera  to begin",
            font=ctk.CTkFont(
                size=18,
                weight="bold"
            ),
            text_color="#777777"
        )

        self.show_no_objects()

        self.objects_count.configure(
            text="0"
        )

    # ========================================================
    # NAVIGATION
    # ========================================================

    def scroll_to_camera(self):
        pass

    # ========================================================
    # CLOSE APPLICATION
    # ========================================================

    def close_app(self):

        self.camera_running = False

        if self.cap is not None:

            self.cap.release()
            self.cap = None

        self.destroy()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app = ObjectDetectorApp()

    app.mainloop()