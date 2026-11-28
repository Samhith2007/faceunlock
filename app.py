import math
import os
import threading
from collections import deque
from datetime import datetime
from tkinter import messagebox

import cv2
import customtkinter as ctk
import numpy as np
from deepface import DeepFace
from PIL import Image
from sklearn.datasets import fetch_lfw_people

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "images", "known_faces")
FALLBACK_REGISTERED_IMAGE = os.path.join(BASE_DIR, "images", "registered_user.jpg")
CAPTURED_IMAGE = os.path.join(BASE_DIR, "captured_face.jpg")
MODEL_NAME = "Facenet512"
COSINE_THRESHOLD = 0.45
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


# ==========================================
# MAIN APPLICATION
# ==========================================
class FaceIDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FaceID Unlocking")
        self.root.geometry("1200x780")
        self.root.minsize(1000, 680)
        self.root.configure(fg_color="#F4F7FC")

        self.running = True
        self.verifying = False
        self.testing_lfw = False
        self.current_frame = None
        self.preview_image = None
        self.history_items = []
        self.menu_buttons = {}
        self.cards = {}
        self.known_embeddings = {}

        # Smoothing buffers for face measurements
        self.w_hist = deque(maxlen=20)
        self.h_hist = deque(maxlen=20)
        self.nose_hist = deque(maxlen=20)
        self.last_eye_dist = None
        self.eye_hold_frames = 0

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        self.create_sidebar()
        self.create_main_area()
        self.select_menu("Dashboard")
        self.refresh_known_faces_label()
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

        if self.cap.isOpened():
            self.update_camera()
        else:
            self.status_label.configure(
                text="CAMERA NOT AVAILABLE",
                fg_color="#FEE2E2",
                text_color="#DC2626",
            )

    def get_dataset_images(self):
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        entries = []

        for filename in sorted(os.listdir(KNOWN_FACES_DIR)):
            if filename.lower().endswith(VALID_EXTENSIONS):
                full_path = os.path.join(KNOWN_FACES_DIR, filename)
                label = os.path.splitext(filename)[0].replace("_", " ").strip()
                if label:
                    entries.append((label, full_path))

        if not entries and os.path.exists(FALLBACK_REGISTERED_IMAGE):
            entries.append(("Authorized Profile", FALLBACK_REGISTERED_IMAGE))

        return entries

    def refresh_known_faces_label(self):
        entries = self.get_dataset_images()
        if not entries:
            self.authorized_users_label.configure(
                text="No images in images/known_faces/"
            )
        else:
            names = ", ".join(name for name, _ in entries)
            self.authorized_users_label.configure(
                text=f"{len(entries)} enrolled: {names}"
            )

    def load_known_embeddings(self):
        entries = self.get_dataset_images()
        if not entries:
            raise RuntimeError(
                "No enrolled face images found.\nAdd photos to images/known_faces/ first."
            )

        loaded = {}
        for person_name, img_path in entries:
            mtime = os.path.getmtime(img_path)
            cache_key = (img_path, mtime)
            cached = self.known_embeddings.get(person_name)
            if cached and cached["key"] == cache_key:
                loaded[person_name] = cached
                continue

            res = DeepFace.represent(
                img_path=img_path,
                model_name=MODEL_NAME,
                detector_backend="opencv",
                enforce_detection=False,
            )
            if res and res[0].get("embedding"):
                loaded[person_name] = {
                    "key": cache_key,
                    "embedding": res[0]["embedding"],
                }

        if not loaded:
            raise RuntimeError("Could not extract face embeddings from dataset images.")

        self.known_embeddings = loaded
        return loaded

    # ======================================
    # SIDEBAR
    # ======================================
    def create_sidebar(self):
        self.sidebar = ctk.CTkFrame(
            self.root, width=230, corner_radius=0, fg_color="#FFFFFF"
        )
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        ctk.CTkLabel(
            self.sidebar,
            text="FACEID",
            font=("Arial", 30, "bold"),
            text_color="#2563EB",
        ).pack(pady=(40, 5))

        ctk.CTkLabel(
            self.sidebar,
            text="SMART SECURITY SYSTEM",
            font=("Arial", 10, "bold"),
            text_color="#64748B",
        ).pack(pady=(0, 45))

        for name in ["Dashboard", "Face Verification", "Security History"]:
            btn = ctk.CTkButton(
                self.sidebar,
                text="   " + name,
                height=45,
                anchor="w",
                font=("Arial", 13, "bold"),
                corner_radius=9,
                command=lambda n=name: self.select_menu(n),
            )
            btn.pack(fill="x", padx=15, pady=5)
            self.menu_buttons[name] = btn

        ctk.CTkLabel(
            self.sidebar,
            text="SYSTEM STATUS",
            font=("Arial", 11, "bold"),
            text_color="#64748B",
        ).pack(anchor="w", padx=22, pady=(65, 12))

        ctk.CTkLabel(
            self.sidebar,
            text="● SYSTEM READY",
            font=("Arial", 12, "bold"),
            text_color="#16A34A",
        ).pack(anchor="w", padx=22)

        ctk.CTkLabel(
            self.sidebar,
            text="DeepFace | FaceNet512",
            font=("Arial", 11),
            text_color="#64748B",
        ).pack(anchor="w", padx=22, pady=8)

        ctk.CTkLabel(
            self.sidebar,
            text="Educational Prototype",
            font=("Arial", 10),
            text_color="#94A3B8",
        ).pack(side="bottom", pady=25)

    def select_menu(self, name):
        for item_name, btn in self.menu_buttons.items():
            if item_name == name:
                btn.configure(
                    fg_color="#EAF1FF", text_color="#2563EB", hover_color="#DCE8FF"
                )
            else:
                btn.configure(
                    fg_color="#FFFFFF", text_color="#475569", hover_color="#F1F5F9"
                )

        for card in self.cards.values():
            card.pack_forget()

        if name == "Dashboard":
            self.header_title.configure(text="FaceID unlocking")
            self.header_sub.configure(
                text="AI-powered facial recognition and verification"
            )
            self.cards["verify"].pack(fill="x", pady=(0, 12))
            self.cards["measurement"].pack(fill="x", pady=6)
            self.cards["embedding"].pack(fill="x", pady=6)
            self.cards["history"].pack(fill="x", pady=6)
        elif name == "Face Verification":
            self.header_title.configure(text="Face Verification")
            self.header_sub.configure(
                text="Verify live face against authorized dataset profiles"
            )
            self.cards["verify"].pack(fill="x", pady=(0, 12))
            self.cards["embedding"].pack(fill="x", pady=6)
        elif name == "Security History":
            self.header_title.configure(text="Security History")
            self.header_sub.configure(
                text="Verification logs and live face measurements"
            )
            self.cards["history"].pack(fill="x", pady=(0, 12))
            self.cards["measurement"].pack(fill="x", pady=6)

    # ======================================
    # MAIN INTERFACE
    # ======================================
    def create_main_area(self):
        self.main = ctk.CTkFrame(
            self.root, fg_color="#F4F7FC", corner_radius=0
        )
        self.main.pack(
            side="right", fill="both", expand=True, padx=25, pady=20
        )

        header = ctk.CTkFrame(self.main, fg_color="transparent")
        header.pack(fill="x", pady=(5, 20))

        self.header_title = ctk.CTkLabel(
            header,
            text="FaceID unlocking",
            font=("Arial", 30, "bold"),
            text_color="#172554",
        )
        self.header_title.pack(anchor="w")

        self.header_sub = ctk.CTkLabel(
            header,
            text="AI-powered facial recognition and verification",
            font=("Arial", 13),
            text_color="#64748B",
        )
        self.header_sub.pack(anchor="w", pady=5)

        content = ctk.CTkFrame(self.main, fg_color="transparent")
        content.pack(fill="both", expand=True)
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=2)
        content.grid_rowconfigure(0, weight=1)

        camera_card = ctk.CTkFrame(
            content, fg_color="#FFFFFF", corner_radius=18
        )
        camera_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=5)

        ctk.CTkLabel(
            camera_card,
            text="LIVE FACE DETECTION",
            font=("Arial", 17, "bold"),
            text_color="#172554",
        ).pack(anchor="w", padx=20, pady=(20, 12))

        self.camera_label = ctk.CTkLabel(
            camera_card,
            text="Starting camera...",
            width=640,
            height=400,
            fg_color="#E2E8F0",
            text_color="#475569",
            corner_radius=12,
        )
        self.camera_label.pack(padx=15, pady=5, fill="both", expand=True)

        ctk.CTkLabel(
            camera_card,
            text="Keep face uncovered and clearly visible in front of camera",
            font=("Arial", 11),
            text_color="#64748B",
        ).pack(pady=12)

        self.right_panel = ctk.CTkScrollableFrame(
            content, fg_color="transparent", corner_radius=0
        )
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(12, 0), pady=5)

        verify_card = ctk.CTkFrame(
            self.right_panel, fg_color="#FFFFFF", corner_radius=18
        )
        verify_card.pack(fill="x", pady=(0, 12))
        self.cards["verify"] = verify_card

        ctk.CTkLabel(
            verify_card,
            text="FACE VERIFICATION",
            font=("Arial", 16, "bold"),
            text_color="#172554",
        ).pack(anchor="w", padx=18, pady=(20, 12))

        ctk.CTkLabel(
            verify_card,
            text="AUTHORIZED DATASET PROFILES",
            font=("Arial", 11, "bold"),
            text_color="#64748B",
        ).pack(anchor="w", padx=18)

        self.authorized_users_label = ctk.CTkLabel(
            verify_card,
            text="Checking images/known_faces/...",
            font=("Arial", 13, "bold"),
            text_color="#1E293B",
            wraplength=320,
            justify="left",
        )
        self.authorized_users_label.pack(anchor="w", padx=18, pady=(5, 16))

        self.status_label = ctk.CTkLabel(
            verify_card,
            text="WAITING FOR VERIFICATION",
            height=48,
            fg_color="#FFF7D6",
            text_color="#A16207",
            corner_radius=10,
            font=("Arial", 12, "bold"),
        )
        self.status_label.pack(fill="x", padx=18, pady=8)

        self.verify_button = ctk.CTkButton(
            verify_card,
            text="VERIFY FACE",
            height=48,
            corner_radius=10,
            font=("Arial", 14, "bold"),
            fg_color="#2563EB",
            hover_color="#1D4ED8",
            command=self.verify_face,
        )
        self.verify_button.pack(fill="x", padx=18, pady=10)

        self.lfw_button = ctk.CTkButton(
            verify_card,
            text="RUN LFW BENCHMARK (6 PAIRS)",
            height=38,
            corner_radius=10,
            fg_color="#F1F5F9",
            hover_color="#E2E8F0",
            text_color="#1E293B",
            command=self.run_lfw_benchmark,
        )
        self.lfw_button.pack(fill="x", padx=18, pady=(0, 18))

        measurement_card = ctk.CTkFrame(
            self.right_panel, fg_color="#FFFFFF", corner_radius=18
        )
        measurement_card.pack(fill="x", pady=6)
        self.cards["measurement"] = measurement_card

        ctk.CTkLabel(
            measurement_card,
            text="FACE MEASUREMENTS",
            font=("Arial", 15, "bold"),
            text_color="#172554",
        ).pack(anchor="w", padx=18, pady=(15, 8))

        self.measurements_label = ctk.CTkLabel(
            measurement_card,
            text="Waiting for face...",
            font=("Arial", 12),
            text_color="#475569",
            justify="left",
        )
        self.measurements_label.pack(anchor="w", padx=18, pady=(5, 15))

        embedding_card = ctk.CTkFrame(
            self.right_panel, fg_color="#FFFFFF", corner_radius=18
        )
        embedding_card.pack(fill="x", pady=6)
        self.cards["embedding"] = embedding_card

        ctk.CTkLabel(
            embedding_card,
            text="FACENET512 EMBEDDING",
            font=("Arial", 15, "bold"),
            text_color="#172554",
        ).pack(anchor="w", padx=18, pady=(15, 5))

        ctk.CTkLabel(
            embedding_card,
            text="512 numerical values generated during verification",
            font=("Arial", 10),
            text_color="#64748B",
        ).pack(anchor="w", padx=18, pady=(0, 8))

        self.embedding_textbox = ctk.CTkTextbox(
            embedding_card, height=130, font=("Consolas", 9), wrap="word"
        )
        self.embedding_textbox.pack(fill="x", padx=12, pady=(0, 12))
        self.embedding_textbox.insert(
            "1.0", "Click VERIFY FACE to generate the embedding."
        )
        self.embedding_textbox.configure(state="disabled")

        history_card = ctk.CTkFrame(
            self.right_panel, fg_color="#FFFFFF", corner_radius=18
        )
        history_card.pack(fill="x", pady=6)
        self.cards["history"] = history_card

        ctk.CTkLabel(
            history_card,
            text="RECENT ACTIVITY",
            font=("Arial", 16, "bold"),
            text_color="#172554",
        ).pack(anchor="w", padx=18, pady=18)

        self.history_label = ctk.CTkLabel(
            history_card,
            text="No verification attempts yet",
            font=("Arial", 11),
            text_color="#64748B",
            justify="left",
        )
        self.history_label.pack(anchor="w", padx=18, pady=(0, 15))

        ctk.CTkLabel(
            self.main,
            text="Python | OpenCV | DeepFace FaceNet512 | LFW",
            font=("Arial", 10),
            text_color="#94A3B8",
        ).pack(pady=(15, 0))

    def cosine_distance(self, emb1, emb2):
        v1 = np.array(emb1, dtype=np.float32)
        v2 = np.array(emb2, dtype=np.float32)
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
        return float(1.0 - np.dot(v1, v2) / denom)

    def update_camera(self):
        if not self.running:
            return
        success, frame = self.cap.read()
        if success:
            frame = cv2.flip(frame, 1)
            self.current_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=6, minSize=(100, 100)
            )

            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (37, 99, 235), 2)

                self.w_hist.append(float(w))
                self.h_hist.append(float(h))
                self.nose_hist.append(h * 0.28)

                avg_w = int(round(float(np.mean(self.w_hist))))
                avg_h = int(round(float(np.mean(self.h_hist))))
                avg_nose = int(round(float(np.mean(self.nose_hist))))

                roi_gray = gray[y : y + h // 2, x : x + w]
                eyes = self.eye_cascade.detectMultiScale(
                    roi_gray, scaleFactor=1.15, minNeighbors=6, minSize=(22, 22)
                )

                if len(eyes) >= 2:
                    eyes_sorted = sorted(eyes, key=lambda e: e[0])
                    c1 = (
                        eyes_sorted[0][0] + eyes_sorted[0][2] / 2,
                        eyes_sorted[0][1] + eyes_sorted[0][3] / 2,
                    )
                    c2 = (
                        eyes_sorted[1][0] + eyes_sorted[1][2] / 2,
                        eyes_sorted[1][1] + eyes_sorted[1][3] / 2,
                    )
                    dist_px = int(round(math.hypot(c2[0] - c1[0], c2[1] - c1[1])))
                    if int(0.22 * w) <= dist_px <= int(0.60 * w):
                        if (
                            self.last_eye_dist is None
                            or abs(dist_px - self.last_eye_dist) >= 4
                        ):
                            self.last_eye_dist = dist_px
                        self.eye_hold_frames = 10

                if self.last_eye_dist is not None and self.eye_hold_frames > 0:
                    eye_dist_text = f"{self.last_eye_dist} px"
                    self.eye_hold_frames -= 1
                else:
                    eye_dist_text = "N/A"

                measurement_text = (
                    f"Eye distance: {eye_dist_text}\n"
                    f"Nose length: {avg_nose} px\n"
                    f"Face width: {avg_w} px\n"
                    f"Face height: {avg_h} px"
                )
                self.measurements_label.configure(text=measurement_text)
            else:
                self.w_hist.clear()
                self.h_hist.clear()
                self.nose_hist.clear()
                self.eye_hold_frames = 0
                self.measurements_label.configure(text="No face detected")

            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display_image = Image.fromarray(display_frame)
            self.preview_image = ctk.CTkImage(
                light_image=display_image,
                dark_image=display_image,
                size=(640, 400),
            )
            self.camera_label.configure(image=self.preview_image, text="")

        self.root.after(15, self.update_camera)

    def verify_face(self):
        if self.verifying:
            return
        if self.current_frame is None:
            messagebox.showerror("Camera Error", "Camera image is not available.")
            return

        self.refresh_known_faces_label()
        if not self.get_dataset_images():
            messagebox.showerror(
                "Dataset Error",
                "No enrolled face images found.\nAdd photos to images/known_faces/ first.",
            )
            return

        frame_to_verify = self.current_frame.copy()
        self.verifying = True
        self.verify_button.configure(state="disabled", text="VERIFYING...")
        self.status_label.configure(
            text="VERIFYING FACE...",
            fg_color="#FFF7D6",
            text_color="#A16207",
        )

        self.embedding_textbox.configure(state="normal")
        self.embedding_textbox.delete("1.0", "end")
        self.embedding_textbox.insert("1.0", "Generating FaceNet512 embedding...")
        self.embedding_textbox.configure(state="disabled")

        threading.Thread(
            target=self.verification_worker,
            args=(frame_to_verify,),
            daemon=True,
        ).start()

    def verification_worker(self, frame):
        try:
            saved = cv2.imwrite(CAPTURED_IMAGE, frame)
            if not saved:
                raise RuntimeError("Could not save camera image.")

            known_map = self.load_known_embeddings()

            live_res = DeepFace.represent(
                img_path=CAPTURED_IMAGE,
                model_name=MODEL_NAME,
                detector_backend="opencv",
                enforce_detection=False,
            )
            if not live_res or not live_res[0].get("embedding"):
                raise RuntimeError("No face detected in live frame.")

            live_embedding = live_res[0]["embedding"]
            embedding_text = "\n".join(
                f"Value {index + 1}: {value:.6f}"
                for index, value in enumerate(live_embedding)
            )

            best_name = None
            best_dist = float("inf")
            for person_name, data in known_map.items():
                dist = self.cosine_distance(data["embedding"], live_embedding)
                if dist < best_dist:
                    best_dist = dist
                    best_name = person_name

            verified = best_dist <= COSINE_THRESHOLD

            self.root.after(
                0,
                lambda v=verified, t=embedding_text, d=best_dist, n=best_name: (
                    self.finish_verification(v, None, t, d, n)
                ),
            )
        except Exception as error:
            error_message = str(error)
            self.root.after(
                0, lambda e=error_message: self.finish_verification(None, e)
            )

    def finish_verification(
        self, verified, error, embedding_text=None, dist=None, matched_name=None
    ):
        self.verifying = False
        if not self.running:
            return
        self.verify_button.configure(state="normal", text="VERIFY FACE")
        self.refresh_known_faces_label()

        if embedding_text is not None:
            self.embedding_textbox.configure(state="normal")
            self.embedding_textbox.delete("1.0", "end")
            self.embedding_textbox.insert("1.0", embedding_text)
            self.embedding_textbox.configure(state="disabled")

        dist_tag = f" (d={dist:.2f})" if dist is not None else ""

        if error:
            self.status_label.configure(
                text="VERIFICATION ERROR",
                fg_color="#FEE2E2",
                text_color="#B91C1C",
            )
            self.add_history("VERIFICATION ERROR")
            messagebox.showerror("Verification Error", error)
        elif verified is True:
            user_tag = f": {matched_name}" if matched_name else ""
            self.status_label.configure(
                text=f"ACCESS GRANTED{user_tag}{dist_tag}",
                fg_color="#DCFCE7",
                text_color="#15803D",
            )
            self.add_history(f"GRANTED{user_tag}{dist_tag}")
        else:
            self.status_label.configure(
                text=f"ACCESS DENIED{dist_tag}",
                fg_color="#FEE2E2",
                text_color="#B91C1C",
            )
            self.add_history(f"ACCESS DENIED{dist_tag}")

    def run_lfw_benchmark(self, pair_count=6):
        if self.testing_lfw:
            return
        self.testing_lfw = True
        self.lfw_button.configure(state="disabled", text="RUNNING LFW...")

        def worker():
            try:
                lfw = fetch_lfw_people(min_faces_per_person=2, resize=1.0)
                same_dists = []
                diff_dists = []
                max_pairs = min(pair_count, len(lfw.images) - 1)

                for i in range(max_pairs):
                    img1 = cv2.cvtColor(
                        (lfw.images[i] * 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR,
                    )
                    img2 = cv2.cvtColor(
                        (lfw.images[i + 1] * 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR,
                    )
                    emb1 = DeepFace.represent(
                        img_path=img1,
                        model_name=MODEL_NAME,
                        detector_backend="opencv",
                        enforce_detection=False,
                    )[0]["embedding"]
                    emb2 = DeepFace.represent(
                        img_path=img2,
                        model_name=MODEL_NAME,
                        detector_backend="opencv",
                        enforce_detection=False,
                    )[0]["embedding"]

                    d = self.cosine_distance(emb1, emb2)
                    if lfw.target[i] == lfw.target[i + 1]:
                        same_dists.append(d)
                    else:
                        diff_dists.append(d)

                parts = [f"LFW({max_pairs})"]
                if same_dists:
                    parts.append(f"Same={np.mean(same_dists):.2f}")
                if diff_dists:
                    parts.append(f"Diff={np.mean(diff_dists):.2f}")
                summary = " | ".join(parts)

                self.root.after(0, lambda s=summary: self._on_lfw_done(s, None))
            except Exception as exc:
                self.root.after(0, lambda e=str(exc): self._on_lfw_done(None, e))

        threading.Thread(target=worker, daemon=True).start()

    def _on_lfw_done(self, summary, err):
        self.testing_lfw = False
        if not self.running:
            return
        self.lfw_button.configure(
            state="normal", text="RUN LFW BENCHMARK (6 PAIRS)"
        )
        if err:
            messagebox.showerror("LFW Error", err)
        else:
            self.add_history(summary)

    def add_history(self, result):
        current_time = datetime.now().strftime("%I:%M:%S %p")
        self.history_items.insert(0, current_time + "  |  " + result)
        self.history_items = self.history_items[:6]
        self.history_label.configure(text="\n\n".join(self.history_items))

    def close_app(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()


if __name__ == "__main__":
    root = ctk.CTk()
    app = FaceIDApp(root)
    root.mainloop()