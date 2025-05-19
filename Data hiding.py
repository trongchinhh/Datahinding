import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.cluster import KMeans
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import hashlib
from skimage.metrics import structural_similarity as ssim
import math

class DataHidingAI:
    def __init__(self, img_height=256, img_width=256):
        self.img_height = img_height
        self.img_width = img_width
        self.model = self.build_model()
        
    def build_model(self):
        """Build a CNN model to predict optimal embedding regions with radiance field features"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_height, self.img_width, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(self.img_height * self.img_width, activation='sigmoid')  # Output a radiance-aware heatmap
        ])
        model.compile(optimizer=Adam(0.001), loss='binary_crossentropy')
        return model
    
    def generate_heatmap(self, image):
        """
        Generate a radiance-aware heatmap using edges, variance, light intensity gradients, and simulated volumetric features.
        """
        # Convert to grayscale and resize
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.img_width, self.img_height))
        
        # Edge detection
        edges = cv2.Canny(gray, 100, 200) / 255.0
        
        # Variance (texture information)
        variance_map = cv2.Laplacian(gray, cv2.CV_64F)
        variance_map = cv2.GaussianBlur(np.abs(variance_map), (5, 5), 0) / np.max(np.abs(variance_map))
        
        # Light intensity gradient (simulating radiance direction)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        intensity_grad = np.sqrt(grad_x**2 + grad_y**2) / np.max(np.sqrt(grad_x**2 + grad_y**2))
        
        # Simulate volumetric radiance (e.g., depth-like feature using Gaussian blur)
        volumetric_sim = cv2.GaussianBlur(gray, (15, 15), 0) / 255.0
        volumetric_sim = np.clip(volumetric_sim, 0, 1)
        
        # Image-specific noise based on content hash with modulo to fit seed range
        image_hash = int(hashlib.md5(image.tobytes()).hexdigest(), 16) % (2**32)
        noise = np.random.RandomState(image_hash).rand(self.img_height, self.img_width) * (gray / 255.0) * 0.2
        
        # Combine features with radiance emphasis
        heatmap = (edges * 0.25 + variance_map * 0.25 + intensity_grad * 0.25 + volumetric_sim * 0.15 + noise * 0.1)
        heatmap = np.clip(heatmap, 0, 1)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap
    
    def train_model(self, X_train, y_train, epochs=10, batch_size=32):
        """Train the model to identify good embedding regions"""
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
    @staticmethod
    def prepare_training_data(images, masks):
        """Prepare training data for the model"""
        X = []
        y = []
        for img, mask in zip(images, masks):
            img = cv2.resize(img, (256, 256))
            mask = cv2.resize(mask, (256, 256))
            img = img / 255.0
            mask = mask / 255.0
            X.append(img)
            y.append(mask.flatten())
        return np.array(X), np.array(y)
    
    @staticmethod
    def calculate_psnr(img1, img2):
        """Calculate PSNR between two images"""
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
        return psnr
    
    @staticmethod
    def calculate_ssim(img1, img2):
        """Calculate SSIM between two images"""
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        return ssim(img1_gray, img2_gray, data_range=255)
    
    @staticmethod
    def calculate_ber(original_data, extracted_data):
        """Calculate Bit Error Rate (BER)"""
        if len(original_data) != len(extracted_data):
            extracted_data = extracted_data[:len(original_data)]
        binary_original = ''.join([format(ord(i), '08b') for i in original_data])
        binary_extracted = ''.join([format(ord(i), '08b') for i in extracted_data])
        binary_extracted = binary_extracted[:len(binary_original)]
        errors = sum(a != b for a, b in zip(binary_original, binary_extracted))
        return errors / len(binary_original)
    
    @staticmethod
    def embed_data(image, secret_data, heatmap, threshold=0.3):
        """
        Embed secret data using clustering-based selection for radiance-aware distribution.
        Logs capacity and metrics for research evaluation.
        """
        binary_data = ''.join([format(ord(i), '08b') for i in secret_data])
        data_len = len(binary_data)
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        flat_heat = heatmap_resized.reshape(-1)
        flat_img = image.reshape(-1, 3)
        stego_image = image.copy()
        
        # Cluster heatmap values for diverse region selection
        image_hash = int(hashlib.md5(image.tobytes()).hexdigest(), 16) % (2**32)
        kmeans = KMeans(n_clusters=min(10, len(flat_heat) // 100), random_state=image_hash % 1000)
        clusters = kmeans.fit_predict(flat_heat.reshape(-1, 1))
        used_indices = []
        data_index = 0
        
        # Select top pixels from each cluster
        for cluster in range(kmeans.n_clusters):
            cluster_indices = np.where(clusters == cluster)[0]
            if len(cluster_indices) == 0:
                continue
            cluster_heat = flat_heat[cluster_indices]
            sorted_indices = cluster_indices[np.argsort(-cluster_heat)]
            for idx in sorted_indices[:min(10, len(sorted_indices))]:
                if data_index >= data_len or len(used_indices) >= (data_len + 2) // 3:
                    break
                if idx not in used_indices and flat_heat[idx] >= threshold:
                    used_indices.append(idx)
                    r, g, b = flat_img[idx]
                    if data_index < data_len:
                        r = (r & 254) | int(binary_data[data_index])
                        data_index += 1
                    if data_index < data_len:
                        g = (g & 254) | int(binary_data[data_index])
                        data_index += 1
                    if data_index < data_len:
                        b = (b & 254) | int(binary_data[data_index])
                        data_index += 1
                    flat_img[idx] = [r, g, b]
        
        # Fallback with remaining top heatmap values
        if data_index < data_len:
            remaining_indices = np.argsort(-flat_heat)
            for idx in remaining_indices:
                if idx in used_indices or data_index >= data_len:
                    continue
                if flat_heat[idx] >= threshold and len(used_indices) < len(flat_heat):
                    used_indices.append(idx)
                    r, g, b = flat_img[idx]
                    if data_index < data_len:
                        r = (r & 254) | int(binary_data[data_index])
                        data_index += 1
                    if data_index < data_len:
                        g = (g & 254) | int(binary_data[data_index])
                        data_index += 1
                    if data_index < data_len:
                        b = (b & 254) | int(binary_data[data_index])
                        data_index += 1
                    flat_img[idx] = [r, g, b]
                if data_index >= data_len:
                    break
        
        # Research metrics
        capacity = len(used_indices) * 3
        return stego_image, used_indices, capacity, data_index
    
    @staticmethod
    def extract_data(image, heatmap, threshold=0.3, data_len=None):
        """
        Extract hidden data using clustering-based selection for consistency.
        """
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        flat_heat = heatmap_resized.reshape(-1)
        flat_img = image.reshape(-1, 3)
        
        binary_data = ''
        extracted_len = 0
        image_hash = int(hashlib.md5(image.tobytes()).hexdigest(), 16) % (2**32)
        kmeans = KMeans(n_clusters=min(10, len(flat_heat) // 100), random_state=image_hash % 1000)
        clusters = kmeans.fit_predict(flat_heat.reshape(-1, 1))
        used_indices = set()
        
        for cluster in range(kmeans.n_clusters):
            cluster_indices = np.where(clusters == cluster)[0]
            if len(cluster_indices) == 0:
                continue
            cluster_heat = flat_heat[cluster_indices]
            sorted_indices = cluster_indices[np.argsort(-cluster_heat)]
            for idx in sorted_indices[:min(10, len(sorted_indices))]:
                if data_len is not None and extracted_len >= data_len * 8:
                    break
                if idx not in used_indices and flat_heat[idx] >= threshold:
                    used_indices.add(idx)
                    r, g, b = flat_img[idx]
                    binary_data += str(r & 1)
                    binary_data += str(g & 1)
                    binary_data += str(b & 1)
                    extracted_len += 3
        
        secret_data = ''
        for i in range(0, min(len(binary_data), data_len * 8 if data_len else len(binary_data)), 8):
            byte = binary_data[i:i+8]
            secret_data += chr(int(byte, 2))
        return secret_data
    
    @staticmethod
    def create_marked_image(image, heatmap, used_indices):
        """
        Create an image with highlighted regions where data was embedded
        """
        marked_image = image.copy()
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        height, width = image.shape[:2]
        for idx in used_indices:
            row = idx // width
            col = idx % width
            if row < height and col < width:
                mask[row, col] = True
        marked_image[mask] = [0, 0, 255]  # Red in BGR
        return marked_image

class SteganographyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Steganography Application")
        self.stego_ai = DataHidingAI()
        
        self.label_message = tk.Label(root, text="Enter Secret Message:", font=("Arial", 16))
        self.label_message.pack(pady=10)
        
        self.message_entry = tk.Text(root, height=5, width=50, font=("Arial", 14))
        self.message_entry.pack(pady=10)
        
        self.select_image_button = tk.Button(root, text="Select Image", command=self.select_image, font=("Arial", 14))
        self.select_image_button.pack(pady=10)
        
        self.image_path_label = tk.Label(root, text="No image selected", font=("Arial", 14))
        self.image_path_label.pack(pady=10)
        
        self.run_button = tk.Button(root, text="Run Steganography", command=self.run_steganography, font=("Arial", 14))
        self.run_button.pack(pady=15)
        
        self.image_frame = tk.Frame(root)
        self.image_frame.pack(pady=15)
        
        self.original_label = tk.Label(self.image_frame)
        self.original_label.grid(row=0, column=0, padx=15)
        
        self.stego_label = tk.Label(self.image_frame)
        self.stego_label.grid(row=0, column=1, padx=15)
        
        self.marked_label = tk.Label(self.image_frame)
        self.marked_label.grid(row=0, column=2, padx=15)
        
        self.original_title = tk.Label(self.image_frame, text="Ảnh gốc", font=("Arial", 16))
        self.original_title.grid(row=1, column=0)
        
        self.stego_title = tk.Label(self.image_frame, text="Ảnh đã giấu tin", font=("Arial", 16))
        self.stego_title.grid(row=1, column=1)
        
        self.marked_title = tk.Label(self.image_frame, text="Vùng đã giấu tin", font=("Arial", 16))
        self.marked_title.grid(row=1, column=2)
        
        self.psnr_label = tk.Label(root, text="PSNR: N/A dB", font=("Arial", 14))
        self.psnr_label.pack(pady=5)
        
        self.metrics_label = tk.Label(root, text="SSIM/BER: N/A", font=("Arial", 14))
        self.metrics_label.pack(pady=5)
        
        self.image_path = None
        self.image_references = []
    
    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.image_path = file_path
            self.image_path_label.config(text=f"Selected: {os.path.basename(file_path)}")
    
    def run_steganography(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please select an image first!")
            return
        
        secret_message = self.message_entry.get("1.0", tk.END).strip()
        if not secret_message:
            messagebox.showerror("Error", "Please enter a secret message!")
            return
        
        image = cv2.imread(self.image_path)
        if image is None:
            messagebox.showerror("Error", f"Failed to load image at {self.image_path}")
            return
        
        heatmap = self.stego_ai.generate_heatmap(image)
        stego_image, used_indices, capacity, data_index = DataHidingAI.embed_data(image.copy(), secret_message, heatmap)
        marked_image = DataHidingAI.create_marked_image(stego_image, heatmap, used_indices)
        extracted_message = DataHidingAI.extract_data(stego_image, heatmap, data_len=len(secret_message))
        
        # Calculate evaluation metrics
        psnr_value = DataHidingAI.calculate_psnr(image, stego_image)
        ssim_value = DataHidingAI.calculate_ssim(image, stego_image)
        ber_value = DataHidingAI.calculate_ber(secret_message, extracted_message)
        
        # Update metrics display
        self.psnr_label.config(text=f"PSNR: {psnr_value:.2f} dB")
        self.metrics_label.config(text=f"SSIM: {ssim_value:.4f}, BER: {ber_value:.4f}")
        
        # Capacity info
        messagebox.showinfo("Capacity Info", f"Max capacity: {capacity} bits ({capacity // 8} chars). Embedded: {data_index // 8} chars.")
        if data_index < len(secret_message) * 8:
            messagebox.showwarning("Warning", f"Message too long! Only {data_index // 8} characters embedded out of {len(secret_message)}.")
      
        print("Original Message:", secret_message)
        print("Extracted Message:", extracted_message)
        
        display_size = (300, 300)
        original_resized = cv2.resize(image, display_size)
        original_rgb = cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB)
        original_pil = Image.fromarray(original_rgb)
        original_photo = ImageTk.PhotoImage(original_pil)
        self.original_label.config(image=original_photo)
        self.image_references.append(original_photo)
        
        stego_resized = cv2.resize(stego_image, display_size)
        stego_rgb = cv2.cvtColor(stego_resized, cv2.COLOR_BGR2RGB)
        stego_pil = Image.fromarray(stego_rgb)
        stego_photo = ImageTk.PhotoImage(stego_pil)
        self.stego_label.config(image=stego_photo)
        self.image_references.append(stego_photo)
        
        marked_resized = cv2.resize(marked_image, display_size)
        marked_rgb = cv2.cvtColor(marked_resized, cv2.COLOR_BGR2RGB)
        marked_pil = Image.fromarray(marked_rgb)
        marked_photo = ImageTk.PhotoImage(marked_pil)
        self.marked_label.config(image=marked_photo)
        self.image_references.append(marked_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = SteganographyGUI(root)
    root.mainloop()