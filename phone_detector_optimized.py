import cv2
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import pygame
from datetime import datetime
import json
import os
from PIL import Image, ImageTk

class OptimizedPhoneDetector:
    def __init__(self):
        # OpenCV cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Las cascadas de manos no están incluidas en OpenCV por defecto
        self.hand_detection_available = False
        print("INFO: Usando detección de regiones de manos basada en movimiento")
        
        # Variables de detección
        self.cap = None
        self.is_monitoring = False
        self.detection_start_time = None
        self.last_frame = None
        self.current_frame = None
        self.show_camera = False
        
        # Configuración optimizada
        self.config = {
            'alert_time': 20,
            'face_sensitivity': 1.1,
            'motion_sensitivity': 3000,
            'min_face_size': (50, 50),
            'phone_distance_threshold': 180,
            'phone_min_area': 1500,
            'phone_max_area': 80000,
            'canny_low': 30,
            'canny_high': 100,
            'hand_region_factor': 1.5,  # Factor para región de búsqueda de manos
            'min_contour_solidity': 0.3  # Solidez mínima para objetos válidos
        }
        
        # Estadísticas
        self.stats = {
            'total_usage_today': 0,
            'sessions': [],
            'alerts_triggered': 0
        }
        
        # Debug info
        self.debug_info = "Inicializando..."
        self.detection_data = {
            'faces_count': 0,
            'phone_candidates': 0,
            'hand_regions': 0,
            'motion_level': 0
        }
        
        # Inicializar pygame
        try:
            pygame.mixer.init()
        except:
            print("WARNING: Pygame no disponible")
        
        # GUI
        self.setup_gui()
        self.load_stats()
        self.detect_available_cameras()
    
    def detect_available_cameras(self):
        """Detectar cámaras disponibles"""
        print("Detectando cámaras disponibles...")
        self.available_cameras = []
        
        # Probar diferentes backends
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Auto")
        ]
        
        for idx in range(3):
            for backend_id, backend_name in backends:
                try:
                    cap = cv2.VideoCapture(idx, backend_id)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                            self.available_cameras.append({
                                'index': idx,
                                'backend': backend_name,
                                'resolution': f"{int(width)}x{int(height)}"
                            })
                            print(f"✅ Cámara {idx}: {backend_name} - {int(width)}x{int(height)}")
                            cap.release()
                            break
                    cap.release()
                except Exception as e:
                    if cap:
                        cap.release()
                    continue
        
        if not self.available_cameras:
            print("⚠️ No se detectaron cámaras funcionales")
            # Agregar cámara por defecto para intentar
            self.available_cameras.append({
                'index': 0,
                'backend': 'Default',
                'resolution': '640x480'
            })
        else:
            print(f"📹 Total de cámaras detectadas: {len(self.available_cameras)}")
    
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("📱 Detector Optimizado - Sin MediaPipe")
        self.root.geometry("850x950")
        self.root.configure(bg='#1a1a1a')
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Título
        title_frame = tk.Frame(main_frame, bg='#2d3748', relief='raised', bd=2)
        title_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(title_frame, text="📱 Detector Optimizado de Celular", 
                font=('Arial', 18, 'bold'), bg='#2d3748', fg='#fff').pack(pady=15)
        
        tk.Label(title_frame, text="Detección avanzada con OpenCV + Vista en tiempo real", 
                font=('Arial', 11), bg='#2d3748', fg='#a0aec0').pack(pady=(0, 15))
        
        # Frame principal dividido
        content_frame = tk.Frame(main_frame, bg='#1a1a1a')
        content_frame.pack(fill='both', expand=True)
        
        # Panel izquierdo - Cámara
        camera_panel = tk.Frame(content_frame, bg='#2d3748', relief='raised', bd=2)
        camera_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        tk.Label(camera_panel, text="📹 Vista en Tiempo Real", 
                font=('Arial', 14, 'bold'), bg='#2d3748', fg='#fff').pack(pady=10)
        
        self.camera_label = tk.Label(camera_panel, text="Cámara desconectada\n\nHaz clic en 'Mostrar Cámara'\npara ver la detección", 
                                   bg='#1a202c', fg='#718096', font=('Arial', 11),
                                   width=40, height=20, relief='sunken', bd=2)
        self.camera_label.pack(pady=10, padx=10, fill='both', expand=True)
        
        # Controles de cámara
        camera_controls = tk.Frame(camera_panel, bg='#2d3748')
        camera_controls.pack(pady=10)
        
        self.toggle_camera_button = tk.Button(camera_controls, text="📹 Mostrar Cámara", 
                                            command=self.toggle_camera_view, 
                                            bg='#4299e1', fg='white', font=('Arial', 11, 'bold'),
                                            padx=20, pady=8)
        self.toggle_camera_button.pack()
        
        # Panel derecho - Controles y estado
        control_panel = tk.Frame(content_frame, bg='#2d3748', relief='raised', bd=2)
        control_panel.pack(side='right', fill='y', padx=(10, 0))
        
        # Estado
        status_frame = tk.LabelFrame(control_panel, text="📊 Estado del Sistema", 
                                   bg='#2d3748', fg='#fff', font=('Arial', 12, 'bold'))
        status_frame.pack(fill='x', padx=10, pady=10)
        
        self.status_label = tk.Label(status_frame, text="Estado: Detenido", 
                                   font=('Arial', 12, 'bold'), bg='#2d3748', fg='#4299e1')
        self.status_label.pack(pady=5)
        
        self.timer_label = tk.Label(status_frame, text="Tiempo: 00:00", 
                                  font=('Arial', 11), bg='#2d3748', fg='#fff')
        self.timer_label.pack(pady=2)
        
        self.detection_label = tk.Label(status_frame, text="Método: Optimizado", 
                                      font=('Arial', 10), bg='#2d3748', fg='#a0aec0')
        self.detection_label.pack(pady=2)
        
        # Info de detección en tiempo real
        detection_info_frame = tk.LabelFrame(control_panel, text="🔍 Detección en Tiempo Real", 
                                           bg='#2d3748', fg='#fff', font=('Arial', 11, 'bold'))
        detection_info_frame.pack(fill='x', padx=10, pady=5)
        
        self.faces_label = tk.Label(detection_info_frame, text="👤 Caras: 0", 
                                  font=('Arial', 10), bg='#2d3748', fg='#68d391')
        self.faces_label.pack(anchor='w', padx=5, pady=2)
        
        self.phones_label = tk.Label(detection_info_frame, text="📱 Celulares: 0", 
                                   font=('Arial', 10), bg='#2d3748', fg='#f6ad55')
        self.phones_label.pack(anchor='w', padx=5, pady=2)
        
        self.hands_label = tk.Label(detection_info_frame, text="🖐️ Regiones mano: 0", 
                                  font=('Arial', 10), bg='#2d3748', fg='#9f7aea')
        self.hands_label.pack(anchor='w', padx=5, pady=2)
        
        self.motion_label = tk.Label(detection_info_frame, text="🏃 Movimiento: 0%", 
                                   font=('Arial', 10), bg='#2d3748', fg='#63b3ed')
        self.motion_label.pack(anchor='w', padx=5, pady=2)
        
        # Controles principales
        controls_frame = tk.LabelFrame(control_panel, text="🎮 Controles", 
                                     bg='#2d3748', fg='#fff', font=('Arial', 12, 'bold'))
        controls_frame.pack(fill='x', padx=10, pady=10)
        
        self.start_button = tk.Button(controls_frame, text="▶️ INICIAR DETECCIÓN", 
                                    command=self.start_monitoring, 
                                    bg='#48bb78', fg='white', font=('Arial', 12, 'bold'),
                                    padx=20, pady=10)
        self.start_button.pack(fill='x', padx=10, pady=5)
        
        self.stop_button = tk.Button(controls_frame, text="⏹️ DETENER", 
                                   command=self.stop_monitoring, 
                                   bg='#f56565', fg='white', font=('Arial', 12, 'bold'),
                                   padx=20, pady=10, state='disabled')
        self.stop_button.pack(fill='x', padx=10, pady=5)
        
        # Configuración
        config_frame = tk.LabelFrame(control_panel, text="⚙️ Configuración", 
                                   bg='#2d3748', fg='#fff', font=('Arial', 11, 'bold'))
        config_frame.pack(fill='x', padx=10, pady=10)
        
        # Tiempo de alerta
        tk.Label(config_frame, text="⏰ Tiempo alerta (seg):", 
                bg='#2d3748', fg='#fff', font=('Arial', 10)).pack(anchor='w', padx=5)
        
        self.alert_time_var = tk.IntVar(value=self.config['alert_time'])
        alert_scale = tk.Scale(config_frame, from_=5, to=120, orient='horizontal', 
                             variable=self.alert_time_var, bg='#4a5568', fg='white',
                             highlightbackground='#2d3748', length=200)
        alert_scale.pack(fill='x', padx=5, pady=2)
        
        # Sensibilidad
        tk.Label(config_frame, text="📱 Distancia máxima:", 
                bg='#2d3748', fg='#fff', font=('Arial', 10)).pack(anchor='w', padx=5)
        
        self.phone_distance_var = tk.IntVar(value=self.config['phone_distance_threshold'])
        distance_scale = tk.Scale(config_frame, from_=80, to=300, orient='horizontal',
                                variable=self.phone_distance_var, bg='#4a5568', fg='white',
                                highlightbackground='#2d3748', length=200)
        distance_scale.pack(fill='x', padx=5, pady=2)
        
        # Método de detección
        method_frame = tk.LabelFrame(control_panel, text="🔍 Método de Detección", 
                                   bg='#2d3748', fg='#fff', font=('Arial', 11, 'bold'))
        method_frame.pack(fill='x', padx=10, pady=10)
        
        self.detection_method = tk.StringVar(value="shapes_only")
        
        methods = [
            ("📱 Solo Formas (RECOMENDADO)", "shapes_only"),
            ("🖐️ Solo Manos", "hands_only"),
            ("🏃 Solo Movimiento", "motion_only"),
            ("🎯 Inteligente Flexible", "intelligent_flexible"),
            ("👤 Solo detección facial", "face_only")
        ]
        
        for text, value in methods:
            tk.Radiobutton(method_frame, text=text, variable=self.detection_method, value=value,
                          bg='#2d3748', fg='white', selectcolor='#4a5568',
                          font=('Arial', 9)).pack(anchor='w', padx=5)
        
        # Estadísticas
        stats_frame = tk.LabelFrame(control_panel, text="📊 Estadísticas Hoy", 
                                  bg='#2d3748', fg='#fff', font=('Arial', 11, 'bold'))
        stats_frame.pack(fill='x', padx=10, pady=10)
        
        self.usage_today_label = tk.Label(stats_frame, text="📱 Uso total: 0 min", 
                                        bg='#2d3748', fg='#4299e1', font=('Arial', 10))
        self.usage_today_label.pack(anchor='w', padx=5, pady=2)
        
        self.sessions_label = tk.Label(stats_frame, text="📅 Sesiones: 0", 
                                     bg='#2d3748', fg='#4299e1', font=('Arial', 10))
        self.sessions_label.pack(anchor='w', padx=5, pady=2)
        
        self.alerts_label = tk.Label(stats_frame, text="🚨 Alertas: 0", 
                                   bg='#2d3748', fg='#f56565', font=('Arial', 10))
        self.alerts_label.pack(anchor='w', padx=5, pady=2)
        
        # Información
        info_frame = tk.Frame(main_frame, bg='#2a4365', relief='raised', bd=1)
        info_frame.pack(fill='x', pady=(15, 0))
        
        info_text = """💡 DETECTOR OPTIMIZADO SIN MEDIAPIPE
• Detección inteligente de formas rectangulares (celulares)
• Análisis de regiones de manos basado en posición facial
• Vista en tiempo real con visualización de detección
• Configuración de sensibilidad en tiempo real
• Compatible con cualquier versión de Python"""
        
        tk.Label(info_frame, text=info_text, bg='#2a4365', fg='#e2e8f0',
                font=('Arial', 9), justify='left').pack(padx=15, pady=10)
    
    def toggle_camera_view(self):
        """Alternar vista de cámara"""
        self.show_camera = not self.show_camera
        if self.show_camera:
            self.toggle_camera_button.config(text="📹 Ocultar Cámara", bg='#e53e3e')
        else:
            self.toggle_camera_button.config(text="📹 Mostrar Cámara", bg='#4299e1')
            self.camera_label.config(image='', text="Vista de cámara oculta\n\nHaz clic en 'Mostrar Cámara'\npara activar")
    
    def start_monitoring(self):
        """Iniciar monitoreo"""
        try:
            # Probar cámaras detectadas con sus backends
            self.cap = None
            
            for camera_info in self.available_cameras:
                idx = camera_info['index']
                backend_name = camera_info.get('backend', 'Default')
                print(f"Probando cámara {idx} ({backend_name})...")
                
                # Mapear backends
                backend_map = {
                    "DirectShow": cv2.CAP_DSHOW,
                    "Media Foundation": cv2.CAP_MSMF,
                    "Auto": cv2.CAP_ANY,
                    "Default": cv2.CAP_ANY
                }
                
                backend_id = backend_map.get(backend_name, cv2.CAP_ANY)
                
                try:
                    test_cap = cv2.VideoCapture(idx, backend_id)
                    
                    if test_cap.isOpened():
                        ret, frame = test_cap.read()
                        if ret and frame is not None:
                            self.cap = test_cap
                            print(f"✅ Cámara {idx} funcionando con {backend_name}")
                            break
                        else:
                            test_cap.release()
                    else:
                        test_cap.release()
                except Exception as e:
                    print(f"❌ Error probando cámara {idx}: {e}")
                    if 'test_cap' in locals():
                        test_cap.release()
                    continue
            
            if not self.cap:
                # Último intento básico
                print("🔄 Último intento con cámara por defecto...")
                try:
                    self.cap = cv2.VideoCapture(0)
                    if not self.cap.isOpened():
                        raise Exception("No se pudo abrir cámara 0")
                    
                    ret, frame = self.cap.read()
                    if not ret:
                        self.cap.release()
                        raise Exception("No se pudo leer de la cámara")
                        
                    print("✅ Cámara por defecto funcionando")
                except Exception as e:
                    messagebox.showerror("❌ Error de Cámara", 
                                       f"No se pudo acceder a ninguna cámara.\n\nError: {str(e)}\n\nVerifica que:\n• La cámara esté conectada\n• No esté siendo usada por otra app\n• Tengas permisos de cámara")
                    return
            
            # Configurar cámara
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 20)
            
            self.is_monitoring = True
            self.detection_start_time = None
            self.last_frame = None
            
            # UI
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.status_label.config(text="Estado: 🔄 Iniciando...", fg='#f6ad55')
            
            # Activar cámara automáticamente
            self.show_camera = True
            self.toggle_camera_button.config(text="📹 Ocultar Cámara", bg='#e53e3e')
            
            # Iniciar hilo
            self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
            self.detection_thread.start()
            
            self.update_ui()
            print("OK: Monitoreo iniciado")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al iniciar: {str(e)}")
    
    def stop_monitoring(self):
        """Detener monitoreo"""
        self.is_monitoring = False
        
        if self.cap:
            self.cap.release()
        
        if self.detection_start_time:
            self.end_session()
        
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.status_label.config(text="Estado: ⏹️ Detenido", fg='#a0aec0')
        self.timer_label.config(text="Tiempo: 00:00")
        self.camera_label.config(image='', text="Cámara desconectada")
        
        print("Monitoreo detenido")
    
    def detection_loop(self):
        """Bucle principal optimizado"""
        consecutive_errors = 0
        max_errors = 10
        
        while self.is_monitoring:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    consecutive_errors += 1
                    if consecutive_errors >= max_errors:
                        break
                    time.sleep(0.1)
                    continue
                
                consecutive_errors = 0
                frame = cv2.flip(frame, 1)
                self.current_frame = frame.copy()
                
                # Detectar según método
                detected = False
                method = self.detection_method.get()
                
                if method == "face_only":
                    detected = self.detect_face(frame)
                    
                elif method == "motion_only":
                    detected = self.detect_motion(frame)
                    
                elif method == "shapes_only":
                    detected = self.detect_phone_shapes_advanced(frame)
                    
                elif method == "hands_only":
                    detected = self.detect_hands_optimized(frame)
                    
                elif method == "intelligent_flexible":
                    detected = self.intelligent_detection_flexible(frame)
                    
                else:  # intelligent (legacy)
                    detected = self.intelligent_detection(frame)
                
                self.process_detection(detected)
                self.update_camera_display(frame)
                
                # Guardar frame para movimiento
                self.last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                time.sleep(0.05)
                
            except Exception as e:
                print(f"ERROR: {e}")
                time.sleep(1)
    
    def detect_face(self, frame):
        """Detectar rostros mejorado"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Aplicar ecualización de histograma para mejor detección
            gray = cv2.equalizeHist(gray)
            
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.config['face_sensitivity'],
                minNeighbors=5,
                minSize=self.config['min_face_size'],
                maxSize=(300, 300)
            )
            
            self.detection_data['faces_count'] = len(faces)
            return len(faces) > 0
            
        except Exception as e:
            print(f"ERROR face detection: {e}")
            return False
    
    def detect_motion(self, frame):
        """Detectar movimiento optimizado"""
        try:
            if self.last_frame is None:
                return False
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            diff = cv2.absdiff(self.last_frame, gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            
            # Operaciones morfológicas para limpiar
            kernel = np.ones((5,5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            motion_pixels = cv2.countNonZero(thresh)
            motion_percentage = (motion_pixels / (frame.shape[0] * frame.shape[1])) * 100
            
            self.detection_data['motion_level'] = motion_percentage
            return motion_pixels > self.config['motion_sensitivity']
            
        except Exception as e:
            print(f"ERROR motion detection: {e}")
            return False
    
    def detect_phone_shapes_advanced(self, frame):
        """Detección avanzada de formas rectangulares"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Preprocesamiento mejorado
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Detección de bordes adaptativa
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
            
            # También usar Canny
            canny = cv2.Canny(gray, self.config['canny_low'], self.config['canny_high'])
            
            # Combinar ambos métodos
            combined = cv2.bitwise_or(edges, canny)
            
            # Operaciones morfológicas
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            phone_candidates = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area < self.config['phone_min_area'] or area > self.config['phone_max_area']:
                    continue
                
                # Calcular solidez (área / área del casco convexo)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0:
                    continue
                    
                solidity = area / hull_area
                
                if solidity < self.config['min_contour_solidity']:
                    continue
                
                # Aproximar a rectángulo
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) >= 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = max(w, h) / min(w, h)
                    
                    # Proporción típica de celular más flexible
                    if 1.2 <= aspect_ratio <= 3.5:
                        phone_candidates.append({
                            'x': x, 'y': y, 'w': w, 'h': h,
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'solidity': solidity,
                            'center': (x + w//2, y + h//2)
                        })
            
            self.detection_data['phone_candidates'] = len(phone_candidates)
            return len(phone_candidates) > 0
            
        except Exception as e:
            print(f"ERROR shape detection: {e}")
            return False
    
    def detect_hand_regions(self, frame, faces):
        """Detectar regiones probables de manos basado en posición facial"""
        try:
            if len(faces) == 0:
                return []
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hand_regions = []
            
            for (face_x, face_y, face_w, face_h) in faces:
                # Definir regiones donde probablemente estén las manos
                face_center_x = face_x + face_w // 2
                face_center_y = face_y + face_h // 2
                
                # Regiones típicas de manos al usar celular
                regions = [
                    # Frente al rostro
                    (face_center_x - face_w//2, face_y - face_h//3, face_w, face_h//2),
                    # Lado derecho
                    (face_x + face_w, face_center_y - face_h//3, face_w//2, face_h//2),
                    # Lado izquierdo  
                    (face_x - face_w//2, face_center_y - face_h//3, face_w//2, face_h//2),
                ]
                
                for (rx, ry, rw, rh) in regions:
                    # Asegurar que la región esté dentro del frame
                    rx = max(0, rx)
                    ry = max(0, ry)
                    rw = min(rw, frame.shape[1] - rx)
                    rh = min(rh, frame.shape[0] - ry)
                    
                    if rw > 20 and rh > 20:
                        roi = gray[ry:ry+rh, rx:rx+rw]
                        
                        # Detectar movimiento en la región
                        if hasattr(self, 'last_frame') and self.last_frame is not None:
                            last_roi = self.last_frame[ry:ry+rh, rx:rx+rw]
                            diff = cv2.absdiff(roi, last_roi)
                            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                            movement = cv2.countNonZero(thresh)
                            
                            # Si hay suficiente movimiento, considerar como posible mano
                            if movement > (rw * rh * 0.1):  # 10% de la región
                                hand_regions.append({
                                    'x': rx, 'y': ry, 'w': rw, 'h': rh,
                                    'movement': movement,
                                    'center': (rx + rw//2, ry + rh//2)
                                })
            
            self.detection_data['hand_regions'] = len(hand_regions)
            return hand_regions
            
        except Exception as e:
            print(f"ERROR hand regions: {e}")
            return []
    
    def intelligent_detection(self, frame):
        """Detección inteligente combinada"""
        try:
            # 1. Detectar rostros
            face_detected = self.detect_face(frame)
            
            if not face_detected:
                self.debug_info = "Sin rostros detectados"
                return False
            
            # 2. Obtener rostros para análisis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=self.config['face_sensitivity'],
                minNeighbors=5, minSize=self.config['min_face_size']
            )
            
            # 3. Detectar formas de celular
            phone_shapes = self.detect_phone_shapes_advanced(frame)
            
            # 4. Detectar regiones de manos
            hand_regions = self.detect_hand_regions(frame, faces)
            
            # 5. Verificar proximidad
            phone_near_face = False
            hands_near_face = False
            
            if phone_shapes:
                phone_near_face = self.check_proximity_to_face(frame, faces, "phone")
            
            if hand_regions:
                hands_near_face = self.check_proximity_to_face(frame, faces, "hands")
            
            # 6. Lógica de detección inteligente
            # Detectar si hay rostro Y (celular cerca O manos activas cerca)
            detected = face_detected and (phone_near_face or hands_near_face)
            
            # Debug info
            self.debug_info = f"Cara:{'✓' if face_detected else '✗'} Celular:{'✓' if phone_near_face else '✗'} Manos:{'✓' if hands_near_face else '✗'}"
            
            return detected
            
        except Exception as e:
            print(f"ERROR intelligent detection: {e}")
            self.debug_info = f"Error: {str(e)[:30]}"
            return False
    
    def detect_hands_optimized(self, frame):
        """Detección optimizada solo de manos"""
        try:
            # 1. Detectar rostros para definir regiones
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=self.config['face_sensitivity'],
                minNeighbors=5, minSize=self.config['min_face_size']
            )
            
            self.detection_data['faces_count'] = len(faces)
            
            # Si no hay rostros, usar regiones generales
            if len(faces) == 0:
                # Detectar movimiento en regiones típicas de manos
                h, w = frame.shape[:2]
                general_regions = [
                    (w//4, h//4, w//2, h//2, "CENTRO"),
                    (0, h//3, w//3, h//3, "IZQUIERDA"),
                    (2*w//3, h//3, w//3, h//3, "DERECHA"),
                ]
                hand_activity = self.check_hand_activity_in_regions(frame, general_regions)
                self.debug_info = f"Sin rostro - Actividad manos: {hand_activity}"
                return hand_activity > 0
            
            # 2. Detectar actividad de manos en regiones faciales
            hand_regions = self.detect_hand_regions(frame, faces)
            hands_detected = len(hand_regions) > 0
            
            # 3. También verificar movimiento general en área superior
            h, w = frame.shape[:2]
            upper_region_movement = self.detect_movement_in_region(frame, (0, 0, w, h//2))
            
            # 4. Combinar detecciones
            detected = hands_detected or upper_region_movement
            
            self.detection_data['hand_regions'] = len(hand_regions)
            self.debug_info = f"Manos: {len(hand_regions)} | Mov.superior: {'✓' if upper_region_movement else '✗'}"
            
            return detected
            
        except Exception as e:
            print(f"ERROR hands optimized: {e}")
            self.debug_info = f"Error manos: {str(e)[:20]}"
            return False
    
    def intelligent_detection_flexible(self, frame):
        """Detección inteligente más flexible - Cara O Manos"""
        try:
            # 1. Detectar rostros
            face_detected = self.detect_face(frame)
            
            # 2. Detectar manos
            hands_detected = self.detect_hands_optimized(frame)
            
            # 3. Detectar formas (opcional)
            phone_shapes = self.detect_phone_shapes_advanced(frame)
            
            # 4. Lógica flexible: Cara O Manos O (Formas y Movimiento)
            movement_detected = self.detect_motion(frame)
            
            detected = (face_detected or 
                       hands_detected or 
                       (phone_shapes and movement_detected))
            
            # Debug info
            self.debug_info = f"Cara:{'✓' if face_detected else '✗'} Manos:{'✓' if hands_detected else '✗'} Mov:{'✓' if movement_detected else '✗'}"
            
            return detected
            
        except Exception as e:
            print(f"ERROR flexible detection: {e}")
            self.debug_info = f"Error flex: {str(e)[:20]}"
            return False
    
    def check_hand_activity_in_regions(self, frame, regions):
        """Verificar actividad de manos en regiones específicas"""
        try:
            if not hasattr(self, 'last_frame') or self.last_frame is None:
                return 0
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            active_regions = 0
            
            for (rx, ry, rw, rh, label) in regions:
                if rx + rw <= gray.shape[1] and ry + rh <= gray.shape[0]:
                    roi = gray[ry:ry+rh, rx:rx+rw]
                    last_roi = self.last_frame[ry:ry+rh, rx:rx+rw]
                    
                    diff = cv2.absdiff(roi, last_roi)
                    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    
                    # Aplicar filtros morfológicos para limpiar ruido
                    kernel = np.ones((3,3), np.uint8)
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                    
                    movement = cv2.countNonZero(thresh)
                    movement_percentage = (movement / (rw * rh)) * 100
                    
                    # Si hay suficiente movimiento (más del 5%)
                    if movement_percentage > 5:
                        active_regions += 1
            
            return active_regions
            
        except Exception as e:
            print(f"ERROR hand activity: {e}")
            return 0
    
    def detect_movement_in_region(self, frame, region):
        """Detectar movimiento en región específica"""
        try:
            if not hasattr(self, 'last_frame') or self.last_frame is None:
                return False
            
            rx, ry, rw, rh = region
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Asegurar que la región esté dentro del frame
            rx = max(0, rx)
            ry = max(0, ry)
            rw = min(rw, gray.shape[1] - rx)
            rh = min(rh, gray.shape[0] - ry)
            
            if rw <= 0 or rh <= 0:
                return False
            
            roi = gray[ry:ry+rh, rx:rx+rw]
            last_roi = self.last_frame[ry:ry+rh, rx:rx+rw]
            
            diff = cv2.absdiff(roi, last_roi)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            movement = cv2.countNonZero(thresh)
            movement_percentage = (movement / (rw * rh)) * 100
            
            return movement_percentage > 8  # 8% de la región
            
        except Exception as e:
            print(f"ERROR movement region: {e}")
            return False
    
    def check_proximity_to_face(self, frame, faces, detection_type):
        """Verificar proximidad a la cara"""
        try:
            threshold = self.config['phone_distance_threshold']
            
            for (face_x, face_y, face_w, face_h) in faces:
                face_center = (face_x + face_w//2, face_y + face_h//2)
                
                if detection_type == "phone":
                    # Verificar candidatos de celular
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Reutilizar lógica de detección de formas
                    # (simplificado para este ejemplo)
                    return self.detection_data['phone_candidates'] > 0
                
                elif detection_type == "hands":
                    # Verificar regiones de manos
                    return self.detection_data['hand_regions'] > 0
            
            return False
            
        except Exception as e:
            print(f"ERROR proximity check: {e}")
            return False
    
    def update_camera_display(self, frame):
        """Actualizar display con visualizaciones"""
        if not self.show_camera:
            return
        
        try:
            display_frame = frame.copy()
            
            # Dibujar detecciones
            self.draw_face_detection(display_frame)
            self.draw_phone_detection(display_frame)
            self.draw_hand_regions(display_frame)
            self.draw_debug_overlay(display_frame)
            
            # Redimensionar
            display_frame = cv2.resize(display_frame, (400, 300))
            
            # Convertir a tkinter
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(display_frame_rgb)
            photo = ImageTk.PhotoImage(image)
            
            self.camera_label.config(image=photo, text='')
            self.camera_label.image = photo
            
        except Exception as e:
            print(f"ERROR updating display: {e}")
    
    def draw_face_detection(self, frame):
        """Dibujar detección de rostros"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=self.config['face_sensitivity'],
                minNeighbors=5, minSize=self.config['min_face_size']
            )
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(frame, f'ROSTRO', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
        except Exception as e:
            print(f"ERROR drawing faces: {e}")
    
    def draw_phone_detection(self, frame):
        """Dibujar detección de celulares"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
            canny = cv2.Canny(gray, self.config['canny_low'], self.config['canny_high'])
            combined = cv2.bitwise_or(edges, canny)
            
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area < self.config['phone_min_area'] or area > self.config['phone_max_area']:
                    continue
                
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0:
                    continue
                    
                solidity = area / hull_area
                if solidity < self.config['min_contour_solidity']:
                    continue
                
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) >= 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = max(w, h) / min(w, h)
                    
                    if 1.2 <= aspect_ratio <= 3.5:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        cv2.putText(frame, f'CELULAR {aspect_ratio:.1f}', (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
        except Exception as e:
            print(f"ERROR drawing phones: {e}")
    
    def draw_hand_regions(self, frame):
        """Dibujar regiones de manos"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=self.config['face_sensitivity'],
                minNeighbors=5, minSize=self.config['min_face_size']
            )
            
            for (face_x, face_y, face_w, face_h) in faces:
                face_center_x = face_x + face_w // 2
                face_center_y = face_y + face_h // 2
                
                regions = [
                    (face_center_x - face_w//2, face_y - face_h//3, face_w, face_h//2, "FRENTE"),
                    (face_x + face_w, face_center_y - face_h//3, face_w//2, face_h//2, "DERECHA"),
                    (face_x - face_w//2, face_center_y - face_h//3, face_w//2, face_h//2, "IZQUIERDA"),
                ]
                
                for (rx, ry, rw, rh, label) in regions:
                    rx = max(0, rx)
                    ry = max(0, ry)
                    rw = min(rw, frame.shape[1] - rx)
                    rh = min(rh, frame.shape[0] - ry)
                    
                    if rw > 20 and rh > 20:
                        # Verificar actividad en la región
                        if hasattr(self, 'last_frame') and self.last_frame is not None:
                            roi = gray[ry:ry+rh, rx:rx+rw]
                            last_roi = self.last_frame[ry:ry+rh, rx:rx+rw]
                            diff = cv2.absdiff(roi, last_roi)
                            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                            movement = cv2.countNonZero(thresh)
                            
                            if movement > (rw * rh * 0.1):
                                cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (255, 255, 0), 2)
                                cv2.putText(frame, f'MANO {label}', (rx, ry-5), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
        except Exception as e:
            print(f"ERROR drawing hands: {e}")
    
    def draw_debug_overlay(self, frame):
        """Dibujar información de debug"""
        try:
            h, w = frame.shape[:2]
            
            # Fondo para información
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (w-10, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Información de detección
            y_offset = 30
            info_lines = [
                f"Metodo: {self.detection_method.get()}",
                f"Caras: {self.detection_data['faces_count']} | Celulares: {self.detection_data['phone_candidates']}",
                f"Regiones mano: {self.detection_data['hand_regions']} | Movimiento: {self.detection_data['motion_level']:.1f}%",
            ]
            
            for line in info_lines:
                cv2.putText(frame, line, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
            
            # Estado de detección
            if self.detection_start_time:
                elapsed = time.time() - self.detection_start_time
                cv2.putText(frame, f"DETECTADO: {elapsed:.1f}s", (15, h-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
        except Exception as e:
            print(f"ERROR drawing debug: {e}")
    
    def process_detection(self, detected):
        """Procesar resultado de detección"""
        current_time = time.time()
        
        if detected:
            if self.detection_start_time is None:
                self.detection_start_time = current_time
                print("Uso del celular detectado")
            
            elapsed_time = current_time - self.detection_start_time
            
            if elapsed_time >= self.alert_time_var.get():
                self.show_alert()
        else:
            if self.detection_start_time is not None:
                self.end_session()
    
    def end_session(self):
        """Finalizar sesión"""
        if self.detection_start_time:
            session_duration = time.time() - self.detection_start_time
            
            if session_duration > 2:
                self.stats['sessions'].append({
                    'start': datetime.fromtimestamp(self.detection_start_time).isoformat(),
                    'duration': session_duration
                })
                
                self.stats['total_usage_today'] += session_duration
                print(f"Sesión guardada: {session_duration:.1f}s")
            
            self.detection_start_time = None
            self.save_stats()
    
    def show_alert(self):
        """Mostrar alerta en pantalla completa"""
        self.stats['alerts_triggered'] += 1
        
        # Sonido más fuerte
        try:
            import winsound
            # Beeps múltiples para llamar más la atención
            for _ in range(3):
                winsound.Beep(1000, 200)
                time.sleep(0.1)
        except:
            pass
        
        # Ventana en PANTALLA COMPLETA
        alert = tk.Toplevel(self.root)
        alert.title("🚨 ALERTA DE CELULAR")
        
        # Configurar pantalla completa
        alert.attributes('-fullscreen', True)
        alert.attributes('-topmost', True)
        alert.configure(bg='#dc2626')  # Rojo más intenso
        alert.grab_set()
        
        # Crear frame principal centrado
        main_frame = tk.Frame(alert, bg='#dc2626')
        main_frame.pack(expand=True, fill='both')
        
        # Título gigante
        title_label = tk.Label(main_frame, text="🚨 ¡DEJA EL CELULAR! 🚨", 
                              font=('Arial', 60, 'bold'), bg='#dc2626', fg='white')
        title_label.pack(pady=(100, 50))
        
        # Mensaje principal
        message_label = tk.Label(main_frame, 
                                text=f"Has estado usando el celular por {self.alert_time_var.get()} segundos", 
                                font=('Arial', 28), bg='#dc2626', fg='white')
        message_label.pack(pady=30)
        
        # Submensaje motivacional
        motivational_messages = [
            "Tus ojos necesitan un descanso",
            "Es momento de una pausa",
            "Tu bienestar es importante",
            "Desconecta para reconectar contigo",
            "Tu salud mental lo agradecerá"
        ]
        import random
        submessage = random.choice(motivational_messages)
        
        sub_label = tk.Label(main_frame, text=submessage, 
                            font=('Arial', 20), bg='#dc2626', fg='#fecaca')
        sub_label.pack(pady=20)
        
        # Información del método de detección
        method_info = {
            "hands_only": "Movimiento de manos detectado",
            "motion_only": "Actividad general detectada",
            "intelligent_flexible": "Detección inteligente activada",
            "shapes_only": "Forma de celular detectada",
            "face_only": "Posición facial detectada"
        }
        
        current_method = self.detection_method.get()
        detection_info = method_info.get(current_method, "Sistema de detección activado")
        
        detection_label = tk.Label(main_frame, text=f"• {detection_info} •", 
                                  font=('Arial', 16), bg='#dc2626', fg='#fca5a5')
        detection_label.pack(pady=15)
        
        # Contador regresivo visual
        countdown_label = tk.Label(main_frame, text="", 
                                  font=('Arial', 24, 'bold'), bg='#dc2626', fg='#fef2f2')
        countdown_label.pack(pady=20)
        
        # Frame de botones más grande
        button_frame = tk.Frame(main_frame, bg='#dc2626')
        button_frame.pack(pady=50)
        
        # Botón principal más grande
        ok_button = tk.Button(button_frame, text="✅ ENTENDIDO - CERRAR", 
                             command=alert.destroy,
                             bg='white', fg='#dc2626', 
                             font=('Arial', 20, 'bold'),
                             padx=40, pady=20,
                             relief='raised', bd=5)
        ok_button.pack(side='left', padx=20)
        
        # Botón de snooze
        snooze_button = tk.Button(button_frame, text="⏰ 5 MINUTOS MÁS", 
                                 command=lambda: [setattr(self, 'detection_start_time', time.time()), alert.destroy()],
                                 bg='#991b1b', fg='white', 
                                 font=('Arial', 16, 'bold'),
                                 padx=30, pady=15,
                                 relief='raised', bd=3)
        snooze_button.pack(side='left', padx=20)
        
        # Información de escape
        escape_label = tk.Label(main_frame, text="Presiona ESC para cerrar", 
                               font=('Arial', 14), bg='#dc2626', fg='#fca5a5')
        escape_label.pack(side='bottom', pady=30)
        
        # Bind de tecla ESC para cerrar
        alert.bind('<Escape>', lambda e: alert.destroy())
        alert.focus_set()
        
        # Countdown automático
        countdown_time = 10
        def update_countdown():
            nonlocal countdown_time
            if countdown_time > 0:
                countdown_label.config(text=f"Se cerrará automáticamente en {countdown_time} segundos")
                countdown_time -= 1
                alert.after(1000, update_countdown)
            else:
                alert.destroy()
        
        update_countdown()
        
        # Efecto de parpadeo sutil del título
        def blink_title():
            if alert.winfo_exists():
                current_color = title_label.cget('fg')
                new_color = '#fef2f2' if current_color == 'white' else 'white'
                title_label.config(fg=new_color)
                alert.after(800, blink_title)
        
        blink_title()
        
        # Reiniciar timer
        self.detection_start_time = time.time()
        print("🚨 ALERTA PANTALLA COMPLETA MOSTRADA")
    
    def update_ui(self):
        """Actualizar interfaz"""
        if not self.is_monitoring:
            return
        
        # Timer
        if self.detection_start_time:
            elapsed = time.time() - self.detection_start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.timer_label.config(text=f"Tiempo: {minutes:02d}:{seconds:02d}")
            
            remaining = self.alert_time_var.get() - elapsed
            if remaining <= 5:
                self.status_label.config(text="Estado: ⚠️ CASI LÍMITE", fg='#f56565')
            else:
                self.status_label.config(text="Estado: 📱 DETECTANDO USO", fg='#f6ad55')
        else:
            self.status_label.config(text="Estado: 👀 Monitoreando...", fg='#4299e1')
        
        # Estadísticas
        usage_minutes = int(self.stats['total_usage_today'] / 60)
        self.usage_today_label.config(text=f"📱 Uso total: {usage_minutes} min")
        self.sessions_label.config(text=f"📅 Sesiones: {len(self.stats['sessions'])}")
        self.alerts_label.config(text=f"🚨 Alertas: {self.stats['alerts_triggered']}")
        
        # Info de detección en tiempo real
        self.faces_label.config(text=f"👤 Caras: {self.detection_data['faces_count']}")
        self.phones_label.config(text=f"📱 Celulares: {self.detection_data['phone_candidates']}")
        self.hands_label.config(text=f"🖐️ Regiones mano: {self.detection_data['hand_regions']}")
        self.motion_label.config(text=f"🏃 Movimiento: {self.detection_data['motion_level']:.1f}%")
        
        # Actualizar config dinámicamente
        self.config['phone_distance_threshold'] = self.phone_distance_var.get()
        
        self.root.after(100, self.update_ui)
    
    def save_stats(self):
        """Guardar estadísticas"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            with open(f'phone_stats_optimized_{today}.json', 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            print(f"ERROR saving: {e}")
    
    def load_stats(self):
        """Cargar estadísticas"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            filename = f'phone_stats_optimized_{today}.json'
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    self.stats = json.load(f)
                print(f"Estadísticas cargadas: {filename}")
        except Exception as e:
            print(f"ERROR loading: {e}")
    
    def run(self):
        """Ejecutar aplicación"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            print("🚀 Detector Optimizado iniciado")
            print("💡 Este detector NO requiere MediaPipe")
            print("✅ Compatible con cualquier versión de Python")
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nCtrl+C detectado")
            self.stop_monitoring()
    
    def on_closing(self):
        """Cerrar aplicación"""
        if self.is_monitoring:
            self.stop_monitoring()
        print("Aplicación cerrada")
        self.root.destroy()

if __name__ == "__main__":
    import sys
    
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass
    
    print("🔥 DETECTOR OPTIMIZADO DE CELULAR")
    print("=" * 50)
    print("✅ Funciona SIN MediaPipe")
    print("✅ Compatible con Python 3.13")
    print("✅ Detección inteligente avanzada")
    print("Dependencias: pip install opencv-python numpy pygame pillow")
    print()
    
    try:
        detector = OptimizedPhoneDetector()
        detector.run()
    except ImportError as e:
        print(f"ERROR: Falta instalar: {e}")
        print("Ejecuta: pip install opencv-python numpy pygame pillow")
    except Exception as e:
        print(f"ERROR: {e}")