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

class AdvancedPhoneDetector:
    def __init__(self):
        # OpenCV para detección facial (Haar Cascades)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Variables de detección
        self.cap = None
        self.is_monitoring = False
        self.detection_start_time = None
        self.last_frame = None
        self.current_frame = None
        self.show_camera = False
        
        # Variables para detección avanzada
        self.phone_detection_enabled = True
        self.hand_tracking_enabled = False  # Para MediaPipe cuando esté disponible
        
        # Configuración
        self.config = {
            'alert_time': 20,
            'face_sensitivity': 1.1,
            'motion_sensitivity': 5000,
            'min_face_size': (30, 30),
            'phone_distance_threshold': 150,
            'phone_min_area': 2000,
            'phone_max_area': 50000,
            'canny_low': 50,
            'canny_high': 150,
            'hand_confidence_threshold': 0.7
        }
        
        # Estadísticas
        self.stats = {
            'total_usage_today': 0,
            'sessions': [],
            'alerts_triggered': 0
        }
        
        # Debug info
        self.debug_info = "Inicializando..."
        self.detection_data = {}
        
        # Inicializar pygame para sonidos
        try:
            pygame.mixer.init()
        except:
            print("WARNING: Pygame no disponible - sin sonidos")
        
        # Crear GUI
        self.setup_gui()
        self.load_stats()
        
        # Información de cámaras disponibles
        self.detect_available_cameras()
        
        # Intentar importar MediaPipe
        self.init_mediapipe()
    
    def init_mediapipe(self):
        """Inicializar MediaPipe si está disponible"""
        try:
            import mediapipe as mp
            self.mp = mp
            self.mp_hands = mp.solutions.hands
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            
            # Inicializar detectores
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.hand_tracking_enabled = True
            print("OK: MediaPipe inicializado - detección de manos habilitada")
            
        except ImportError:
            print("INFO: MediaPipe no disponible - usando detección básica")
            self.hand_tracking_enabled = False
            self.mp = None
    
    def detect_available_cameras(self):
        """Detectar cámaras disponibles y sus backends"""
        print("Detectando cámaras disponibles...")
        available_cameras = []
        
        # Probar diferentes backends
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_ANY, "Auto")
        ]
        
        for idx in range(3):  # Probar hasta 3 cámaras
            for backend_id, backend_name in backends:
                try:
                    cap = cv2.VideoCapture(idx, backend_id)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            available_cameras.append({
                                'index': idx,
                                'backend': backend_name,
                                'resolution': f"{int(width)}x{int(height)}",
                                'fps': int(fps)
                            })
                            print(f"OK: Cámara {idx}: {backend_name} - {int(width)}x{int(height)} @ {int(fps)}fps")
                            cap.release()
                            break
                    cap.release()
                except:
                    continue
        
        if not available_cameras:
            print("WARNING: No se detectaron cámaras funcionales")
        else:
            print(f"Total de cámaras detectadas: {len(available_cameras)}")
            
        self.available_cameras = available_cameras
        
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("📱 Detector Avanzado de Uso de Celular")
        self.root.geometry("800x900")
        self.root.configure(bg='#2c3e50')
        
        # Estilo
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame principal con scroll
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Título
        title_frame = tk.Frame(main_frame, bg='#34495e', relief='raised', bd=2)
        title_frame.pack(fill='x', pady=(0, 10))
        
        title_label = tk.Label(title_frame, text="📱 Detector Avanzado de Uso de Celular", 
                              font=('Arial', 16, 'bold'), bg='#34495e', fg='white')
        title_label.pack(pady=10)
        
        subtitle_label = tk.Label(title_frame, text="Con Vista Previa y Detección Inteligente", 
                                 font=('Arial', 10), bg='#34495e', fg='#bdc3c7')
        subtitle_label.pack(pady=(0, 10))
        
        # Frame para cámara y controles (lado a lado)
        camera_controls_frame = tk.Frame(main_frame, bg='#2c3e50')
        camera_controls_frame.pack(fill='x', pady=(0, 10))
        
        # Vista previa de cámara
        camera_frame = tk.Frame(camera_controls_frame, bg='#34495e', relief='raised', bd=2)
        camera_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        tk.Label(camera_frame, text="📹 Vista Previa", font=('Arial', 12, 'bold'), 
                bg='#34495e', fg='white').pack(pady=5)
        
        self.camera_label = tk.Label(camera_frame, text="Cámara desconectada", 
                                   bg='#2c3e50', fg='#7f8c8d', font=('Arial', 10))
        self.camera_label.pack(pady=10, padx=10, fill='both', expand=True)
        
        # Botón para mostrar/ocultar cámara
        self.toggle_camera_button = tk.Button(camera_frame, text="📹 Mostrar Cámara", 
                                            command=self.toggle_camera_view, bg='#3498db', fg='white',
                                            font=('Arial', 10, 'bold'))
        self.toggle_camera_button.pack(pady=5)
        
        # Estado y controles
        status_controls_frame = tk.Frame(camera_controls_frame, bg='#34495e', relief='raised', bd=2)
        status_controls_frame.pack(side='right', fill='y', padx=(5, 0))
        
        tk.Label(status_controls_frame, text="📊 Estado", font=('Arial', 12, 'bold'), 
                bg='#34495e', fg='white').pack(pady=5)
        
        self.status_label = tk.Label(status_controls_frame, text="Estado: Detenido", 
                                   font=('Arial', 11, 'bold'), bg='#34495e', fg='#3498db')
        self.status_label.pack(pady=5, padx=10)
        
        self.timer_label = tk.Label(status_controls_frame, text="Tiempo: 00:00", 
                                  font=('Arial', 10), bg='#34495e', fg='white')
        self.timer_label.pack(pady=2, padx=10)
        
        self.detection_label = tk.Label(status_controls_frame, text="Método: Inteligente", 
                                      font=('Arial', 9), bg='#34495e', fg='#95a5a6')
        self.detection_label.pack(pady=2, padx=10)
        
        self.debug_label = tk.Label(status_controls_frame, text="Debug: Inicializando...", 
                                  font=('Arial', 8), bg='#34495e', fg='#f39c12')
        self.debug_label.pack(pady=2, padx=10)
        
        # Controles principales
        controls_frame = tk.Frame(status_controls_frame, bg='#34495e')
        controls_frame.pack(pady=15, padx=10)
        
        self.start_button = tk.Button(controls_frame, text="▶️ Iniciar", 
                                    command=self.start_monitoring, bg='#27ae60', fg='white',
                                    font=('Arial', 11, 'bold'), width=10)
        self.start_button.pack(pady=2)
        
        self.stop_button = tk.Button(controls_frame, text="⏹️ Detener", 
                                   command=self.stop_monitoring, bg='#e74c3c', fg='white',
                                   font=('Arial', 11, 'bold'), width=10, state='disabled')
        self.stop_button.pack(pady=2)
        
        # Configuración
        config_frame = tk.LabelFrame(main_frame, text="⚙️ Configuración", bg='#2c3e50', fg='white',
                                   font=('Arial', 12, 'bold'))
        config_frame.pack(fill='x', pady=10)
        
        # Configuración en dos columnas
        config_left = tk.Frame(config_frame, bg='#2c3e50')
        config_left.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        config_right = tk.Frame(config_frame, bg='#2c3e50')
        config_right.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        # Tiempo de alerta (izquierda)
        tk.Label(config_left, text="⏰ Tiempo de alerta:", bg='#2c3e50', fg='white', 
                font=('Arial', 10, 'bold')).pack(anchor='w')
        
        self.alert_time_var = tk.IntVar(value=self.config['alert_time'])
        alert_scale = tk.Scale(config_left, from_=5, to=120, orient='horizontal', 
                             variable=self.alert_time_var, bg='#34495e', fg='white',
                             highlightbackground='#2c3e50', length=250)
        alert_scale.pack(fill='x', pady=2)
        
        # Sensibilidad de detección (derecha)
        tk.Label(config_right, text="📱 Sensibilidad celular:", bg='#2c3e50', fg='white',
                font=('Arial', 10, 'bold')).pack(anchor='w')
        
        self.phone_distance_var = tk.IntVar(value=self.config['phone_distance_threshold'])
        phone_distance_scale = tk.Scale(config_right, from_=50, to=300, orient='horizontal',
                                      variable=self.phone_distance_var, bg='#34495e', fg='white',
                                      highlightbackground='#2c3e50', length=250,
                                      label="Distancia máxima")
        phone_distance_scale.pack(fill='x', pady=2)
        
        # Método de detección
        method_frame = tk.Frame(main_frame, bg='#2c3e50')
        method_frame.pack(fill='x', pady=5)
        
        tk.Label(method_frame, text="🔍 Método de detección:", bg='#2c3e50', fg='white',
                font=('Arial', 12, 'bold')).pack(anchor='w')
        
        self.detection_method = tk.StringVar(value="advanced_hybrid")
        
        method_options = [
            ("🎯 Detección Avanzada (RECOMENDADO)", "advanced_hybrid"),
            ("📱 Solo formas rectangulares", "shape_detection"),
            ("👤 Solo detección facial", "face_only"),
            ("🏃 Solo detección de movimiento", "motion_only")
        ]
        
        # Agregar opción de MediaPipe si está disponible
        if self.hand_tracking_enabled:
            method_options.insert(1, ("🖐️ Detección de manos (MediaPipe)", "mediapipe_hands"))
        
        for text, value in method_options:
            tk.Radiobutton(method_frame, text=text, variable=self.detection_method, value=value,
                          bg='#2c3e50', fg='white', selectcolor='#34495e',
                          font=('Arial', 10)).pack(anchor='w', padx=10)
        
        # Estadísticas
        stats_frame = tk.LabelFrame(main_frame, text="📊 Estadísticas de Hoy", bg='#2c3e50', fg='white',
                                  font=('Arial', 12, 'bold'))
        stats_frame.pack(fill='x', pady=10)
        
        stats_inner = tk.Frame(stats_frame, bg='#2c3e50')
        stats_inner.pack(fill='x', padx=10, pady=5)
        
        self.usage_today_label = tk.Label(stats_inner, text="📱 Uso total: 0 min", bg='#2c3e50', fg='#3498db',
                                        font=('Arial', 11))
        self.usage_today_label.pack(anchor='w', pady=2)
        
        self.sessions_label = tk.Label(stats_inner, text="📅 Sesiones: 0", bg='#2c3e50', fg='#3498db',
                                     font=('Arial', 11))
        self.sessions_label.pack(anchor='w', pady=2)
        
        self.alerts_label = tk.Label(stats_inner, text="🚨 Alertas: 0", bg='#2c3e50', fg='#e74c3c',
                                   font=('Arial', 11))
        self.alerts_label.pack(anchor='w', pady=2)
        
        # Información
        info_frame = tk.Frame(main_frame, bg='#34495e', relief='sunken', bd=1)
        info_frame.pack(fill='x', pady=5)
        
        info_text = """💡 NUEVO: Detector Avanzado
• Vista previa de cámara en tiempo real
• Detección inteligente de formas y posiciones
• Visualización de detección con marcos y puntos
• Algoritmos optimizados para mayor precisión"""
        
        info_label = tk.Label(info_frame, text=info_text, bg='#34495e', fg='#bdc3c7',
                            font=('Arial', 9), justify='left')
        info_label.pack(padx=10, pady=5)
    
    def toggle_camera_view(self):
        """Alternar vista de cámara"""
        self.show_camera = not self.show_camera
        if self.show_camera:
            self.toggle_camera_button.config(text="📹 Ocultar Cámara")
        else:
            self.toggle_camera_button.config(text="📹 Mostrar Cámara")
            self.camera_label.config(image='', text="Vista de cámara oculta")
    
    def update_camera_display(self, frame):
        """Actualizar display de cámara con visualizaciones"""
        if not self.show_camera:
            return
            
        try:
            # Crear una copia del frame para dibujar
            display_frame = frame.copy()
            
            # Agregar visualizaciones según el método de detección
            method = self.detection_method.get()
            
            if method == "advanced_hybrid" or method == "shape_detection":
                self.draw_shape_detection(display_frame)
            
            if method == "face_only" or method == "advanced_hybrid":
                self.draw_face_detection(display_frame)
                
            if method == "mediapipe_hands" and self.hand_tracking_enabled:
                self.draw_hand_detection(display_frame)
            
            # Agregar información de debug
            self.draw_debug_info(display_frame)
            
            # Redimensionar para display
            display_frame = cv2.resize(display_frame, (320, 240))
            
            # Convertir a formato tkinter
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(display_frame_rgb)
            photo = ImageTk.PhotoImage(image)
            
            # Actualizar label
            self.camera_label.config(image=photo, text='')
            self.camera_label.image = photo  # Mantener referencia
            
        except Exception as e:
            print(f"ERROR: Error actualizando display: {e}")
    
    def draw_face_detection(self, frame):
        """Dibujar detección de rostros"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.config['face_sensitivity'],
                minNeighbors=5,
                minSize=self.config['min_face_size']
            )
            
            for (x, y, w, h) in faces:
                # Dibujar rectángulo verde para rostros
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, 'ROSTRO', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
        except Exception as e:
            print(f"ERROR: Error dibujando rostros: {e}")
    
    def draw_shape_detection(self, frame):
        """Dibujar detección de formas rectangulares"""
        try:
            phone_candidates = self.detect_phone_shapes(frame)
            
            for phone in phone_candidates:
                x, y, w, h = phone['x'], phone['y'], phone['w'], phone['h']
                # Dibujar rectángulo azul para posibles celulares
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f'CELULAR?', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, f'{phone["aspect_ratio"]:.1f}', (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
        except Exception as e:
            print(f"ERROR: Error dibujando formas: {e}")
    
    def draw_hand_detection(self, frame):
        """Dibujar detección de manos con MediaPipe"""
        if not self.hand_tracking_enabled:
            return
            
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dibujar landmarks de manos
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Calcular centro de la mano
                    landmarks = hand_landmarks.landmark
                    h, w, _ = frame.shape
                    
                    hand_center_x = int(landmarks[9].x * w)  # Punto central de la mano
                    hand_center_y = int(landmarks[9].y * h)
                    
                    cv2.circle(frame, (hand_center_x, hand_center_y), 5, (255, 255, 0), -1)
                    cv2.putText(frame, 'MANO', (hand_center_x-20, hand_center_y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    
        except Exception as e:
            print(f"ERROR: Error dibujando manos: {e}")
    
    def draw_debug_info(self, frame):
        """Dibujar información de debug"""
        try:
            h, w = frame.shape[:2]
            
            # Fondo para texto
            cv2.rectangle(frame, (10, 10), (w-10, 80), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (w-10, 80), (255, 255, 255), 2)
            
            # Información de debug
            cv2.putText(frame, f"Metodo: {self.detection_method.get()}", (15, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Debug: {self.debug_info}", (15, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Estado de detección
            if self.detection_start_time:
                elapsed = time.time() - self.detection_start_time
                cv2.putText(frame, f"DETECTADO: {elapsed:.1f}s", (15, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
        except Exception as e:
            print(f"ERROR: Error dibujando debug: {e}")
    
    def start_monitoring(self):
        """Iniciar monitoreo"""
        try:
            # Verificar cámara usando las cámaras detectadas
            self.cap = None
            
            # Primero probar las cámaras detectadas
            if hasattr(self, 'available_cameras') and self.available_cameras:
                for camera_info in self.available_cameras:
                    idx = camera_info['index']
                    backend = camera_info['backend']
                    print(f"Probando cámara {idx} ({backend})...")
                    
                    # Usar el backend específico si es posible
                    backend_map = {
                        "DirectShow": cv2.CAP_DSHOW,
                        "Media Foundation": cv2.CAP_MSMF,
                        "V4L2": cv2.CAP_V4L2,
                        "Auto": cv2.CAP_ANY
                    }
                    
                    backend_id = backend_map.get(backend, cv2.CAP_ANY)
                    test_cap = cv2.VideoCapture(idx, backend_id)
                    
                    if test_cap.isOpened():
                        ret, frame = test_cap.read()
                        if ret:
                            self.cap = test_cap
                            print(f"OK: Cámara {idx} funcionando con {backend}")
                            break
                        else:
                            test_cap.release()
                    else:
                        test_cap.release()
            
            # Si no funcionó, probar método tradicional
            if not self.cap:
                print("Probando método tradicional...")
                for idx in [0, 1, 2]:
                    print(f"Probando cámara {idx}...")
                    test_cap = cv2.VideoCapture(idx)
                    if test_cap.isOpened():
                        ret, frame = test_cap.read()
                        if ret:
                            self.cap = test_cap
                            print(f"OK: Cámara {idx} funcionando")
                            break
                        else:
                            test_cap.release()
                    else:
                        test_cap.release()
            
            if not self.cap or not self.cap.isOpened():
                error_msg = """❌ No se pudo acceder a ninguna cámara

Verifica que:
• La cámara esté conectada
• No esté siendo usada por otra aplicación
• Tengas permisos de cámara
• Windows Camera esté cerrada
• Skype/Teams/Zoom estén cerrados

Soluciones comunes:
• Reinicia la aplicación
• Desconecta y conecta la cámara
• Reinicia el servicio de Windows Camera"""
                messagebox.showerror("❌ Error de Cámara", error_msg)
                return
            
            # Configurar cámara
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 20)
                
                current_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                current_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                backend = self.cap.getBackendName()
                
                print(f"Cámara configurada: {current_width}x{current_height} ({backend})")
                
            except Exception as config_error:
                print(f"WARNING: Error configurando cámara: {config_error}")
            
            self.is_monitoring = True
            self.detection_start_time = None
            self.last_frame = None
            
            # Actualizar UI
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.status_label.config(text="Estado: 🔄 Iniciando...", fg='#f39c12')
            
            # Habilitar vista de cámara automáticamente
            self.show_camera = True
            self.toggle_camera_button.config(text="📹 Ocultar Cámara")
            
            # Iniciar hilo de detección
            self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
            self.detection_thread.start()
            
            # Iniciar actualización de UI
            self.update_ui()
            
            print("OK: Monitoreo iniciado")
            
        except Exception as e:
            messagebox.showerror("❌ Error", f"Error al iniciar monitoreo:\n{str(e)}")
    
    def stop_monitoring(self):
        """Detener monitoreo"""
        self.is_monitoring = False
        
        if self.cap:
            self.cap.release()
        
        if self.detection_start_time:
            self.end_session()
        
        # Actualizar UI
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.status_label.config(text="Estado: ⏹️ Detenido", fg='#95a5a6')
        self.timer_label.config(text="Tiempo: 00:00")
        self.camera_label.config(image='', text="Cámara desconectada")
        
        print("Monitoreo detenido")
    
    def detection_loop(self):
        """Bucle principal de detección"""
        consecutive_errors = 0
        max_errors = 10
        
        while self.is_monitoring:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    consecutive_errors += 1
                    print(f"ERROR: Error leyendo frame ({consecutive_errors}/{max_errors})")
                    
                    if consecutive_errors >= max_errors:
                        print("ERROR CRITICO: Demasiados errores consecutivos")
                        self.show_camera_error()
                        break
                    
                    time.sleep(0.1)
                    continue
                
                # Resetear contador de errores
                consecutive_errors = 0
                
                # Voltear horizontalmente para efecto espejo
                frame = cv2.flip(frame, 1)
                self.current_frame = frame.copy()
                
                # Detectar según método seleccionado
                detected = False
                method = self.detection_method.get()
                
                if method == "face_only":
                    detected = self.detect_face(frame)
                    self.debug_info = f"Rostros: {'✓' if detected else '✗'}"
                    
                elif method == "motion_only":
                    detected = self.detect_motion(frame)
                    self.debug_info = f"Movimiento: {'✓' if detected else '✗'}"
                    
                elif method == "shape_detection":
                    detected = self.detect_phone_near_face(frame)
                    self.debug_info = f"Forma celular: {'✓' if detected else '✗'}"
                    
                elif method == "mediapipe_hands" and self.hand_tracking_enabled:
                    detected = self.detect_hands_near_face(frame)
                    
                elif method == "advanced_hybrid":
                    # Método más avanzado
                    face_detected = self.detect_face(frame)
                    
                    if self.hand_tracking_enabled:
                        hands_detected = self.detect_hands_near_face(frame)
                        phone_detected = self.detect_phone_near_face(frame)
                        # Detectar si hay rostro Y (manos cerca O forma de celular)
                        detected = face_detected and (hands_detected or phone_detected)
                        self.debug_info = f"Cara:{face_detected} Manos:{hands_detected} Forma:{phone_detected}"
                    else:
                        phone_detected = self.detect_phone_near_face(frame)
                        detected = face_detected and phone_detected
                        self.debug_info = f"Cara: {'✓' if face_detected else '✗'}, Forma: {'✓' if phone_detected else '✗'}"
                
                # Procesar detección
                self.process_detection(detected)
                
                # Actualizar display de cámara
                self.update_camera_display(frame)
                
                # Guardar frame para comparación de movimiento
                self.last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                time.sleep(0.05)  # 20 FPS
                
            except Exception as e:
                print(f"ERROR: Error en bucle de detección: {e}")
                time.sleep(1)
    
    def detect_face(self, frame):
        """Detectar rostros usando Haar Cascades"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.config['face_sensitivity'],
                minNeighbors=5,
                minSize=self.config['min_face_size']
            )
            
            return len(faces) > 0
            
        except Exception as e:
            print(f"ERROR: Error en detección facial: {e}")
            return False
    
    def detect_motion(self, frame):
        """Detectar movimiento comparando frames"""
        try:
            if self.last_frame is None:
                return False
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calcular diferencia absoluta
            diff = cv2.absdiff(self.last_frame, gray)
            
            # Aplicar umbral
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            # Contar píxeles que cambiaron
            motion_pixels = cv2.countNonZero(thresh)
            
            return motion_pixels > self.config['motion_sensitivity']
            
        except Exception as e:
            print(f"ERROR: Error en detección de movimiento: {e}")
            return False
    
    def detect_phone_shapes(self, frame):
        """Detectar formas rectangulares que podrían ser celulares"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Aplicar filtro gaussiano para reducir ruido
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Detectar bordes
            edges = cv2.Canny(blurred, self.config['canny_low'], self.config['canny_high'])
            
            # Encontrar contornos
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            phone_candidates = []
            
            for contour in contours:
                # Calcular área del contorno
                area = cv2.contourArea(contour)
                
                # Filtrar por área
                if area < self.config['phone_min_area'] or area > self.config['phone_max_area']:
                    continue
                
                # Aproximar contorno a polígono
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Buscar formas rectangulares (4 vértices)
                if len(approx) >= 4:
                    # Calcular rectángulo delimitador
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calcular proporción (celulares suelen ser 16:9 o 18:9)
                    aspect_ratio = max(w, h) / min(w, h)
                    
                    # Filtrar por proporción típica de celular (1.4 a 2.8)
                    if 1.4 <= aspect_ratio <= 2.8:
                        phone_candidates.append({
                            'x': x, 'y': y, 'w': w, 'h': h,
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'center': (x + w//2, y + h//2)
                        })
            
            return phone_candidates
            
        except Exception as e:
            print(f"ERROR: Error en detección de formas: {e}")
            return []
    
    def detect_phone_near_face(self, frame):
        """Detectar si hay un objeto similar a celular cerca de la cara"""
        try:
            # Detectar rostros
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.config['face_sensitivity'],
                minNeighbors=5,
                minSize=self.config['min_face_size']
            )
            
            if len(faces) == 0:
                return False
            
            # Detectar formas rectangulares
            phone_candidates = self.detect_phone_shapes(frame)
            
            if not phone_candidates:
                return False
            
            # Verificar si algún candidato está cerca de alguna cara
            for face_x, face_y, face_w, face_h in faces:
                face_center = (face_x + face_w//2, face_y + face_h//2)
                
                for phone in phone_candidates:
                    phone_center = phone['center']
                    
                    # Calcular distancia entre cara y objeto
                    distance = np.sqrt(
                        (face_center[0] - phone_center[0])**2 + 
                        (face_center[1] - phone_center[1])**2
                    )
                    
                    # Si el objeto está cerca de la cara
                    if distance < self.config['phone_distance_threshold']:
                        return True
            
            return False
            
        except Exception as e:
            print(f"ERROR: Error en detección de celular cerca de cara: {e}")
            return False
    
    def detect_hands_near_face(self, frame):
        """Detectar manos cerca de la cara usando MediaPipe"""
        if not self.hand_tracking_enabled:
            return False
            
        try:
            # Detectar rostros primero
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.config['face_sensitivity'],
                minNeighbors=5,
                minSize=self.config['min_face_size']
            )
            
            if len(faces) == 0:
                self.debug_info = "Sin rostros detectados"
                return False
            
            # Detectar manos
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if not results.multi_hand_landmarks:
                self.debug_info = "Sin manos detectadas"
                return False
            
            # Verificar si las manos están cerca de la cara
            h, w = frame.shape[:2]
            
            hands_near_face = 0
            total_hands = len(results.multi_hand_landmarks)
            
            for face_x, face_y, face_w, face_h in faces:
                face_center = (face_x + face_w//2, face_y + face_h//2)
                
                for hand_landmarks in results.multi_hand_landmarks:
                    # Obtener centro de la mano (punto 9 es el centro)
                    hand_center_x = int(hand_landmarks.landmark[9].x * w)
                    hand_center_y = int(hand_landmarks.landmark[9].y * h)
                    
                    # Calcular distancia mano-cara
                    distance = np.sqrt(
                        (face_center[0] - hand_center_x)**2 + 
                        (face_center[1] - hand_center_y)**2
                    )
                    
                    if distance < self.config['phone_distance_threshold']:
                        hands_near_face += 1
            
            self.debug_info = f"Manos: {total_hands}, Cerca cara: {hands_near_face}"
            
            # Detectar si al menos una mano está cerca de la cara
            return hands_near_face > 0
            
        except Exception as e:
            print(f"ERROR: Error en detección de manos: {e}")
            self.debug_info = f"Error manos: {str(e)[:20]}"
            return False
    
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
            
            if session_duration > 2:  # Solo contar sesiones > 2 segundos
                self.stats['sessions'].append({
                    'start': datetime.fromtimestamp(self.detection_start_time).isoformat(),
                    'duration': session_duration
                })
                
                self.stats['total_usage_today'] += session_duration
                print(f"Sesión guardada: {session_duration:.1f}s")
            
            self.detection_start_time = None
            self.save_stats()
    
    def show_alert(self):
        """Mostrar alerta"""
        self.stats['alerts_triggered'] += 1
        
        # Sonido
        self.play_alert_sound()
        
        # Ventana emergente
        alert = tk.Toplevel(self.root)
        alert.title("🚨 ALERTA")
        alert.geometry("500x300")
        alert.configure(bg='#e74c3c')
        alert.attributes('-topmost', True)
        alert.transient(self.root)
        alert.grab_set()
        
        # Centrar ventana
        alert.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        # Contenido
        tk.Label(alert, text="🚨 ¡DEJA EL CELULAR! 🚨", 
                font=('Arial', 24, 'bold'), bg='#e74c3c', fg='white').pack(pady=20)
        
        tk.Label(alert, text=f"Has estado {self.alert_time_var.get()} segundos usando el celular", 
                font=('Arial', 14), bg='#e74c3c', fg='white').pack(pady=10)
        
        tk.Label(alert, text="El sistema detectó uso prolongado", 
                font=('Arial', 12), bg='#e74c3c', fg='#f8c9ca').pack(pady=5)
        
        button_frame = tk.Frame(alert, bg='#e74c3c')
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="✅ Entendido", command=alert.destroy,
                 bg='white', fg='#e74c3c', font=('Arial', 12, 'bold'),
                 padx=20, pady=5).pack(side='left', padx=10)
        
        tk.Button(button_frame, text="⏰ +5 min más", 
                 command=lambda: [setattr(self, 'detection_start_time', time.time()), alert.destroy()],
                 bg='#c0392b', fg='white', font=('Arial', 12),
                 padx=20, pady=5).pack(side='left', padx=10)
        
        # Auto-cerrar
        alert.after(10000, alert.destroy)
        
        # Reiniciar timer
        self.detection_start_time = time.time()
        print("ALERTA MOSTRADA")
    
    def show_camera_error(self):
        """Mostrar error de cámara y detener monitoreo"""
        self.root.after(0, self._show_camera_error_ui)
    
    def _show_camera_error_ui(self):
        """Mostrar UI de error de cámara en el hilo principal"""
        error_msg = """💥 Error crítico de cámara

La cámara se desconectó o dejó de funcionar.

Soluciones:
• Verifica que la cámara esté conectada
• Cierra otras aplicaciones que usen la cámara
• Reconecta la cámara USB
• Reinicia la aplicación"""
        
        messagebox.showerror("💥 Error de Cámara", error_msg)
        self.stop_monitoring()
    
    def play_alert_sound(self):
        """Reproducir sonido simple"""
        try:
            # Usar beep del sistema si pygame no funciona
            import winsound
            winsound.Beep(1000, 500)  # 1000Hz por 500ms
        except:
            try:
                # Intentar con pygame
                duration = 0.5
                sample_rate = 22050
                frames = int(duration * sample_rate)
                arr = np.zeros(frames)
                
                for i in range(frames):
                    arr[i] = np.sin(2 * np.pi * 1000 * i / sample_rate)
                
                arr = (arr * 32767).astype(np.int16)
                sound = pygame.sndarray.make_sound(arr)
                sound.play()
            except:
                print("WARNING: No se pudo reproducir sonido")
    
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
            
            # Estado según tiempo
            remaining = self.alert_time_var.get() - elapsed
            if remaining <= 5:
                self.status_label.config(text="Estado: ⚠️ CASI LÍMITE", fg='#e74c3c')
            else:
                self.status_label.config(text="Estado: 📱 Usando celular", fg='#f39c12')
        else:
            self.status_label.config(text="Estado: 👀 Monitoreando...", fg='#3498db')
        
        # Estadísticas
        usage_minutes = int(self.stats['total_usage_today'] / 60)
        self.usage_today_label.config(text=f"📱 Uso total: {usage_minutes} min")
        self.sessions_label.config(text=f"📅 Sesiones: {len(self.stats['sessions'])}")
        self.alerts_label.config(text=f"🚨 Alertas: {self.stats['alerts_triggered']}")
        
        # Debug info
        method = self.detection_method.get()
        method_names = {
            "advanced_hybrid": "Avanzado",
            "mediapipe_hands": "Manos (MediaPipe)",
            "shape_detection": "Formas",
            "face_only": "Solo Cara",
            "motion_only": "Solo Movimiento"
        }
        self.detection_label.config(text=f"Método: {method_names.get(method, method)}")
        
        if hasattr(self, 'debug_info'):
            self.debug_label.config(text=f"Debug: {self.debug_info}")
        
        # Actualizar configuraciones dinámicamente
        self.config['phone_distance_threshold'] = self.phone_distance_var.get()
        
        self.root.after(100, self.update_ui)
    
    def clear_stats(self):
        """Limpiar estadísticas"""
        if messagebox.askyesno("🗑️ Confirmar", "¿Limpiar todas las estadísticas de hoy?"):
            self.stats = {'total_usage_today': 0, 'sessions': [], 'alerts_triggered': 0}
            self.save_stats()
            print("Estadísticas limpiadas")
    
    def save_stats(self):
        """Guardar estadísticas"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            with open(f'phone_stats_advanced_{today}.json', 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            print(f"ERROR: Error guardando: {e}")
    
    def load_stats(self):
        """Cargar estadísticas"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            filename = f'phone_stats_advanced_{today}.json'
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    self.stats = json.load(f)
                print(f"Estadísticas cargadas: {filename}")
        except Exception as e:
            print(f"ERROR: Error cargando: {e}")
    
    def run(self):
        """Ejecutar aplicación"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            print("Aplicación avanzada iniciada")
            print("Tip: Habilita la vista de cámara para ver la detección en tiempo real")
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nCtrl+C detectado - cerrando...")
            self.stop_monitoring()
    
    def on_closing(self):
        """Cerrar aplicación"""
        if self.is_monitoring:
            self.stop_monitoring()
        print("Aplicación cerrada")
        self.root.destroy()

if __name__ == "__main__":
    # Configurar codificación para Windows
    import sys
    
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass
    
    print("Detector Avanzado de Uso de Celular")
    print("=" * 50)
    print("Dependencias: pip install opencv-python numpy pygame pillow")
    print("Opcional: pip install mediapipe (para detección de manos)")
    print()
    
    try:
        detector = AdvancedPhoneDetector()
        detector.run()
    except ImportError as e:
        print(f"ERROR: Falta instalar: {e}")
        print("Ejecuta: pip install opencv-python numpy pygame pillow")
    except Exception as e:
        print(f"ERROR: {e}")