# 📱 Detector de Uso de Celular

Un sistema inteligente de monitoreo que detecta cuando estás usando el celular y te alerta para tomar descansos saludables.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Stable-success)

## 🎯 Características Principales

### 🔍 **Detección Inteligente**
- **Detección de formas rectangulares** - Identifica objetos con proporción de celular
- **Detección de manos optimizada** - Analiza movimiento de manos en regiones específicas
- **Detección facial** - Complementa otros métodos para mayor precisión
- **Múltiples algoritmos** - Combina diferentes técnicas para máxima efectividad

### 📺 **Vista en Tiempo Real**
- **Cámara en vivo** con visualización de detección
- **Marcos de colores** para diferentes tipos de detección:
  - 🟢 **Verde**: Rostros detectados
  - 🔵 **Azul**: Formas de celular identificadas
  - 🟡 **Amarillo**: Regiones de manos activas
- **Información de debug** superpuesta en tiempo real

### 🚨 **Alertas Impactantes**
- **Pantalla completa** - Imposible de ignorar
- **Mensajes motivacionales** aleatorios
- **Sonidos de alerta** múltiples
- **Contador regresivo** automático
- **Cierre con ESC** o botones grandes

## 🚀 Instalación Rápida

### Requisitos
- Python 3.8 o superior
- Cámara web (integrada o USB)
- Windows/macOS/Linux

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/monitor-crazy.git
cd monitor-crazy
```

### 2. Instalar dependencias
```bash
pip install opencv-python numpy pygame pillow
```

### 3. Ejecutar
```bash
python phone_detector_optimized.py
```

## 📖 Guía de Uso

### Inicio Rápido
1. **Ejecuta** el detector
2. **Haz clic** en "📹 Mostrar Cámara"
3. **Selecciona** tu método preferido de detección
4. **Presiona** "▶️ INICIAR DETECCIÓN"
5. **¡Observa** la detección en tiempo real!

### Métodos de Detección

#### 🖐️ **Solo Manos (RECOMENDADO)**
- Detecta movimiento de manos cerca del rostro
- Funciona incluso sin detectar rostro
- Ideal para uso general de celular

#### 📱 **Solo Formas**
- Identifica objetos rectangulares con proporción de celular (1.2-3.5)
- Muy preciso para celulares visibles
- Menos falsos positivos

#### 🏃 **Solo Movimiento**
- Detecta actividad general en la imagen
- Más sensible, puede captar cualquier uso
- Bueno para actividad constante

#### 🎯 **Inteligente Flexible**
- Combina múltiples métodos
- Lógica: Cara O Manos O (Formas + Movimiento)
- Máxima cobertura de detección

## 🛠️ Crear Ejecutable

Para crear un archivo .exe que no requiera Python instalado:

### Opción 1: PyInstaller (RECOMENDADO)
```bash
pip install pyinstaller
pyinstaller --onefile --windowed phone_detector_optimized.py
```

### Opción 2: Auto-py-to-exe (Interfaz gráfica)
```bash
pip install auto-py-to-exe
auto-py-to-exe
```

El ejecutable se creará en la carpeta `dist/`

## 📊 Estadísticas y Datos

El sistema guarda automáticamente:
- **Tiempo total** de uso diario
- **Número de sesiones** por día
- **Cantidad de alertas** disparadas
- **Duración de cada sesión** con timestamps

## 🔧 Configuración Recomendada

Para **máxima efectividad**:
- **Método**: 📱 Solo Formas (si funciona bien) o 🖐️ Solo Manos
- **Tiempo de alerta**: 15-30 segundos
- **Distancia máxima**: 150-200 píxeles

## 🚧 Solución de Problemas

### ❌ "No se pudo acceder a la cámara"
**Solución:**
1. Cierra otras aplicaciones que usen la cámara
2. Verifica permisos de cámara en Windows
3. Reconecta cámara USB si es externa

### ❌ "Demasiados falsos positivos"
**Solución:**
1. Usa método "Solo Formas"
2. Aumenta área mínima en configuración
3. Reduce sensibilidad de movimiento

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.

## 👨‍💻 Uso

Desarrollado para ayudar a mantener hábitos saludables con dispositivos móviles.

---

**¡Cuida tu bienestar digital! 📱✨**