# Sistema de Detección de Movimiento de Manos

## Descripción
Sistema básico para detectar y rastrear movimientos de manos en tiempo real usando OpenCV. Este proyecto detecta manos mediante segmentación por color de piel y análisis de contornos.

## Características
- ✅ Detección en tiempo real de hasta 2 manos
- ✅ Seguimiento de movimientos (arriba, abajo, izquierda, derecha)
- ✅ Visualización de contornos y centros de masa
- ✅ Interfaz visual con información en pantalla
- ✅ Calibración de parámetros de detección

## Requisitos del Sistema
- Python 3.8 o superior
- Cámara web funcional
- Windows/Linux/macOS

## Dependencias
```bash
pip install opencv-python numpy
```

## Instalación
1. Clona o descarga este repositorio
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecuta el programa:
   ```bash
   python seguimiento_corporal.py
   ```

## Uso

### Controles durante la ejecución:
- **'q'** - Salir del programa
- **'r'** - Reiniciar el seguimiento de movimiento
- **'c'** - Modo de calibración (en desarrollo)
- **Esc** - Cerrar ventanas

### Instrucciones:
1. Ejecuta el programa
2. Colócate frente a la cámara con buena iluminación
3. Muestra tus manos al campo de visión
4. El sistema detectará automáticamente tus manos y mostrará:
   - Contornos verdes alrededor de las manos detectadas
   - Puntos azules en el centro de cada mano
   - Información de movimiento en tiempo real
   - Área de cada mano detectada

## Funcionalidades Implementadas

### 1. Detección de Piel
- Utiliza el espacio de color HSV para mejor detección
- Filtros morfológicos para limpiar la detección
- Parámetros ajustables para diferentes tonos de piel

### 2. Análisis de Contornos
- Filtrado por área mínima para eliminar ruido
- Detección de hasta 2 manos simultáneamente
- Cálculo de centro de masa para seguimiento

### 3. Seguimiento de Movimiento
- Detección de dirección de movimiento
- Umbral configurable para sensibilidad
- Histórico de posiciones para análisis temporal

### 4. Visualización
- Overlay de información en tiempo real
- Ventana separada para visualizar máscara de piel
- Indicadores visuales claros y concisos

## Estructura del Código

```
DetectorMovimientoManos/
├── __init__()              # Inicialización de parámetros
├── inicializar_camara()    # Configuración de cámara web
├── detectar_piel()         # Segmentación por color de piel
├── encontrar_contornos_manos() # Análisis de contornos
├── calcular_centro_masa()  # Cálculo de centros
├── detectar_movimiento()   # Análisis de movimiento
├── determinar_direccion()  # Clasificación de dirección
├── dibujar_informacion()   # Renderizado visual
└── ejecutar()              # Bucle principal
```

## Parámetros Configurables

### Detección de Piel (HSV)
- `lower_skin = [0, 20, 70]` - Límite inferior HSV
- `upper_skin = [20, 255, 255]` - Límite superior HSV

### Detección de Movimiento
- `movement_threshold = 50` - Umbral mínimo de movimiento (píxeles)
- `min_area = 1000` - Área mínima para considerar una mano
- `max_history = 10` - Máximo de posiciones en histórico

## Limitaciones Actuales
- Dependiente de condiciones de iluminación
- Detección basada en color de piel (puede variar entre personas)
- No incluye reconocimiento de gestos específicos
- MediaPipe no disponible para Python 3.13 (implementación alternativa)

## Mejoras Futuras
- [ ] Migración a MediaPipe cuando esté disponible
- [ ] Reconocimiento de gestos específicos
- [ ] Calibración automática de color de piel
- [ ] Mejora en condiciones de poca luz
- [ ] Detección de dedos individuales
- [ ] Grabación y análisis de movimientos

## Solución de Problemas

### La cámara no se detecta
- Verifica que la cámara esté conectada y funcionando
- Prueba cambiar el `camera_id` en el código (0, 1, 2...)
- Asegúrate de que ningún otro programa esté usando la cámara

### Mala detección de manos
- Ajusta la iluminación (preferible luz natural)
- Modifica los parámetros HSV en el código
- Usa la función de calibración ('c' durante ejecución)
- Asegúrate de tener un fondo contrastante

### Rendimiento lento
- Reduce la resolución de la cámara en `inicializar_camara()`
- Ajusta los parámetros de procesamiento
- Cierra otros programas que usen recursos del sistema

## Migración a MediaPipe

Cuando MediaPipe esté disponible para Python 3.13, se puede migrar fácilmente:

```python
import mediapipe as mp

# Inicialización de MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

## Contribuciones
Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## Licencia
Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## Autor
Desarrollado con GitHub Copilot
Fecha: 13 de octubre de 2025

