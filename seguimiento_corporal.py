#!/usr/bin/env python3
"""
Sistema básico de detección de movimientos de manos
Usando OpenCV para detección de movimiento por color y contornos
Autor: GitHub Copilot
Fecha: 2025-10-13
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional

class DetectorMovimientoManos:
    """
    Clase para detectar movimientos básicos de manos usando OpenCV
    """
    
    def __init__(self):
        self.cap = None
        self.is_running = False
        
        # Parámetros para detección de piel
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Variables para seguimiento de movimiento
        self.previous_positions = []
        self.movement_threshold = 50
        self.max_history = 10
        
        # Configuración de ventana
        self.window_name = "Detector de Movimiento de Manos"
        
    def inicializar_camara(self, camera_id: int = 0) -> bool:
        """
        Inicializa la cámara web
        
        Args:
            camera_id: ID de la cámara (0 para cámara por defecto)
            
        Returns:
            bool: True si la cámara se inicializó correctamente
        """
        try:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                print("Error: No se pudo abrir la cámara")
                return False
                
            # Configurar resolución
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            print("Cámara inicializada correctamente")
            return True
            
        except Exception as e:
            print(f"Error al inicializar la cámara: {e}")
            return False
    
    def detectar_piel(self, frame: np.ndarray) -> np.ndarray:
        """
        Detecta áreas de piel en la imagen usando espacios de color HSV
        
        Args:
            frame: Frame de video en formato BGR
            
        Returns:
            np.ndarray: Máscara binaria con áreas de piel detectadas
        """
        # Convertir a HSV para mejor detección de piel
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Crear máscara para color de piel
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # Aplicar operaciones morfológicas para limpiar la máscara
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Aplicar blur para suavizar
        mask = cv2.medianBlur(mask, 5)
        
        return mask
    
    def encontrar_contornos_manos(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Encuentra contornos que probablemente corresponden a manos
        
        Args:
            mask: Máscara binaria con áreas de piel
            
        Returns:
            List[np.ndarray]: Lista de contornos detectados
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos por área mínima
        min_area = 1000
        hand_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                hand_contours.append(contour)
        
        # Ordenar por área (las manos más grandes primero)
        hand_contours.sort(key=cv2.contourArea, reverse=True)
        
        # Retornar máximo 2 contornos (2 manos)
        return hand_contours[:2]
    
    def calcular_centro_masa(self, contour: np.ndarray) -> Tuple[int, int]:
        """
        Calcula el centro de masa de un contorno
        
        Args:
            contour: Contorno de la mano
            
        Returns:
            Tuple[int, int]: Coordenadas (x, y) del centro de masa
        """
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        return (0, 0)
    
    def detectar_movimiento(self, posiciones_actuales: List[Tuple[int, int]]) -> str:
        """
        Detecta el tipo de movimiento basado en las posiciones actuales y anteriores
        
        Args:
            posiciones_actuales: Lista de posiciones actuales de las manos
            
        Returns:
            str: Descripción del movimiento detectado
        """
        if not self.previous_positions or not posiciones_actuales:
            self.previous_positions = posiciones_actuales
            return "Sin movimiento"
        
        if len(posiciones_actuales) != len(self.previous_positions):
            self.previous_positions = posiciones_actuales
            return "Cambio en número de manos"
        
        movimientos = []
        
        for i, (pos_actual, pos_anterior) in enumerate(zip(posiciones_actuales, self.previous_positions)):
            dx = pos_actual[0] - pos_anterior[0]
            dy = pos_actual[1] - pos_anterior[1]
            distancia = np.sqrt(dx**2 + dy**2)
            
            if distancia > self.movement_threshold:
                direccion = self.determinar_direccion(dx, dy)
                movimientos.append(f"Mano {i+1}: {direccion}")
        
        self.previous_positions = posiciones_actuales
        
        if movimientos:
            return " | ".join(movimientos)
        else:
            return "Manos estacionarias"
    
    def determinar_direccion(self, dx: int, dy: int) -> str:
        """
        Determina la dirección del movimiento basado en dx y dy
        
        Args:
            dx: Diferencia en x
            dy: Diferencia en y
            
        Returns:
            str: Dirección del movimiento
        """
        if abs(dx) > abs(dy):
            return "Derecha" if dx > 0 else "Izquierda"
        else:
            return "Abajo" if dy > 0 else "Arriba"
    
    def dibujar_informacion(self, frame: np.ndarray, contours: List[np.ndarray], 
                           posiciones: List[Tuple[int, int]], movimiento: str) -> np.ndarray:
        """
        Dibuja información visual en el frame
        
        Args:
            frame: Frame original
            contours: Lista de contornos detectados
            posiciones: Lista de posiciones de las manos
            movimiento: Descripción del movimiento
            
        Returns:
            np.ndarray: Frame con información visual
        """
        # Dibujar contornos de las manos
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        
        # Dibujar centros de masa y etiquetas
        for i, (contour, pos) in enumerate(zip(contours, posiciones)):
            # Dibujar centro
            cv2.circle(frame, pos, 8, (255, 0, 0), -1)
            
            # Dibujar etiqueta
            cv2.putText(frame, f"Mano {i+1}", (pos[0]-30, pos[1]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Dibujar área del contorno
            area = cv2.contourArea(contour)
            cv2.putText(frame, f"Area: {int(area)}", (pos[0]-40, pos[1]+40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Dibujar información de movimiento
        cv2.putText(frame, f"Movimiento: {movimiento}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Dibujar número de manos detectadas
        cv2.putText(frame, f"Manos detectadas: {len(contours)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Dibujar instrucciones
        cv2.putText(frame, "Presiona 'q' para salir, 'r' para reiniciar", (10, frame.shape[0]-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def ejecutar(self):
        """
        Bucle principal de detección de movimiento de manos
        """
        if not self.inicializar_camara():
            return
        
        print("Sistema de detección iniciado...")
        print("Instrucciones:")
        print("- Muestra tus manos frente a la cámara")
        print("- Presiona 'q' para salir")
        print("- Presiona 'r' para reiniciar el seguimiento")
        print("- Presiona 'c' para calibrar detección de piel")
        
        self.is_running = True
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: No se pudo leer el frame de la cámara")
                    break
                
                # Voltear horizontalmente para efecto espejo
                frame = cv2.flip(frame, 1)
                
                # Detectar piel
                mask = self.detectar_piel(frame)
                
                # Encontrar contornos de manos
                contours = self.encontrar_contornos_manos(mask)
                
                # Calcular posiciones de las manos
                posiciones = [self.calcular_centro_masa(contour) for contour in contours]
                
                # Detectar movimiento
                movimiento = self.detectar_movimiento(posiciones)
                
                # Dibujar información en el frame
                frame_con_info = self.dibujar_informacion(frame, contours, posiciones, movimiento)
                
                # Mostrar el resultado
                cv2.imshow(self.window_name, frame_con_info)
                
                # Mostrar máscara de piel en ventana separada (opcional)
                cv2.imshow("Detección de Piel", mask)
                
                # Manejar eventos de teclado
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Saliendo del sistema...")
                    break
                elif key == ord('r'):
                    print("Reiniciando seguimiento...")
                    self.previous_positions = []
                elif key == ord('c'):
                    print("Calibrando detección de piel...")
                    self.calibrar_deteccion_piel()
                
        except KeyboardInterrupt:
            print("\nInterrumpido por el usuario")
        except Exception as e:
            print(f"Error durante la ejecución: {e}")
        finally:
            self.limpiar_recursos()
    
    def calibrar_deteccion_piel(self):
        """
        Permite ajustar los parámetros de detección de piel
        """
        print("Calibración de piel - Ajusta los valores con las teclas:")
        print("'1','2' - Ajustar Hue mínimo/máximo")
        print("'3','4' - Ajustar Saturación mínimo/máximo")  
        print("'5','6' - Ajustar Value mínimo/máximo")
        print("'Esc' - Finalizar calibración")
        
        # Esta función se puede expandir para permitir calibración interactiva
        pass
    
    def limpiar_recursos(self):
        """
        Libera los recursos utilizados
        """
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Recursos liberados correctamente")


def main():
    """
    Función principal
    """
    print("=== Sistema de Detección de Movimiento de Manos ===")
    print("Versión: 1.0")
    print("Usando OpenCV para detección básica")
    print("")
    
    detector = DetectorMovimientoManos()
    detector.ejecutar()


if __name__ == "__main__":
    main()
