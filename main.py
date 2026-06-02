from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from dtaidistance import dtw_ndim

app = FastAPI(title="Taekwondo Biomechanics Evaluator")

# DTO para recibir el payload estructurado desde NestJS
class EvaluationRequest(BaseModel):
    user_data: dict      # Corresponde a executionData
    master_data: dict    # Corresponde a jointsData
    movement_type: str   # Corresponde a techniqueName (ej. "jirugi")

def process_frames(data_dict: dict, mov_type: str) -> np.ndarray:
    """
    Aplana los JSON en matrices matemáticas.
    Aplica normalización espacial (restando la pelvis) e integra los cuaterniones.
    """
    frames = data_dict.get("frames", [])
    matrix = []
    
    for f in frames:
        # Soporte para ambas nomenclaturas detectadas
        nodes = f.get("trackers", f.get("bones", {}))
        if not nodes:
            continue
        
        # 1. Extraer la pelvis para usarla como punto de origen (0,0,0)
        px, py, pz = 0.0, 0.0, 0.0
        if "pelvis" in nodes and "position" in nodes["pelvis"]:
            px = nodes["pelvis"]["position"]["x"]
            py = nodes["pelvis"]["position"]["y"]
            pz = nodes["pelvis"]["position"]["z"]

        flat_vector = []
        
        # 2. Determinar qué nodos evaluar según la técnica
        if mov_type.lower() == "jirugi":
            target_nodes = ["hand_r", "hand_l"]
        else:
            # Por defecto asumimos técnicas de pateo (Ap Chagi, Dollyo Chagi)
            target_nodes = ["foot_r", "foot_l", "head"]

        # 3. Construir el vector dimensional para este frame
        for node in target_nodes:
            if node in nodes:
                pos = nodes[node].get("position", {"x": 0.0, "y": 0.0, "z": 0.0})
                # Soporte para ambas llaves de cuaterniones
                quat = nodes[node].get("rotation_quat", nodes[node].get("rotation_quaternion", {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}))
                
                # Inserción: Posición Normalizada (3) + Cuaternión (4) = 7 dimensiones por nodo
                flat_vector.extend([
                    pos["x"] - px, pos["y"] - py, pos["z"] - pz,
                    quat["x"], quat["y"], quat["z"], quat["w"]
                ])
            else:
                # Fallback de seguridad si falta un tracker en un frame específico (evita crashes)
                flat_vector.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
                
        matrix.append(flat_vector)
        
    # El tipo np.double es crítico para la precisión de dtaidistance
    return np.array(matrix, dtype=np.double)


@app.post("/evaluate")
def evaluate_movement(payload: EvaluationRequest):
    try:
        # Pre-procesamiento de las matrices
        user_matrix = process_frames(payload.user_data, payload.movement_type)
        master_matrix = process_frames(payload.master_data, payload.movement_type)

        if len(user_matrix) == 0 or len(master_matrix) == 0:
            raise ValueError("Las matrices de datos están vacías. Verifica el formato del JSON.")

        # Ejecución del Dynamic Time Warping multidimensional exacto
        distance = dtw_ndim.distance(user_matrix, master_matrix)

        # Lógica de conversión de Distancia a Porcentaje (Score)
        # El valor 15000.0 es un umbral configurable. Si al probar en VR notas 
        # que califica muy duro, súbelo a 20000.0. Si califica muy suave, bájalo a 10000.0.
        max_error = 15000.0 
        score = max(0.0, 100.0 * (1.0 - (distance / max_error)))

        return {
            "score": round(score, 2),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
