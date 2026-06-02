from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import math
from dtaidistance import dtw_ndim

app = FastAPI(title="Taekwondo Biomechanics Evaluator")

class EvaluationRequest(BaseModel):
    user_data: dict      
    master_data: dict    
    movement_type: str   

def extract_feature_matrices(data_dict: dict, mov_type: str, scale_factor: float) -> dict:
    """
    En lugar de devolver una sola matriz gigante, devuelve un diccionario 
    con matrices separadas para cada parte del cuerpo y tipo de movimiento.
    """
    frames = data_dict.get("frames", [])
    
    # Inicializamos el diccionario de características
    features = {
        "right_path": [], "right_rotation": [],
        "left_path": [], "left_rotation": [],
        "head_path": []
    }
    
    if not frames:
        return features

    for f in frames:
        nodes = f.get("trackers", f.get("bones", {}))
        if not nodes:
            continue
        
        # Origen pélvico
        px, py, pz = 0.0, 0.0, 0.0
        if "pelvis" in nodes and "position" in nodes["pelvis"]:
            px, py, pz = nodes["pelvis"]["position"]["x"], nodes["pelvis"]["position"]["y"], nodes["pelvis"]["position"]["z"]

        # Extraemos mano derecha (o pie derecho dependiendo del movimiento)
        node_r = "hand_r" if mov_type.lower() == "jirugi" else "foot_r"
        if node_r in nodes:
            pos = nodes[node_r].get("position", {"x": 0, "y": 0, "z": 0})
            quat = nodes[node_r].get("rotation_quat", nodes[node_r].get("rotation_quaternion", {"x":0,"y":0,"z":0,"w":1}))
            features["right_path"].append([(pos["x"]-px)/scale_factor, (pos["y"]-py)/scale_factor, (pos["z"]-pz)/scale_factor])
            features["right_rotation"].append([quat["x"], quat["y"], quat["z"], quat["w"]])

        # Extraemos mano izquierda (o pie izquierdo)
        node_l = "hand_l" if mov_type.lower() == "jirugi" else "foot_l"
        if node_l in nodes:
            pos = nodes[node_l].get("position", {"x": 0, "y": 0, "z": 0})
            quat = nodes[node_l].get("rotation_quat", nodes[node_l].get("rotation_quaternion", {"x":0,"y":0,"z":0,"w":1}))
            features["left_path"].append([(pos["x"]-px)/scale_factor, (pos["y"]-py)/scale_factor, (pos["z"]-pz)/scale_factor])
            features["left_rotation"].append([quat["x"], quat["y"], quat["z"], quat["w"]])

    # Convertir todas las listas a numpy arrays dobles
    for key in features:
        features[key] = np.array(features[key], dtype=np.double)
        
    return features

@app.post("/evaluate")
def evaluate_movement(payload: EvaluationRequest):
    try:
        # 1. Calculamos el Factor de Escala basado en el maestro (o el alumno)
        scale_factor = 1.0
        try:
            first_frame = payload.master_data.get("frames", [])[0].get("trackers", payload.master_data.get("frames", [])[0].get("bones", {}))
            p0 = first_frame["pelvis"]["position"]
            h0 = first_frame["head"]["position"]
            torso_length = math.sqrt((h0["x"] - p0["x"])**2 + (h0["y"] - p0["y"])**2 + (h0["z"] - p0["z"])**2)
            if torso_length > 0.1: scale_factor = torso_length
        except:
            pass

        # 2. Extraemos las matrices desglosadas
        user_features = extract_feature_matrices(payload.user_data, payload.movement_type, scale_factor)
        master_features = extract_feature_matrices(payload.master_data, payload.movement_type, scale_factor)

        # 3. Calculamos distancias individuales
        results = {}
        total_distance = 0.0
        
        # Nombres legibles para el reporte
        nombres_legibles = {
            "right_path": "trayectoria de la extremidad derecha",
            "right_rotation": "giro/rotación de la extremidad derecha",
            "left_path": "trayectoria de la extremidad izquierda",
            "left_rotation": "giro/rotación de la extremidad izquierda"
        }

        peor_falla = ""
        mayor_error = 0.0

        for key in ["right_path", "right_rotation", "left_path", "left_rotation"]:
            if len(user_features[key]) > 0 and len(master_features[key]) > 0:
                dist = dtw_ndim.distance(user_features[key], master_features[key])
                results[key] = dist
                total_distance += dist
                
                # Buscamos qué parte del cuerpo generó el mayor error
                if dist > mayor_error:
                    mayor_error = dist
                    peor_falla = nombres_legibles[key]

        # 4. Cálculo del Score General
        max_error_global = 15000.0  # Ajustar tras pruebas
        score_general = max(0.0, 100.0 * (1.0 - (total_distance / max_error_global)))

        # 5. Generación del Feedback Textual
        if score_general > 90:
            feedback = "¡Técnica excelente! Movimiento y rotación correctos."
        elif score_general > 70:
            feedback = f"Buen intento, pero debes prestar mucha atención a la {peor_falla}."
        else:
            feedback = f"Técnica incorrecta. Tu principal problema está en la {peor_falla}. Revisa la demostración del maestro e inténtalo de nuevo."

        return {
            "success": True,
            "score": round(score_general, 2),
            "feedback": feedback,
            "detailed_metrics": results # Útil para que lo guardes en Supabase para analítica en tu tesis
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
