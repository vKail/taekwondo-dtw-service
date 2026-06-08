from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.spatial.transform import Rotation as R

app = FastAPI(title="Taekwondo Biomechanics Evaluator")

class EvaluationRequest(BaseModel):
    user_data: dict      
    master_data: dict    
    movement_type: str   

def calculate_torso_length(first_frame_nodes: dict) -> float:
    """
    Calcula la longitud del torso usando la distancia euclidiana entre la cabeza y la pelvis.
    """
    try:
        h0 = first_frame_nodes.get("head", {}).get("position", {"x":0,"y":0,"z":0})
        p0 = first_frame_nodes.get("pelvis", {}).get("position", {"x":0,"y":0,"z":0})
        h_pos = np.array([h0["x"], h0["y"], h0["z"]])
        p_pos = np.array([p0["x"], p0["y"], p0["z"]])
        dist = np.linalg.norm(h_pos - p_pos)
        return dist if dist > 0.1 else 1.0
    except Exception:
        return 1.0

def normalize_quaternion_to_euler(quat_dict: dict) -> np.ndarray:
    """
    Convierte cuaterniones del diccionario a Ángulos de Euler (Roll, Pitch, Yaw) en grados.
    """
    x = quat_dict.get("x", 0.0)
    y = quat_dict.get("y", 0.0)
    z = quat_dict.get("z", 0.0)
    w = quat_dict.get("w", 1.0)
    if x == 0 and y == 0 and z == 0 and w == 0:
        w = 1.0
    rotation = R.from_quat([x, y, z, w])
    return rotation.as_euler('xyz', degrees=True)

def unwrap_euler_series(euler_series: list) -> np.ndarray:
    """
    Aplica np.unwrap a una serie de ángulos de Euler para evitar saltos de 360 grados (Gimbal Lock).
    """
    arr = np.array(euler_series)
    arr_rad = np.deg2rad(arr)
    arr_rad_unwrapped = np.unwrap(arr_rad, axis=0)
    return np.rad2deg(arr_rad_unwrapped)

def extract_nodes(frame: dict) -> dict:
    return frame.get("trackers", frame.get("bones", {}))

def get_joint_keys_for_movement(movement_type: str) -> list:
    """
    Devuelve estrictamente las articulaciones clave basadas en el JSON proporcionado.
    El JSON analizado solo contiene: head, pelvis, hand_r, hand_l, foot_r, foot_l.
    Excluimos 'pelvis' porque en el JSON actúa como el origen estático.
    """
    mov = movement_type.lower()
    if mov == "jirugi":
        return ["head", "hand_r", "hand_l"]
    elif mov in ["ap_chagi", "dollyo_chagi", "yop_chagi", "chagi"]:
        return ["head", "foot_r", "foot_l"]
    else:
        return ["head", "hand_r", "hand_l", "foot_r", "foot_l"]

def find_worst_moment(master_series, student_series, path):
    """
    Itera sobre el camino de alineación DTW y encuentra el índice del alumno (j)
    donde ocurrió la mayor distancia euclidiana frente al maestro.
    """
    max_d = -1.0
    worst_student_idx = 0
    for i, j in path:
        d = euclidean(master_series[i], student_series[j])
        if d > max_d:
            max_d = d
            worst_student_idx = j
    return worst_student_idx

@app.post("/evaluate")
def evaluate_movement(payload: EvaluationRequest):
    try:
        user_frames = payload.user_data.get("frames", [])
        master_frames = payload.master_data.get("frames", [])

        if not user_frames or not master_frames:
            raise HTTPException(status_code=400, detail="Los datos deben contener 'frames'.")

        # 1. FACTORES DE ESCALA INDEPENDIENTES
        u_nodes_0 = extract_nodes(user_frames[0])
        m_nodes_0 = extract_nodes(master_frames[0])
        
        scale_user = calculate_torso_length(u_nodes_0)
        scale_master = calculate_torso_length(m_nodes_0)

        joint_keys = get_joint_keys_for_movement(payload.movement_type)

        joint_metrics = {}

        for joint in joint_keys:
            master_pos_series = []
            student_pos_series = []
            master_rot_series = []
            student_rot_series = []

            for u_frame in user_frames:
                u_nodes = extract_nodes(u_frame)
                if joint not in u_nodes: continue
                
                up = u_nodes[joint].get("position", {"x":0,"y":0,"z":0})
                u_pos = np.array([up["x"], up["y"], up["z"]]) / scale_user
                student_pos_series.append(u_pos)

                uq = u_nodes[joint].get("rotation_quat", u_nodes[joint].get("rotation_quaternion", {"x":0,"y":0,"z":0,"w":1}))
                student_rot_series.append(normalize_quaternion_to_euler(uq))

            for m_frame in master_frames:
                m_nodes = extract_nodes(m_frame)
                if joint not in m_nodes: continue

                mp = m_nodes[joint].get("position", {"x":0,"y":0,"z":0})
                m_pos = np.array([mp["x"], mp["y"], mp["z"]]) / scale_master
                master_pos_series.append(m_pos)

                mq = m_nodes[joint].get("rotation_quat", m_nodes[joint].get("rotation_quaternion", {"x":0,"y":0,"z":0,"w":1}))
                master_rot_series.append(normalize_quaternion_to_euler(mq))

            if not student_pos_series or not master_pos_series:
                continue

            # Desenvolver ángulos para evitar Gimbal Lock
            student_rot_series = unwrap_euler_series(student_rot_series)
            master_rot_series = unwrap_euler_series(master_rot_series)

            # DTW - Posiciones
            dist_pos, path_pos = fastdtw(master_pos_series, student_pos_series, dist=euclidean)
            norm_error_pos = dist_pos / len(path_pos)

            # DTW - Rotaciones
            dist_rot, path_rot = fastdtw(master_rot_series, student_rot_series, dist=euclidean)
            norm_error_rot = dist_rot / len(path_rot)
            
            joint_metrics[joint] = {
                "pos_error": norm_error_pos,
                "rot_error": norm_error_rot,
                "path_pos": path_pos,
                "path_rot": path_rot,
                "master_pos_series": master_pos_series,
                "student_pos_series": student_pos_series,
                "master_rot_series": master_rot_series,
                "student_rot_series": student_rot_series
            }

        if not joint_metrics:
            raise HTTPException(status_code=400, detail="No se encontraron articulaciones válidas para comparar.")

        # 2. DECISIÓN DE ERROR CRÍTICO (Posición vs Rotación)
        ROT_THRESHOLD = 15.0 # Grados de tolerancia promedio
        POS_THRESHOLD = 0.15 # 15% del tamaño del torso como tolerancia
        
        worst_severity = -1.0
        critical_joint = None
        error_type = None

        for joint, metrics in joint_metrics.items():
            r_sev = metrics["rot_error"] / ROT_THRESHOLD
            p_sev = metrics["pos_error"] / POS_THRESHOLD
            
            if r_sev > worst_severity:
                worst_severity = r_sev
                critical_joint = joint
                error_type = "rotación"
                
            if p_sev > worst_severity:
                worst_severity = p_sev
                critical_joint = joint
                error_type = "trayectoria"

        # Extraer métricas de la peor articulación para el JSON
        max_angular_dev = joint_metrics[critical_joint]["rot_error"]
        max_positional_dev = joint_metrics[critical_joint]["pos_error"]

        # 3. ENCONTRAR EL MOMENTO EXACTO DE LA FALLA
        c_metrics = joint_metrics[critical_joint]
        if error_type == "rotación":
            worst_idx = find_worst_moment(c_metrics["master_rot_series"], c_metrics["student_rot_series"], c_metrics["path_rot"])
        else:
            worst_idx = find_worst_moment(c_metrics["master_pos_series"], c_metrics["student_pos_series"], c_metrics["path_pos"])

        total_student_frames = len(user_frames)
        worst_frame_data = user_frames[worst_idx]
        
        # Datos exactos para Unreal Engine 5
        frame_num = worst_frame_data.get("frame", worst_idx)
        time_sec = worst_frame_data.get("time_sec", 0.0)
        percentage_execution = round((worst_idx / max(1, total_student_frames - 1)) * 100, 2)

        # 4. GENERACIÓN DEL FEEDBACK INTELIGENTE
        if percentage_execution < 33.3:
            fase_texto = "al inicio del movimiento (fase de preparación)"
        elif percentage_execution < 66.6:
            fase_texto = "a la mitad del movimiento (fase de ejecución)"
        else:
            fase_texto = "al final del movimiento (fase de impacto/recuperación)"

        traduccion = {
            "head": "cabeza",
            "hand_r": "mano derecha",
            "hand_l": "mano izquierda",
            "foot_r": "pie derecho",
            "foot_l": "pie izquierdo"
        }
        falla_legible = traduccion.get(critical_joint, critical_joint)

        passed = worst_severity <= 1.0  # Menor a 1.0 significa que no superó los umbrales
        score_general = max(0.0, 100.0 - (worst_severity * 20.0))

        if passed:
            feedback = "¡Técnica excelente! Movimiento y rotación correctos."
        else:
            if error_type == "rotación":
                feedback = f"Técnica incorrecta en la {falla_legible}. Tu error principal fue en el giro o rotación de la articulación, ocurriendo específicamente {fase_texto}."
            else:
                feedback = f"Técnica incorrecta en la {falla_legible}. La trayectoria se desvió demasiado del maestro, y el mayor error ocurrió {fase_texto}."

        return {
            "success": True,
            "score": round(score_general, 2),
            "feedback": feedback,
            "detailed_metrics": {
                "max_angular_deviation": round(max_angular_dev, 2),
                "max_positional_deviation": round(max_positional_dev, 2),
                "critical_failure_joint": critical_joint,
                "critical_failure_type": error_type,
                "error_percentage_execution": percentage_execution,
                "error_frame_exact": frame_num,
                "error_time_sec": round(time_sec, 4),
                "passed": passed
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
