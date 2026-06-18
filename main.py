from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
import time
import logging
import uuid
import json
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("dtw_evaluator")

app = FastAPI(title="Taekwondo Biomechanics Evaluator")

class EvaluationRequest(BaseModel):
    user_data: dict      
    master_data: dict    
    movement_type: str   

def calculate_torso_length(first_frame_nodes: dict) -> float:
    try:
        h0 = first_frame_nodes.get("head", {}).get("position", {"x":0,"y":0,"z":0})
        p0 = first_frame_nodes.get("pelvis", {}).get("position", {"x":0,"y":0,"z":0})
        h_pos = np.array([h0["x"], h0["y"], h0["z"]])
        p_pos = np.array([p0["x"], p0["y"], p0["z"]])
        dist = np.linalg.norm(h_pos - p_pos)
        return dist if dist > 0.1 else 1.0
    except Exception:
        return 1.0

def extract_yaw_from_quat(quat_dict: dict) -> float:
    x = quat_dict.get("x", 0.0)
    y = quat_dict.get("y", 0.0)
    z = quat_dict.get("z", 0.0)
    w = quat_dict.get("w", 1.0)
    if x == 0 and y == 0 and z == 0 and w == 0:
        w = 1.0
    r = R.from_quat([x, y, z, w])
    euler = r.as_euler('zyx', degrees=False)
    return euler[0] 

def calculate_spatial_alignment(student_frames, master_frames) -> R:
    """Calcula la matriz de rotación en Z usando la POSICIÓN de los pies como origen direccional."""
    try:
        u_nodes_0 = student_frames[0].get("trackers", student_frames[0].get("bones", {}))
        m_nodes_0 = master_frames[0].get("trackers", master_frames[0].get("bones", {}))
        
        u_foot_r = u_nodes_0.get("foot_r", {}).get("position", {"x":0,"y":0,"z":0})
        u_foot_l = u_nodes_0.get("foot_l", {}).get("position", {"x":0,"y":0,"z":0})
        
        m_foot_r = m_nodes_0.get("foot_r", {}).get("position", {"x":0,"y":0,"z":0})
        m_foot_l = m_nodes_0.get("foot_l", {}).get("position", {"x":0,"y":0,"z":0})
        
        u_vec_x = u_foot_r["x"] - u_foot_l["x"]
        u_vec_y = u_foot_r["y"] - u_foot_l["y"]
        
        m_vec_x = m_foot_r["x"] - m_foot_l["x"]
        m_vec_y = m_foot_r["y"] - m_foot_l["y"]
        
        u_yaw = np.arctan2(u_vec_y, u_vec_x)
        m_yaw = np.arctan2(m_vec_y, m_vec_x)
        
        delta_yaw = m_yaw - u_yaw
        return R.from_euler('z', delta_yaw, degrees=False)
    except Exception:
        return R.from_euler('z', 0, degrees=False)

def subsequence_dtw(master_series, student_series, mode="pos"):
    """
    Subsequence DTW vectorizado parcialmente para máxima velocidad.
    Encuentra la mejor ventana del maestro dentro de una grabación más larga del estudiante.
    mode puede ser "pos" (euclidean) o "rot" (quat_angular_distance vectorizado).
    """
    X = np.array(master_series)
    Y = np.array(student_series)
    
    if mode == "pos":
        C = cdist(X, Y, metric='euclidean')
    else:
        dot_products = np.clip(np.abs(np.dot(X, Y.T)), 0.0, 1.0)
        C = np.degrees(2 * np.arccos(dot_products))
        
    N, M = C.shape
    D = np.zeros((N, M))
    D[0, :] = C[0, :]
    for i in range(1, N):
        D[i, 0] = D[i-1, 0] + C[i, 0]
        
    for i in range(1, N):
        for j in range(1, M):
            c1 = D[i-1, j-1]
            c2 = D[i-1, j]
            c3 = D[i, j-1]
            min_c = c1
            if c2 < min_c: min_c = c2
            if c3 < min_c: min_c = c3
            D[i, j] = C[i, j] + min_c
            
    j_end = int(np.argmin(D[-1, :]))
    min_cost = float(D[-1, j_end])
    
    path = []
    i = N - 1
    j = j_end
    path.append((i, j))
    while i > 0:
        if j == 0:
            i -= 1
        else:
            c1 = D[i-1, j-1]
            c2 = D[i-1, j]
            c3 = D[i, j-1]
            if c1 <= c2 and c1 <= c3:
                i, j = i-1, j-1
            elif c2 <= c1 and c2 <= c3:
                i, j = i-1, j
            else:
                i, j = i, j-1
        path.append((i, j))
        
    path.reverse()
    return min_cost, path

def extract_nodes(frame: dict) -> dict:
    return frame.get("trackers", frame.get("bones", {}))

def get_joint_keys_for_movement(movement_type: str) -> list:
    """Selecciona las articulaciones a evaluar según la técnica de la tesis."""
    mov = movement_type.lower().strip()
    
    # Técnicas de Mano
    if mov in ["jirugi", "palgup chigi", "palgup_chigi"]:
        return ["hand_r", "hand_l"]
        
    # Técnicas de Pie (Patadas)
    elif mov in ["ap chagi", "ap_chagi", "dollyo chagi", "dollyo_chagi", 
                 "yop chagi", "yop_chagi", "naeryo chagi", "naeryo_chagi",
                 "bandal chagi", "bandal_chagi", "mireo chagi", "mireo_chagi", "chagi"]:
        return ["foot_r", "foot_l"]
        
    # Por defecto evalúa extremidades
    else:
        return ["hand_r", "hand_l", "foot_r", "foot_l"]

def find_worst_moment_pos(master_series, student_series, path):
    max_d = -1.0
    w_m, w_s = 0, 0
    for i, j in path:
        d = np.linalg.norm(np.array(master_series[i]) - np.array(student_series[j]))
        if d > max_d:
            max_d = d
            w_m, w_s = i, j
    return w_m, w_s

def find_worst_moment_rot(master_series, student_series, path):
    max_d = -1.0
    w_m, w_s = 0, 0
    for i, j in path:
        dp = np.clip(np.abs(np.dot(master_series[i], student_series[j])), 0.0, 1.0)
        d = np.degrees(2 * np.arccos(dp))
        if d > max_d:
            max_d = d
            w_m, w_s = i, j
    return w_m, w_s

def safe_quat(q_dict):
    x, y, z, w = q_dict.get("x", 0.0), q_dict.get("y", 0.0), q_dict.get("z", 0.0), q_dict.get("w", 1.0)
    if x == 0 and y == 0 and z == 0 and w == 0:
        w = 1.0
    return R.from_quat([x, y, z, w])

@app.post("/evaluate")
def evaluate_movement(payload: EvaluationRequest):
    start_time = time.time()
    request_id = str(uuid.uuid4())
    try:
        user_frames = payload.user_data.get("frames", [])
        master_frames = payload.master_data.get("frames", [])

        if not user_frames or not master_frames:
            raise HTTPException(status_code=400, detail="Los datos deben contener 'frames'.")

        u_nodes_0 = extract_nodes(user_frames[0])
        m_nodes_0 = extract_nodes(master_frames[0])
        
        scale_user = calculate_torso_length(u_nodes_0)
        scale_master = calculate_torso_length(m_nodes_0)
        
        # Alineación Geométrica basada en la Pelvis
        alignment_rot = calculate_spatial_alignment(user_frames, master_frames)

        joint_keys = get_joint_keys_for_movement(payload.movement_type)
        joint_metrics = {}

        for joint in joint_keys:
            master_pos_series = []
            student_pos_series = []
            master_rot_series = []
            student_rot_series = []

            # Extraer rotación inicial de la articulación para cálculo relativo
            u_nodes_0 = extract_nodes(user_frames[0])
            u_quat_0 = R.from_quat([0,0,0,1])
            if joint in u_nodes_0:
                u_quat_0 = safe_quat(u_nodes_0[joint].get("rotation_quat", u_nodes_0[joint].get("rotation_quaternion", {})))
                
            m_nodes_0 = extract_nodes(master_frames[0])
            m_quat_0 = R.from_quat([0,0,0,1])
            if joint in m_nodes_0:
                m_quat_0 = safe_quat(m_nodes_0[joint].get("rotation_quat", m_nodes_0[joint].get("rotation_quaternion", {})))

            for u_frame in user_frames:
                u_nodes = extract_nodes(u_frame)
                if joint not in u_nodes: continue
                
                up = u_nodes[joint].get("position", {"x":0,"y":0,"z":0})
                u_pelvis = u_nodes.get("pelvis", {}).get("position", {"x":0,"y":0,"z":0})
                
                # Centrado en la pelvis
                u_pos_centered = np.array([up["x"] - u_pelvis["x"], up["y"] - u_pelvis["y"], up["z"] - u_pelvis["z"]])
                # Alineación Z y Escala
                u_pos_aligned = alignment_rot.apply(u_pos_centered)
                student_pos_series.append(u_pos_aligned / scale_user)

                uq_dict = u_nodes[joint].get("rotation_quat", u_nodes[joint].get("rotation_quaternion", {}))
                u_quat = safe_quat(uq_dict)
                
                # Rotación RELATIVA desde el frame 0
                rel_u_quat = u_quat_0.inv() * u_quat
                student_rot_series.append(rel_u_quat.as_quat()) # Usamos as_quat, que devuelve numpy array [x,y,z,w]

            for m_frame in master_frames:
                m_nodes = extract_nodes(m_frame)
                if joint not in m_nodes: continue

                mp = m_nodes[joint].get("position", {"x":0,"y":0,"z":0})
                m_pelvis = m_nodes.get("pelvis", {}).get("position", {"x":0,"y":0,"z":0})
                m_pos_centered = np.array([mp["x"] - m_pelvis["x"], mp["y"] - m_pelvis["y"], mp["z"] - m_pelvis["z"]])
                master_pos_series.append(m_pos_centered / scale_master)

                mq_dict = m_nodes[joint].get("rotation_quat", m_nodes[joint].get("rotation_quaternion", {}))
                m_quat = safe_quat(mq_dict)
                
                # Rotación RELATIVA desde el frame 0
                rel_m_quat = m_quat_0.inv() * m_quat
                master_rot_series.append(rel_m_quat.as_quat())

            if not student_pos_series or not master_pos_series:
                continue

            # Usar Subsequence DTW en lugar de Global DTW
            dist_pos, path_pos = subsequence_dtw(master_pos_series, student_pos_series, mode="pos")
            dist_rot, path_rot = subsequence_dtw(master_rot_series, student_rot_series, mode="rot")
            
            joint_metrics[joint] = {
                "pos_error": dist_pos / len(path_pos),
                "rot_error": dist_rot / len(path_rot),
                "path_pos": path_pos,
                "path_rot": path_rot,
                "master_pos_series": master_pos_series,
                "student_pos_series": student_pos_series,
                "master_rot_series": master_rot_series,
                "student_rot_series": student_rot_series
            }

        if not joint_metrics:
            raise HTTPException(status_code=400, detail="No se encontraron articulaciones válidas para comparar.")

        # UMBRALES DE TOLERANCIA
        ROT_THRESHOLD = 45.0 # 45 Grados de desviación permitida para trackers de VR
        POS_THRESHOLD = 0.40 # 40% de la longitud del torso (Mayor holgura por limitaciones físicas/trackers)
        
        joints_analysis = {}
        global_worst_severity = -1.0
        global_critical_joint = None

        for joint, metrics in joint_metrics.items():
            r_sev = metrics["rot_error"] / ROT_THRESHOLD
            p_sev = metrics["pos_error"] / POS_THRESHOLD
            
            worst_joint_severity = max(r_sev, p_sev)
            passed = worst_joint_severity <= 1.0
            
            if r_sev > p_sev:
                failure_type = "rotación" if not passed else "ninguno"
                w_m, w_s = find_worst_moment_rot(metrics["master_rot_series"], metrics["student_rot_series"], metrics["path_rot"])
                diagnostic = ""
            else:
                failure_type = "trayectoria" if not passed else "ninguno"
                w_m, w_s = find_worst_moment_pos(metrics["master_pos_series"], metrics["student_pos_series"], metrics["path_pos"])
                
                diagnostic = ""
                if not passed:
                    m_pos = metrics["master_pos_series"][w_m]
                    s_pos = metrics["student_pos_series"][w_s]
                    diff = s_pos - m_pos
                    
                    detalles = []
                    # Z (Altura)
                    if diff[2] > 0.15: detalles.append("muy alta")
                    elif diff[2] < -0.15: detalles.append("muy baja")
                    
                    # Y (Profundidad) asumiendo +Y adelante
                    if diff[1] > 0.15: detalles.append("muy estirada al frente")
                    elif diff[1] < -0.15: detalles.append("muy recogida hacia el cuerpo")
                    
                    # X (Apertura)
                    if joint in ["hand_r", "foot_r"]:
                        if diff[0] < -0.15: detalles.append("muy abierta hacia afuera")
                        elif diff[0] > 0.15: detalles.append("cruzada hacia adentro")
                    else:
                        if diff[0] > 0.15: detalles.append("muy abierta hacia afuera")
                        elif diff[0] < -0.15: detalles.append("cruzada hacia adentro")
                        
                    if detalles:
                        diagnostic = " (específicamente: " + ", ".join(detalles) + ")"
                
            frame_data = user_frames[w_s]
            frame_num = frame_data.get("frame", w_s)
            time_sec = frame_data.get("time_sec", 0.0)
            percentage = (w_s / max(1, len(user_frames) - 1)) * 100
            
            joints_analysis[joint] = {
                "passed": passed,
                "max_angular_deviation": round(metrics["rot_error"], 2),
                "max_positional_deviation": round(metrics["pos_error"], 2),
                "failure_type": failure_type,
                "diagnostic": diagnostic,
                "error_percentage_execution": round(percentage, 2),
                "error_frame_exact": frame_num,
                "error_time_sec": round(time_sec, 4)
            }
            
            if worst_joint_severity > global_worst_severity:
                global_worst_severity = worst_joint_severity
                global_critical_joint = joint

        global_passed = global_worst_severity <= 1.0
        
        # Sistema de Puntuación Académico Segmentado (Aún más flexible)
        if global_worst_severity <= 1.0:
            # 0.0 a 1.0 -> 100 a 90 (Excelente)
            score_general = 100.0 - (global_worst_severity * 10.0)
        elif global_worst_severity <= 2.0:
            # 1.0 a 2.0 -> 90 a 75 (Bueno, con errores de hardware / menores)
            score_general = 90.0 - ((global_worst_severity - 1.0) * 15.0)
        elif global_worst_severity <= 3.0:
            # 2.0 a 3.0 -> 75 a 60 (Aceptable con observaciones)
            score_general = 75.0 - ((global_worst_severity - 2.0) * 15.0)
        else:
            # > 3.0 -> 60 a 0 (Técnica deficiente)
            score_general = max(0.0, 60.0 - ((global_worst_severity - 3.0) * 10.0))
        
        # GENERACIÓN DEL TEXTO COMPUESTO
        traduccion = {
            "hand_r": "la mano derecha", "hand_l": "la mano izquierda", 
            "foot_r": "el pie derecho", "foot_l": "el pie izquierdo"
        }
        
        if global_passed:
            if score_general >= 90:
                feedback = f"¡Técnica Excelente! Puntuación: {score_general:.1f}/100. Movimiento y posturas correctos."
            else:
                feedback = f"¡Buena Ejecución! Puntuación: {score_general:.1f}/100. Pasaste la prueba, pero tienes detalles menores."
        else:
            c_joint_data = joints_analysis[global_critical_joint]
            c_name = traduccion.get(global_critical_joint, global_critical_joint)
            c_type = c_joint_data["failure_type"]
            c_perc = c_joint_data["error_percentage_execution"]
            c_diag = c_joint_data.get("diagnostic", "")
            
            if c_perc < 33.3: fase = "al inicio del movimiento"
            elif c_perc < 66.6: fase = "a la mitad del movimiento"
            else: fase = "al final del movimiento"
            
            falla_desc = "un giro incorrecto del ángulo" if c_type == "rotación" else f"una posición incorrecta{c_diag}"
            feedback = f"Puntuación: {score_general:.1f}/100. Necesitas mejorar. Tu principal problema fue {falla_desc} en {c_name} {fase}."
            
            secondary_errors = []
            for j, data in joints_analysis.items():
                if j != global_critical_joint and not data["passed"]:
                    s_name = traduccion.get(j, j)
                    s_type = data["failure_type"]
                    s_perc = data["error_percentage_execution"]
                    s_diag = data.get("diagnostic", "")
                    
                    if s_perc < 33.3: s_fase = "al inicio"
                    elif s_perc < 66.6: s_fase = "a la mitad"
                    else: s_fase = "al final"
                    
                    s_falla = "giro incorrecto" if s_type == "rotación" else f"posición incorrecta{s_diag}"
                    secondary_errors.append(f"{s_falla} en {s_name} {s_fase}")
            
            if secondary_errors:
                if len(secondary_errors) == 1:
                    feedback += f" Además, notamos un error de {secondary_errors[0]}."
                else:
                    feedback += " Adicionalmente fallaron: " + ", ".join(secondary_errors) + "."
                    
            good_joints = [traduccion.get(j, j) for j, data in joints_analysis.items() if data["passed"]]
            if good_joints:
                if len(good_joints) == 1:
                    feedback += f" Por otro lado, {good_joints[0]} se ejecutó bien."
                else:
                    last = good_joints.pop() if len(good_joints) > 1 else ""
                    good_str = ", ".join(good_joints) + " y " + last if last else good_joints[0]
                    feedback += f" Por otro lado, ejecutaste bien: {good_str}."

        duration_ms = round((time.time() - start_time) * 1000, 2)
        differences_count = sum(1 for data in joints_analysis.values() if not data["passed"])
        
        # Generar log estructurado
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "dtw_evaluation_executed",
            "requestId": request_id,
            "algorithm": "subsequence-dtw-biomechanics-v1",
            "durationMs": duration_ms,
            "inputSizeA": len(master_frames),
            "inputSizeB": len(user_frames),
            "similarity": round(score_general / 100.0, 2), # Escala de 0 a 1 como en el ejemplo
            "differencesCount": differences_count
        }
        logger.info(json.dumps(log_data))

        return {
            "success": True,
            "score": round(score_general, 2),
            "feedback": feedback,
            "detailed_metrics": {
                "global_passed": global_passed,
                "critical_failure_joint": global_critical_joint,
                "joints_analysis": joints_analysis
            }
        }

    except HTTPException as e:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.error(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "dtw_evaluation_error",
            "requestId": request_id,
            "durationMs": duration_ms,
            "error": str(e.detail)
        }))
        raise
    except Exception as e:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.error(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "dtw_evaluation_error",
            "requestId": request_id,
            "durationMs": duration_ms,
            "error": str(e)
        }))
        raise HTTPException(status_code=500, detail=str(e))
