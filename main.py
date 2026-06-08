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
    """Extrae el ángulo Yaw (rotación en eje Z) del cuaternión en radianes."""
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
    """Calcula la matriz de rotación en Z para alinear el alumno con el maestro."""
    try:
        u_head_0 = student_frames[0].get("trackers", student_frames[0].get("bones", {})).get("head", {})
        m_head_0 = master_frames[0].get("trackers", master_frames[0].get("bones", {})).get("head", {})
        
        uq = u_head_0.get("rotation_quat", u_head_0.get("rotation_quaternion", {"x":0,"y":0,"z":0,"w":1}))
        mq = m_head_0.get("rotation_quat", m_head_0.get("rotation_quaternion", {"x":0,"y":0,"z":0,"w":1}))
        
        u_yaw = extract_yaw_from_quat(uq)
        m_yaw = extract_yaw_from_quat(mq)
        
        delta_yaw = m_yaw - u_yaw
        return R.from_euler('z', delta_yaw, degrees=False)
    except Exception:
        return R.from_euler('z', 0, degrees=False)

def unwrap_euler_series(euler_series: list) -> np.ndarray:
    arr = np.array(euler_series)
    arr_rad = np.deg2rad(arr)
    arr_rad_unwrapped = np.unwrap(arr_rad, axis=0)
    return np.rad2deg(arr_rad_unwrapped)

def extract_nodes(frame: dict) -> dict:
    return frame.get("trackers", frame.get("bones", {}))

def get_joint_keys_for_movement(movement_type: str) -> list:
    mov = movement_type.lower()
    if mov == "jirugi":
        return ["head", "hand_r", "hand_l"]
    elif mov in ["ap_chagi", "dollyo_chagi", "yop_chagi", "chagi"]:
        return ["head", "foot_r", "foot_l"]
    else:
        return ["head", "hand_r", "hand_l", "foot_r", "foot_l"]

def find_worst_moment(master_series, student_series, path):
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

        u_nodes_0 = extract_nodes(user_frames[0])
        m_nodes_0 = extract_nodes(master_frames[0])
        
        scale_user = calculate_torso_length(u_nodes_0)
        scale_master = calculate_torso_length(m_nodes_0)
        
        # Alineación Geométrica de Coordenadas de Mundo
        alignment_rot = calculate_spatial_alignment(user_frames, master_frames)

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
                u_pelvis = u_nodes.get("pelvis", {}).get("position", {"x":0,"y":0,"z":0})
                
                # Centrado en la pelvis
                u_pos_centered = np.array([up["x"] - u_pelvis["x"], up["y"] - u_pelvis["y"], up["z"] - u_pelvis["z"]])
                # Alineación Z y Escala
                u_pos_aligned = alignment_rot.apply(u_pos_centered)
                student_pos_series.append(u_pos_aligned / scale_user)

                uq = u_nodes[joint].get("rotation_quat", u_nodes[joint].get("rotation_quaternion", {"x":0,"y":0,"z":0,"w":1}))
                qw = uq.get("w", 1.0)
                if uq.get("x",0) == 0 and uq.get("y",0) == 0 and uq.get("z",0) == 0 and qw == 0: qw = 1.0
                
                # Alinear la rotación de la articulación con el mundo
                u_quat = R.from_quat([uq.get("x",0), uq.get("y",0), uq.get("z",0), qw])
                aligned_u_quat = alignment_rot * u_quat
                student_rot_series.append(aligned_u_quat.as_euler('xyz', degrees=True))

            for m_frame in master_frames:
                m_nodes = extract_nodes(m_frame)
                if joint not in m_nodes: continue

                mp = m_nodes[joint].get("position", {"x":0,"y":0,"z":0})
                m_pelvis = m_nodes.get("pelvis", {}).get("position", {"x":0,"y":0,"z":0})
                m_pos_centered = np.array([mp["x"] - m_pelvis["x"], mp["y"] - m_pelvis["y"], mp["z"] - m_pelvis["z"]])
                master_pos_series.append(m_pos_centered / scale_master)

                mq = m_nodes[joint].get("rotation_quat", m_nodes[joint].get("rotation_quaternion", {"x":0,"y":0,"z":0,"w":1}))
                qw = mq.get("w", 1.0)
                if mq.get("x",0) == 0 and mq.get("y",0) == 0 and mq.get("z",0) == 0 and qw == 0: qw = 1.0
                m_quat = R.from_quat([mq.get("x",0), mq.get("y",0), mq.get("z",0), qw])
                master_rot_series.append(m_quat.as_euler('xyz', degrees=True))

            if not student_pos_series or not master_pos_series:
                continue

            student_rot_series = unwrap_euler_series(student_rot_series)
            master_rot_series = unwrap_euler_series(master_rot_series)

            dist_pos, path_pos = fastdtw(master_pos_series, student_pos_series, dist=euclidean)
            dist_rot, path_rot = fastdtw(master_rot_series, student_rot_series, dist=euclidean)
            
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
        ROT_THRESHOLD = 15.0 
        POS_THRESHOLD = 0.15 
        
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
                worst_idx = find_worst_moment(metrics["master_rot_series"], metrics["student_rot_series"], metrics["path_rot"])
            else:
                failure_type = "trayectoria" if not passed else "ninguno"
                worst_idx = find_worst_moment(metrics["master_pos_series"], metrics["student_pos_series"], metrics["path_pos"])
                
            frame_data = user_frames[worst_idx]
            frame_num = frame_data.get("frame", worst_idx)
            time_sec = frame_data.get("time_sec", 0.0)
            percentage = (worst_idx / max(1, len(user_frames) - 1)) * 100
            
            joints_analysis[joint] = {
                "passed": passed,
                "max_angular_deviation": round(metrics["rot_error"], 2),
                "max_positional_deviation": round(metrics["pos_error"], 2),
                "failure_type": failure_type,
                "error_percentage_execution": round(percentage, 2),
                "error_frame_exact": frame_num,
                "error_time_sec": round(time_sec, 4)
            }
            
            if worst_joint_severity > global_worst_severity:
                global_worst_severity = worst_joint_severity
                global_critical_joint = joint

        global_passed = global_worst_severity <= 1.0
        score_general = max(0.0, 100.0 - (global_worst_severity * 20.0))
        
        # GENERACIÓN DEL TEXTO COMPUESTO
        traduccion = {
            "head": "la cabeza", "hand_r": "la mano derecha", "hand_l": "la mano izquierda", 
            "foot_r": "el pie derecho", "foot_l": "el pie izquierdo"
        }
        
        if global_passed:
            feedback = "¡Técnica excelente! Movimiento y posturas correctos."
        else:
            c_joint_data = joints_analysis[global_critical_joint]
            c_name = traduccion.get(global_critical_joint, global_critical_joint)
            c_type = c_joint_data["failure_type"]
            c_perc = c_joint_data["error_percentage_execution"]
            
            if c_perc < 33.3: fase = "al inicio del movimiento"
            elif c_perc < 66.6: fase = "a la mitad del movimiento"
            else: fase = "al final del movimiento"
            
            feedback = f"Técnica incorrecta. Tu principal problema fue el error de {c_type} en {c_name} {fase}."
            
            secondary_errors = []
            for j, data in joints_analysis.items():
                if j != global_critical_joint and not data["passed"]:
                    s_name = traduccion.get(j, j)
                    s_type = data["failure_type"]
                    s_perc = data["error_percentage_execution"]
                    if s_perc < 33.3: s_fase = "al inicio"
                    elif s_perc < 66.6: s_fase = "a la mitad"
                    else: s_fase = "al final"
                    secondary_errors.append(f"{s_type} en {s_name} {s_fase}")
            
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

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
