import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import pandas as pd
import time
import io
import plotly.graph_objects as go
import subprocess

# ==========================
# PARAMETER MODEL & CONFIG
# ==========================
CONF_THRESHOLD = 0.45

st.set_page_config(
    page_title="ITS Traffic Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS Styling
st.markdown("""
<style>
header[data-testid="stHeader"] { display: none; }
.block-container {
    padding-top: 0.5rem;
    padding-left: 1rem;
    padding-right: 1rem;
    max-width: 100%;
}
.sticky-header {
    position: sticky;
    top: 0;
    z-index: 999;
    background: white;
    padding: 8px 0;
    border-bottom: 2px solid #e2e8f0;
    margin-bottom: 10px;
}
.sticky-header h2 {
    margin: 0;
    padding: 0;
    font-size: 1.8rem;
    text-align: center;
    color: #1E3A8A;
}
.sticky-header p {
    margin: 0;
    padding: 0;
    font-size: 0.9rem;
    text-align: center;
    color: #64748B;
}
div[data-testid="stMetric"] {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
div[data-testid="stMetricValue"] {
    font-size: 1.3rem !important;
    color: #1E3A8A;
}
div[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
}
hr { margin: 0.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# Header Utama
st.markdown("""
<div class="sticky-header">
    <h2>INTELLIGENT TRANSPORT SYSTEM DASHBOARD</h2>
    <p>Sistem Vehicle Counting & Analisis Lalu Lintas Berbasis Deep Learning</p>
</div>
""", unsafe_allow_html=True)

# ==========================
# INIT SESSION STATE
# ==========================
if "processed" not in st.session_state:
    st.session_state.processed = False
if "output_video_path" not in st.session_state:
    st.session_state.output_video_path = None
if "excel_bytes" not in st.session_state:
    st.session_state.excel_bytes = None
if "traffic_log" not in st.session_state:
    st.session_state.traffic_log = []
if "interval_log" not in st.session_state:
    st.session_state.interval_log = []
if "summary_data" not in st.session_state:
    st.session_state.summary_data = {}
if "info_data" not in st.session_state:
    st.session_state.info_data = {}
if "is_portrait" not in st.session_state:
    st.session_state.is_portrait = False

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model YOLO: {e}. Pastikan file 'models/best.pt' tersedia.")

# ==========================
# METRIC CARDS TOP BAR
# ==========================
m1, m2, m3, m4, m5, m6 = st.columns(6)
metric_total = m1.empty()
metric_car = m2.empty()
metric_motor = m3.empty()
metric_bus = m4.empty()
metric_truck = m5.empty()
metric_fps = m6.empty()

def update_metrics(total=0, car=0, motor=0, bus=0, truck=0, fps=0):
    metric_total.metric("📊 TOTAL", total)
    metric_car.metric("🚗 MOBIL", car)
    metric_motor.metric("🏍️ MOTOR", motor)
    metric_bus.metric("🚌 BUS", bus)
    metric_truck.metric("🚛 TRUK", truck)
    metric_fps.metric("⚡ FPS", f"{fps:.1f}" if isinstance(fps, (int, float)) else fps)

if st.session_state.processed:
    s = st.session_state.summary_data
    update_metrics(s.get("total", 0), s.get("car", 0), s.get("motor", 0), s.get("bus", 0), s.get("truck", 0), s.get("fps", 0))
else:
    update_metrics()

st.divider()

# ==========================
# HELPER: FUNGSI BIKIN GRAFIK GARIS
# ==========================
def create_line_chart(df_interval):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_interval["Detik"],
        y=df_interval["Kendaraan Melintas"],
        mode='lines+markers',
        name='Jumlah Kendaraan',
        line=dict(color='#1E3A8A', width=3, shape='spline'),
        marker=dict(size=6, color='#2563EB'),
        fill='tozeroy',
        fillcolor='rgba(30, 58, 138, 0.1)'
    ))
    fig.update_layout(
        title="<b>Rekap Volume Kendaraan Per 2 Detik</b>",
        xaxis_title="Detik ke-",
        yaxis_title="Jumlah Kendaraan",
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        hovermode="x unified",
        template="plotly_white"
    )
    return fig

# ==========================
# LAYOUT UTAMA (SPLIT SCREEN)
# ==========================
left, right = st.columns([1, 2])

with left:
    st.subheader("⚙️ Panel Kontrol")
    uploaded_file = st.file_uploader("Upload Video Lalu Lintas", type=["mp4", "avi", "mov", "mkv","CRDOWNLOAD"])
    
    # Reset total jika tombol close [X] diklik pada file uploader
    if uploaded_file is None and st.session_state.processed:
        st.session_state.processed = False
        st.session_state.output_video_path = None
        st.session_state.excel_bytes = None
        st.session_state.traffic_log = []
        st.session_state.interval_log = []
        st.session_state.summary_data = {}
        st.session_state.info_data = {}
        st.rerun()

    line_pct = st.slider("Posisi Virtual Line (%)", 10, 90, 50, 5)
    
    process_button = st.button("🚀 PROSES VIDEO", use_container_width=True, type="primary")
    status_box = st.empty()
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    st.divider()
    excel_download_placeholder = st.empty()
    video_download_placeholder = st.empty()

with right:
    st.subheader("📺 Monitoring & Visualisasi")
    preview_video = st.empty()
    
    # TAMPILAN PREVIEW SEBELUM PROSES (OVERLAY VIRTUAL LINE SAMAR)
    if uploaded_file is not None and not st.session_state.processed:
        # Gunakan tempfile biasa agar tidak mengunci handle
        temp_dir_prev = tempfile.mkdtemp()
        temp_preview_path = os.path.join(temp_dir_prev, "preview_temp.mp4")
        
        with open(temp_preview_path, "wb") as f:
            f.write(uploaded_file.read())
        uploaded_file.seek(0)
        
        cap_prev = cv2.VideoCapture(temp_preview_path)
        ret_prev, frame_prev = cap_prev.read()
        if ret_prev:
            h_prev, w_prev, _ = frame_prev.shape
            line_y_prev = int(h_prev * line_pct / 100)
            
            # Buat overlay samar (dimmed/opacity)
            dimmed_frame = cv2.addWeighted(frame_prev, 0.45, np.zeros_like(frame_prev), 0.55, 0)
            cv2.line(dimmed_frame, (0, line_y_prev), (w_prev, line_y_prev), (0, 0, 255), 3)
            cv2.putText(dimmed_frame, f"Garis Virtual ({line_pct}%)", (20, line_y_prev - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            preview_video.image(
                cv2.cvtColor(dimmed_frame, cv2.COLOR_BGR2RGB), 
                caption="Preview Garis Virtual (Sesuaikan slider di kiri)", 
                use_container_width=True
            )
            
        cap_prev.release()
        
        # Hapus file dan folder temporary dengan aman (mengabaikan error penguncian OS)
        try:
            os.remove(temp_preview_path)
            os.rmdir(temp_dir_prev)
        except Exception:
            pass  # Biarkan OS/Streamlit membersihkan sisa temp file saat restart
            
    elif not st.session_state.processed:
        preview_video.info("Silakan upload video lalu atur posisi garis virtual.")

# ==========================
# PROSES PENGOLAHAN VIDEO
# ==========================
if uploaded_file is not None and process_button:
    status_box.warning("Mempersiapkan pemrosesan video...")
    
    temp_dir = tempfile.mkdtemp()
    input_video_path = os.path.join(temp_dir, "input.mp4")
    output_video_path = os.path.join(temp_dir, "hasil_deteksi.mp4")
    
    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.read())
        
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video <= 0: fps_video = 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Deteksi orientasi video
    st.session_state.is_portrait = height > width
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps_video, (width, height))
    
    line_y = int(height * line_pct / 100)
    crossed_ids = set()
    previous_center = {}
    counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
    
    traffic_log = []
    interval_log = []
    
    interval_seconds = 2
    interval_count = 0
    frame_number = 0
    start_time = time.time()
    
    status_box.info("Sedang memproses deteksi YOLO...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_number += 1
        progress = int((frame_number / total_frames) * 100)
        progress_bar.progress(progress)
        
        elapsed = time.time() - start_time
        fps_process = frame_number / elapsed if elapsed > 0 else 0
        remaining = ((total_frames - frame_number) / fps_process) if fps_process > 0 else 0
        
        progress_text.markdown(f"**Progress:** {progress}% | **Frame:** {frame_number}/{total_frames} | **Speed:** {fps_process:.1f} FPS | **Sisa:** {remaining:.1f}s")
        
        results = model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            conf=CONF_THRESHOLD, imgsz=640, verbose=False, device="cpu"
        )
        
        cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 3)
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().numpy()
            classes = results[0].boxes.cls.int().cpu().numpy()
            
            for box, obj_id, cls_id in zip(boxes, ids, classes):
                x1, y1, x2, y2 = box
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                class_name = model.names[cls_id].lower()
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                cv2.putText(frame, f"{class_name} #{obj_id}", (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if obj_id in previous_center:
                    old_y = previous_center[obj_id]
                    if (old_y < line_y <= cy) or (cy <= line_y < old_y):
                        if obj_id not in crossed_ids:
                            crossed_ids.add(obj_id)
                            if class_name in counts:
                                counts[class_name] += 1
                                interval_count += 1
                                current_second = int(frame_number / fps_video)
                                
                                traffic_log.append({
                                    "No": len(traffic_log) + 1,
                                    "Detik": current_second,
                                    "Waktu": time.strftime("%H:%M:%S", time.gmtime(current_second)),
                                    "ID": int(obj_id),
                                    "Kategori": class_name.upper()
                                })
                previous_center[obj_id] = cy
        
        # LOGIKA PER 2 DETIK (REKAP INTERVAL FIX)
        if frame_number % int(fps_video * interval_seconds) == 0:
            interval_log.append({
                "Detik": int(frame_number / fps_video),
                "Kendaraan Melintas": interval_count
            })
            interval_count = 0  # Reset counter interval saja
            
        update_metrics(sum(counts.values()), counts["car"], counts["motorcycle"], counts["bus"], counts["truck"], fps_process)
        writer.write(frame)
        
    cap.release()
    writer.release()

    preview_path = output_video_path.replace(".mp4", "_preview.mp4")

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", output_video_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        preview_path
    ])

    st.session_state.preview_video_path = preview_path
    total_time = time.time() - start_time
    
    # Jika ada sisa detik terakhir yang belum tercatat di interval
    if frame_number % int(fps_video * interval_seconds) != 0:
        interval_log.append({
            "Detik": int(frame_number / fps_video),
            "Kendaraan Melintas": interval_count
        })

    # GENERATE REPORT EXCEL DENGAN MULTI-SHEET
    df_log = pd.DataFrame(traffic_log) if traffic_log else pd.DataFrame([{"No": "-", "Detik": "-", "Waktu": "-", "ID": "-", "Kategori": "NIHIL"}])
    df_interval = pd.DataFrame(interval_log) if interval_log else pd.DataFrame([{"Detik": 0, "Kendaraan Melintas": 0}])
    df_summary = pd.DataFrame({
        "Kategori": ["Mobil", "Motor", "Bus", "Truk", "TOTAL"],
        "Jumlah": [counts["car"], counts["motorcycle"], counts["bus"], counts["truck"], sum(counts.values())]
    })
    info_df = pd.DataFrame({
        "Parameter": ["Durasi Video (detik)", "Jumlah Frame", "FPS Video", "Waktu Proses (detik)", "Confidence", "Virtual Line (%)"],
        "Nilai": [round(total_frames / fps_video, 2), total_frames, round(fps_video, 2), round(total_time, 2), CONF_THRESHOLD, line_pct]
    })
    
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer_excel:
        df_log.to_excel(writer_excel, sheet_name="Log Kendaraan", index=False)
        df_interval.to_excel(writer_excel, sheet_name="Rekap Interval 2s", index=False)
        df_summary.to_excel(writer_excel, sheet_name="Ringkasan Total", index=False)
        info_df.to_excel(writer_excel, sheet_name="Parameter Informasi", index=False)
        
    # SIMPAN HASIL KEDALAM SESSION STATE
    st.session_state.processed = True
    st.session_state.output_video_path = output_video_path
    st.session_state.excel_bytes = excel_buffer.getvalue()
    st.session_state.traffic_log = traffic_log
    st.session_state.interval_log = interval_log
    st.session_state.summary_data = {
        "total": sum(counts.values()), "car": counts["car"], "motor": counts["motorcycle"],
        "bus": counts["bus"], "truck": counts["truck"], "fps": fps_process
    }
    st.session_state.info_data = {
        "duration": round(total_frames / fps_video, 1),
        "proc_time": round(total_time, 1),
        "frames": total_frames,
        "fps_vid": round(fps_video, 1),
        "conf": CONF_THRESHOLD,
        "line": f"{line_pct}%"
    }
    
    status_box.success("Analisis Video Selesai!")

# ==========================
# MENAMPILKAN HASIL ANALISIS
# ==========================
if st.session_state.processed:
    # Tombol Download di Panel Kiri (Aman dari Refresh Reset)
    with left:
        excel_download_placeholder.download_button(
            label="📊 Download Laporan CSV/Excel",
            data=st.session_state.excel_bytes,
            file_name=f"Laporan_Traffic_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        with open(st.session_state.output_video_path, "rb") as f:
            v_bytes = f.read()
        video_download_placeholder.download_button(
            label="🎥 Download Video Hasil",
            data=v_bytes,
            file_name="Hasil_Deteksi.mp4",
            mime="video/mp4",
            use_container_width=True
        )

    # Player Video Hasil di Panel Kanan
    with right:
        if (
            "preview_video_path" in st.session_state
            and os.path.exists(st.session_state.preview_video_path)
        ):
            preview_video.video(st.session_state.preview_video_path)

    st.divider()

    # PENYESUAIAN LAYOUT DINAMIS (PORTRAIT VS LANDSCAPE)
    df_interval = pd.DataFrame(st.session_state.interval_log)
    fig_chart = create_line_chart(df_interval)
    info = st.session_state.info_data

    if st.session_state.is_portrait:
        # Video Portrait: Parameter di Kiri, Grafik di Kanan
        col_param, col_chart = st.columns([1, 2])
        with col_param:
            st.markdown("### 📋 Parameter Analisis")
            st.json(info)
        with col_chart:
            st.plotly_chart(fig_chart, use_container_width=True)
    else:
        # Video Landscape: Parameter & Ringkasan di Atas, Grafik Lebar di Bawah
        col_p1, col_p2 = st.columns([1, 1])
        with col_p1:
            st.markdown("### 📌 Summary Parameter")
            p_df = pd.DataFrame(list(info.items()), columns=["Parameter", "Nilai"])
            st.dataframe(p_df, use_container_width=True, hide_index=True)
        with col_p2:
            st.markdown("### 📈 Visualisasi Tren")
            st.plotly_chart(fig_chart, use_container_width=True)
