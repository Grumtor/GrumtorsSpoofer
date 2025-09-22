# -*- coding: utf-8 -*-
# streamlit_video_app.py
# Video Spoofer (Streamlit) — Normal / BW / BW contrasté / Golden Hour (+ Miroir x2 optionnel)
# Export en ZIP (arborescence /Nom/Normal + /Nom/Miroir, ou tout à la racine).
#
# Notes :
# - Requiert ffmpeg (présent sur Streamlit Community Cloud en général).
# - Encodage H.264 (libx264) + AAC. Conteneur .mp4
# - Les effets sont appliqués frame-par-frame via PIL, donc le temps de rendu dépend de la durée/résolution.

import os, io, zipfile, tempfile
from typing import List, Tuple, Dict, Optional, Iterable

import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import streamlit as st
from moviepy.editor import VideoFileClip, vfx

# ---------- Config ----------
SUPPORTED_VIDS = {'.mp4', '.mov', '.m4v', '.webm'}
ALLOWED_ROTATIONS = (0, 90, 180, 270)

# ---------- Effets (sur frame PIL) ----------
def effect_normal(img: Image.Image) -> Image.Image:
    return img

def effect_bw(img: Image.Image) -> Image.Image:
    return img.convert("L").convert("RGB")

def effect_bw_contrast(img: Image.Image) -> Image.Image:
    g = img.convert("L")
    g = ImageOps.autocontrast(g, cutoff=0)
    g = ImageEnhance.Contrast(g).enhance(1.35)
    return g.convert("RGB")

def effect_golden_hour(img: Image.Image) -> Image.Image:
    base = img.convert("RGB")
    r, g, b = base.split()
    r = r.point(lambda i: min(255, int(i * 1.08)))
    g = g.point(lambda i: min(255, int(i * 1.02)))
    b = b.point(lambda i: max(0,   int(i * 0.95)))
    warmed = Image.merge("RGB", (r, g, b))
    warmed = ImageEnhance.Color(warmed).enhance(1.12)
    warmed = ImageEnhance.Contrast(warmed).enhance(1.06)
    warmed = ImageEnhance.Brightness(warmed).enhance(1.03)
    return warmed

def apply_effect_pipeline_pil(img: Image.Image, pipeline: Iterable[str]) -> Image.Image:
    out = img
    for step in pipeline:
        if step == "normal":
            out = effect_normal(out)
        elif step == "bw":
            out = effect_bw(out)
        elif step == "bwcontrast":
            out = effect_bw_contrast(out)
        elif step == "goldenhour":
            out = effect_golden_hour(out)
    return out

def variant_suffix(pipeline: Iterable[str]) -> str:
    parts = list(pipeline)
    return "_" + "_".join(parts) if parts else "_normal"

def generate_variants(e_normal: bool, e_bw: bool, e_bwc: bool, e_gh: bool) -> List[List[str]]:
    variants = []
    if e_normal: variants.append(["normal"])
    if e_bw:     variants.append(["bw"])
    if e_bwc:    variants.append(["bwcontrast"])
    if e_gh:     variants.append(["goldenhour"])
    if not variants: variants.append(["normal"])
    return variants

def choose_preview_pipeline(e_normal: bool, e_bw: bool, e_bwc: bool, e_gh: bool) -> List[str]:
    if e_bwc: return ["bwcontrast"]
    if e_bw:  return ["bw"]
    if e_gh:  return ["goldenhour"]
    if e_normal: return ["normal"]
    return ["normal"]

# ---------- Frame mapper (numpy <-> PIL) ----------
def frame_mapper_factory(pipeline: List[str], mirror: bool) :
    """
    Retourne une fonction f(frame: np.ndarray) -> np.ndarray
    frame: HxWx3 uint8
    """
    def mapper(frame: np.ndarray) -> np.ndarray:
        img = Image.fromarray(frame)
        if mirror:
            img = ImageOps.mirror(img)
        img = apply_effect_pipeline_pil(img, pipeline)
        return np.array(img)
    return mapper

# ---------- Clip processing ----------
def process_clip(
    clip: VideoFileClip,
    pipeline: List[str],
    mirror: bool,
    rotate_deg: int,
    resize_pct: int
) -> VideoFileClip:
    # Rotation (MoviePy applique rotation "visuelle")
    if rotate_deg in ALLOWED_ROTATIONS and rotate_deg != 0:
        clip = clip.fx(vfx.rotate, angle=rotate_deg)  # angle en degrés

    # Resize
    if resize_pct != 100:
        clip = clip.fx(vfx.resize, resize_pct / 100.0)

    # Effets + miroir (par frame)
    mapper = frame_mapper_factory(pipeline, mirror)
    clip = clip.fl_image(mapper)

    return clip

# ---------- App ----------
st.set_page_config(page_title="Photo / Video Spoofer", page_icon="🎬", layout="wide")
st.title("🎬 Grumtor's Spoofer — Photo / Video")

# Session
if "rotation_map" not in st.session_state:
    st.session_state.rotation_map: Dict[str, int] = {}
if "selected_name" not in st.session_state:
    st.session_state.selected_name: Optional[str] = None
if "last_uploaded_names" not in st.session_state:
    st.session_state.last_uploaded_names: List[str] = []

with st.sidebar:
    st.header("Fichiers")
    vids = st.file_uploader(
        "Glissez-déposez vos vidéos (multi)",
        type=[e.strip(".") for e in SUPPORTED_VIDS],
        accept_multiple_files=True
    )
    names = [v.name for v in vids] if vids else []

    if names != st.session_state.last_uploaded_names:
        st.session_state.last_uploaded_names = names
        st.session_state.rotation_map = {n: 0 for n in names}
        if names:
            st.session_state.selected_name = names[0]

    if names:
        st.write(f"**{len(names)} fichiers**")
        selected_name = st.selectbox(
            "Aperçu / réglages du fichier",
            options=names,
            index=names.index(st.session_state.selected_name) if st.session_state.selected_name in names else 0
        )
        st.session_state.selected_name = selected_name

        colr1, colr2 = st.columns(2)
        with colr1:
            if st.button("⟲ 90° CCW"):
                st.session_state.rotation_map[selected_name] = (st.session_state.rotation_map.get(selected_name, 0) - 90) % 360
        with colr2:
            if st.button("⟳ 90° CW"):
                st.session_state.rotation_map[selected_name] = (st.session_state.rotation_map.get(selected_name, 0) + 90) % 360

    st.divider()
    st.header("Réglages export")
    mirror_all = st.checkbox("Miroir x2 à l’export")
    flat_export = st.checkbox("Tout dans un seul dossier")
    resize_export = st.checkbox("Redimensionner (%)")
    export_scale_pct = st.slider("Taille export (%)", 10, 200, 100, 1, disabled=not resize_export)

    st.divider()
    st.header("Effets à exporter")
    eff_normal = st.checkbox("Normal", value=True)
    eff_bw = st.checkbox("Black & White")
    eff_bwc = st.checkbox("Black & White contrasté")
    eff_gh = st.checkbox("Golden Hour (chaud)")
    variants = generate_variants(eff_normal, eff_bw, eff_bwc, eff_gh)

    # Compteur d'export
    count_vids = len(names)
    mirror_states_count = (2 if mirror_all else 1)
    total_exports = count_vids * mirror_states_count * max(1, len(variants))
    st.markdown(f"**À exporter : {total_exports}**")

# ---------- Preview ----------
col_left, col_right = st.columns([5,5], gap="large")

with col_left:
    st.subheader("Aperçu (image clé rapide)")
    if vids and st.session_state.selected_name:
        vfile = next((f for f in vids if f.name == st.session_state.selected_name), None)
        try:
            if vfile:
                # On ouvre depuis un fichier temporaire pour MoviePy
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(vfile.name)[1]) as tmp:
                    tmp.write(vfile.getvalue())
                    tmp_path = tmp.name

                angle = st.session_state.rotation_map.get(vfile.name, 0)
                preview_pipeline = choose_preview_pipeline(eff_normal, eff_bw, eff_bwc, eff_gh)

                with VideoFileClip(tmp_path) as clip:
                    # On prend un frame au milieu
                    t = max(0, clip.duration/2.0 if clip.duration else 0.0)
                    frame = clip.get_frame(t)  # numpy HxWx3
                    # Applique rotation + resize + miroir aperçu ? → On suit réglages export hors "miroir x2"
                    # On garde miroir= False pour l’aperçu (le switch x2 est à l’export)
                    # Redimension pour l’aperçu (optionnel, ici on n'applique pas pour ne pas dégrader l’aperçu)
                    img = Image.fromarray(frame)

                    # rotation
                    if angle in ALLOWED_ROTATIONS and angle != 0:
                        # PIL rotate positive angle is CCW
                        rot_map = {90: 90, 180: 180, 270: 270}  # direct
                        img = img.rotate(rot_map[angle], expand=True)

                    img = apply_effect_pipeline_pil(img, preview_pipeline)

                    # Rendu
                    zoom = st.slider("Zoom aperçu", 25, 200, 100, 5)
                    w, h = img.size
                    new_w = max(1, int(w * (zoom/100.0)))
                    new_h = max(1, int(h * (zoom/100.0)))
                    if (new_w, new_h) != (w, h):
                        img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

                    st.image(img, caption=f"{vfile.name} | Rot {angle}° | Aperçu {''.join(preview_pipeline)}", use_column_width=True)
        except Exception as e:
            st.error(f"Erreur aperçu: {e}")
    else:
        st.info("Ajoutez des vidéos dans la barre latérale pour afficher un aperçu.")

with col_right:
    st.subheader("Export")
    if not vids:
        st.warning("Ajoutez au moins une vidéo.")
    else:
        # Paramètres d'encodage
        st.caption("Encodage: H.264 + AAC (MP4)")
        crf = st.slider("Qualité (CRF H.264, plus bas = meilleure qualité)", 16, 28, 20, 1)
        preset = st.selectbox("Preset ffmpeg", ["ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"], index=5)
        keep_audio = st.checkbox("Conserver l'audio", value=True)

        do_export = st.button("⟱ Exporter en ZIP")
        if do_export:
            try:
                progress = st.progress(0)
                status = st.empty()

                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_STORED) as zf:
                    mirror_states = [False, True] if mirror_all else [False]
                    total_ops = len(vids) * len(mirror_states) * len(variants)
                    done = 0

                    for vfile in vids:
                        base = os.path.splitext(os.path.basename(vfile.name))[0]
                        angle = st.session_state.rotation_map.get(vfile.name, 0)
                        scale_pct = int(export_scale_pct) if resize_export else 100

                        # Sauvegarde l'upload dans un temp file pour MoviePy/ffmpeg
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(vfile.name)[1]) as src_tmp:
                            src_tmp.write(vfile.getvalue())
                            src_path = src_tmp.name

                        # Ouvre clip source une seule fois
                        with VideoFileClip(src_path, audio=keep_audio) as src_clip:
                            fps = src_clip.fps or 25

                            for mstate in mirror_states:
                                for pipeline in variants:
                                    suf = ""
                                    if angle in ALLOWED_ROTATIONS and angle != 0:
                                        suf += f"_rot{angle}"
                                    if flat_export and mstate:
                                        suf += "_mir"
                                    suf += variant_suffix(pipeline)
                                    if scale_pct != 100:
                                        suf += f"_{scale_pct}pct"

                                    out_name = f"{base}{suf}.mp4"
                                    arcdir = "" if flat_export else os.path.join(base, "Miroir" if mstate else "Normal")
                                    arcname = out_name if not arcdir else os.path.join(arcdir, out_name)

                                    # Traite le clip
                                    proc = process_clip(
                                        clip=src_clip,
                                        pipeline=pipeline,
                                        mirror=mstate,
                                        rotate_deg=angle if angle in ALLOWED_ROTATIONS else 0,
                                        resize_pct=scale_pct
                                    )

                                    # Ecrit vers un fichier temp (MoviePy requiert un chemin)
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as dst_tmp:
                                        dst_path = dst_tmp.name

                                    # Export ffmpeg
                                    # audio on/off, bitrate laissé à ffmpeg avec crf/preset
                                    audio_codec = "aac" if keep_audio and src_clip.audio is not None else None
                                    proc.write_videofile(
                                        dst_path,
                                        codec="libx264",
                                        audio=audio_codec is not None,
                                        audio_codec=audio_codec if audio_codec else "aac",
                                        fps=fps,
                                        preset=preset,
                                        ffmpeg_params=["-crf", str(crf)],
                                        verbose=False,
                                        logger=None
                                    )
                                    proc.close()

                                    # Ajoute au ZIP
                                    with open(dst_path, "rb") as f:
                                        data = f.read()
                                        zf.writestr(arcname, data)

                                    # Nettoyage du temp out
                                    try:
                                        os.remove(dst_path)
                                    except Exception:
                                        pass

                                    done += 1
                                    progress.progress(min(1.0, done / max(1, total_ops)))
                                    status.write(f"Export: {arcname}")

                        # Nettoyage du temp in
                        try:
                            os.remove(src_path)
                        except Exception:
                            pass

                progress.progress(1.0)
                status.write("Export terminé ! Téléchargez ci-dessous.")
                zip_buf.seek(0)
                st.download_button(
                    label="⬇️ Télécharger le ZIP",
                    data=zip_buf,
                    file_name="video_spoofer_export.zip",
                    mime="application/zip"
                )
            except Exception as e:
                st.error(f"Erreurs lors de l'export : {e}")
