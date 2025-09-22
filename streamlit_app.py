# -*- coding: utf-8 -*-
# streamlit_app.py
# Image Spoofer (Streamlit) — Normal / BW / BW contrasté / Golden Hour (+ Miroir x2 optionnel)
# Export en ZIP (arborescence ou tout à la racine).
#
# UI :
# - Uploader multiple (drag & drop), sélection d’une image pour aperçu
# - Rotation par image, miroir aperçu, zoom d’aperçu
# - Cases à cocher pour variantes export
# - Redimensionnement (%), sRGB+strip, strip metadata
# - "Miroir x2 à l’export", "Tout dans un seul dossier"
# - Bouton "Exporter" → génère un ZIP téléchargeable (sans écrire sur disque)
#
# Notes :
# - pillow-heif est optionnel (HEIC/HEIF). Si absent, ces formats ne seront pas lisibles.
# - L’export choisit JPEG si source jpg/jpeg, sinon PNG.

import os, io, zipfile
from typing import List, Tuple, Dict, Optional, Iterable
from PIL import Image, ImageOps, ImageEnhance

import streamlit as st

# ---------- Optionnels ----------
HEIF_OK = False
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
    HEIF_OK = True
except Exception:
    HEIF_OK = False

CMS_OK = True
try:
    from PIL import ImageCms
except Exception:
    CMS_OK = False

# ---------- Resampling ----------
try:
    LANCZOS = Image.Resampling.LANCZOS
    BICUBIC = Image.Resampling.BICUBIC
except Exception:
    LANCZOS = Image.LANCZOS
    BICUBIC = Image.BICUBIC

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.heic', '.heif'}

# ---------- Helpers (reprennent ta logique) ----------
def is_heif(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in {'.heic', '.heif'}

def open_image_high_fidelity(file) -> Image.Image:
    """
    file: st.uploaded_file (possède .read() / .getvalue()) ou bytes
    """
    if hasattr(file, "read"):
        data = file.read()
    elif isinstance(file, (bytes, bytearray)):
        data = file
    else:
        raise RuntimeError("Type de fichier non supporté.")
    buf = io.BytesIO(data)
    img = Image.open(buf)
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    return img

def choose_output_format(src_name: str) -> Tuple[str, str]:
    ext = os.path.splitext(src_name)[1].lower()
    if ext in {'.jpg', '.jpeg'}: return '.jpg', 'JPEG'
    return '.png', 'PNG'

def safe_save_to_bytes(img: Image.Image, pil_format: str,
                       strip: bool = True,
                       orig_exif: Optional[bytes] = None,
                       orig_icc: Optional[bytes] = None) -> bytes:
    """
    Sauvegarde en mémoire (bytes). Conserve exif/icc si strip=False.
    """
    out = io.BytesIO()
    kwargs = {}
    if not strip:
        if orig_exif and pil_format == 'JPEG':
            kwargs['exif'] = orig_exif
        if orig_icc:
            kwargs['icc_profile'] = orig_icc

    if pil_format == 'JPEG':
        img = img.convert('RGB')
        img.save(out, format='JPEG', quality=100, optimize=True, subsampling=0, progressive=True, **kwargs)
    elif pil_format == 'PNG':
        if strip:
            from PIL.PngImagePlugin import PngInfo
            pnginfo = PngInfo()
            img.save(out, format='PNG', pnginfo=pnginfo, compress_level=0, **kwargs)
        else:
            img.save(out, format='PNG', compress_level=0, **kwargs)
    else:
        img.save(out, format=pil_format, **kwargs)
    return out.getvalue()

def apply_rotation(img: Image.Image, deg: int) -> Image.Image:
    deg = int(deg) % 360
    if deg == 0: return img
    return img.rotate(deg, expand=True, resample=BICUBIC)

def apply_mirror(img: Image.Image, enabled: bool) -> Image.Image:
    return ImageOps.mirror(img) if enabled else img

def convert_to_srgb_pixels(img: Image.Image, src_icc: Optional[bytes]) -> Image.Image:
    try:
        if CMS_OK and src_icc:
            in_prof = ImageCms.ImageCmsProfile(io.BytesIO(src_icc))
            out_prof = ImageCms.createProfile("sRGB")
            converted = ImageCms.profileToProfile(img, in_prof, out_prof, outputMode='RGB')
            return converted.convert('RGB')
    except Exception:
        pass
    return img.convert('RGB')

# ---- Effets ----
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

def apply_effect_pipeline(img: Image.Image, pipeline: Iterable[str]) -> Image.Image:
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

# ---------- Session state ----------
if "rotation_map" not in st.session_state:
    st.session_state.rotation_map: Dict[str, int] = {}
if "mirror_preview" not in st.session_state:
    st.session_state.mirror_preview = False
if "selected_name" not in st.session_state:
    st.session_state.selected_name: Optional[str] = None
if "last_uploaded_names" not in st.session_state:
    st.session_state.last_uploaded_names: List[str] = []

# ---------- UI ----------
st.set_page_config(page_title="SPOOFER (Streamlit)", page_icon="🖼️", layout="wide")
st.title("SPOOFER — Image Spoofer (Streamlit)")

with st.sidebar:
    st.header("Fichiers")
    files = st.file_uploader(
        "Glissez-déposez vos images (multi)",
        type=[e.strip(".") for e in SUPPORTED_EXTS],
        accept_multiple_files=True
    )
    st.caption("Formats: JPG, PNG, BMP, TIF(F), WEBP, HEIC/HEIF (si pillow-heif installé)")

    names = [f.name for f in files] if files else []
    # Reset rotations si nouvel upload
    if names != st.session_state.last_uploaded_names:
        st.session_state.last_uploaded_names = names
        st.session_state.rotation_map = {n: 0 for n in names}
        if names:
            st.session_state.selected_name = names[0]

    if names:
        st.write(f"**{len(names)} fichiers**")
        selected_name = st.selectbox("Aperçu du fichier", options=names, index=names.index(st.session_state.selected_name) if st.session_state.selected_name in names else 0)
        st.session_state.selected_name = selected_name

        colr1, colr2, colr3 = st.columns([1,1,1])
        with colr1:
            if st.button("⟲ -90°"):
                st.session_state.rotation_map[selected_name] = (st.session_state.rotation_map.get(selected_name, 0) - 90) % 360
        with colr2:
            if st.button("⟳ +90°"):
                st.session_state.rotation_map[selected_name] = (st.session_state.rotation_map.get(selected_name, 0) + 90) % 360
        with colr3:
            st.toggle("Miroir (aperçu)", key="mirror_preview")

    st.divider()
    st.header("Réglages export")
    mirror_all = st.checkbox("Miroir x2 à l’export")
    flat_export = st.checkbox("Tout dans un seul dossier")
    resize_export = st.checkbox("Redimensionner (%)")
    export_scale_pct = st.slider("Taille export (%)", 10, 200, 100, 1, disabled=not resize_export)
    strip_metadata = st.checkbox("Supprimer toutes les métadonnées", value=True)
    convert_srgb_then_strip = st.checkbox("sRGB puis retirer l’ICC", value=False, disabled=not CMS_OK)

    st.caption("Astuce : activer sRGB garantit des couleurs cohérentes sur le web. Si cochée, l’ICC sera retiré.")

    st.divider()
    st.header("Effets à exporter")
    eff_normal = st.checkbox("Normal", value=True)
    eff_bw = st.checkbox("Black & White")
    eff_bwc = st.checkbox("Black & White contrasté")
    eff_gh = st.checkbox("Golden Hour (chaud)")
    variants = generate_variants(eff_normal, eff_bw, eff_bwc, eff_gh)

    # Compteur d'export
    count_imgs = len(names)
    mirror_states_count = (2 if mirror_all else 1)
    total_exports = count_imgs * mirror_states_count * max(1, len(variants))
    st.markdown(f"**À exporter : {total_exports}**")

# ---------- Preview ----------
col_left, col_right = st.columns([5,5], gap="large")

with col_left:
    st.subheader("Aperçu")
    if files and st.session_state.selected_name:
        file = next((f for f in files if f.name == st.session_state.selected_name), None)
        try:
            if file:
                # On ne consomme pas définitivement le buffer de l'UploadedFile
                bytes_for_preview = file.getvalue()
                img = Image.open(io.BytesIO(bytes_for_preview))
                angle = st.session_state.rotation_map.get(file.name, 0)
                img = apply_rotation(img, angle)
                img = apply_mirror(img, st.session_state.mirror_preview)

                pipeline = choose_preview_pipeline(eff_normal, eff_bw, eff_bwc, eff_gh)
                img = apply_effect_pipeline(img, pipeline)

                zoom = st.slider("Zoom aperçu", 25, 200, 100, 5)
                w, h = img.size
                new_w = max(1, int(w * (zoom/100.0)))
                new_h = max(1, int(h * (zoom/100.0)))
                if (new_w, new_h) != (w, h):
                    img = img.resize((new_w, new_h), resample=LANCZOS)

                st.image(img, caption=f"{file.name} | {w}×{h}px | Rot {angle}° | Mir {'ON' if st.session_state.mirror_preview else 'OFF'} | {''.join(pipeline)}", use_column_width=True)
        except Exception as e:
            st.error(f"Erreur aperçu: {e}")
    else:
        st.info("Ajoutez des images dans la barre latérale pour afficher l’aperçu.")

with col_right:
    st.subheader("Export")
    if not files:
        st.warning("Ajoutez au moins une image.")
    else:
        do_export = st.button("⟱ Exporter en ZIP")
        if do_export:
            try:
                progress = st.progress(0)
                status = st.empty()

                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_STORED) as zf:
                    # Préparation des options
                    do_srgb = bool(convert_srgb_then_strip)
                    do_resize = bool(resize_export)
                    scale_pct = max(10, min(200, int(export_scale_pct))) / 100.0
                    do_strip = bool(strip_metadata or convert_srgb_then_strip)

                    mirror_states = [False, True] if mirror_all else [st.session_state.mirror_preview]
                    total_ops = len(files) * len(mirror_states) * len(variants)
                    done = 0

                    for f in files:
                        # lecture bytes sans consommer file pour d'autres usages
                        raw = f.getvalue()
                        src = Image.open(io.BytesIO(raw))
                        src_icc = src.info.get('icc_profile')
                        src_exif = src.info.get('exif')
                        base = os.path.splitext(os.path.basename(f.name))[0]
                        out_ext, pil_fmt = choose_output_format(f.name)

                        angle = st.session_state.rotation_map.get(f.name, 0)
                        if angle not in (0, 90, 180, 270):
                            angle = 0

                        for mstate in mirror_states:
                            for pipeline in variants:
                                img = Image.open(io.BytesIO(raw))
                                img = apply_rotation(img, angle)
                                img = apply_mirror(img, mstate)
                                img = apply_effect_pipeline(img, pipeline)

                                if do_srgb:
                                    img = convert_to_srgb_pixels(img, src_icc)

                                if do_resize and scale_pct != 1.0:
                                    w, h = img.size
                                    img = img.resize((max(1,int(w*scale_pct)), max(1,int(h*scale_pct))), resample=LANCZOS)

                                suf = ""
                                if angle: suf += f"_rot{angle}"
                                if flat_export and mstate:
                                    suf += "_mir"
                                suf += variant_suffix(pipeline)
                                if do_resize and scale_pct != 1.0:
                                    suf += f"_{int(scale_pct*100)}pct"

                                # Chemin dans le ZIP
                                if flat_export:
                                    arcdir = ""
                                else:
                                    arcdir = os.path.join(base, "Miroir" if mstate else "Normal")

                                out_name = f"{base}{suf}{out_ext}"
                                arcname = out_name if not arcdir else os.path.join(arcdir, out_name)

                                # Écriture dans le zip
                                data = safe_save_to_bytes(
                                    img, pil_fmt,
                                    strip=do_strip,
                                    orig_exif=src_exif,
                                    orig_icc=src_icc
                                )
                                zf.writestr(arcname, data)

                                done += 1
                                progress.progress(min(1.0, done / max(1, total_ops)))
                                status.write(f"Export: {arcname}")

                progress.progress(1.0)
                status.write("Export terminé ! Téléchargez ci-dessous.")
                zip_buf.seek(0)
                st.download_button(
                    label="⬇️ Télécharger le ZIP",
                    data=zip_buf,
                    file_name="spoofer_export.zip",
                    mime="application/zip"
                )
            except Exception as e:
                st.error(f"Erreurs lors de l'export : {e}")

st.caption(
    f"HEIC/HEIF activé: {'✅' if HEIF_OK else '❌'} | "
    f"sRGB (ImageCms): {'✅' if CMS_OK else '❌'}"
)
