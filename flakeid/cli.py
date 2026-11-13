"""Command-line interface:  python find_flakes.py IMAGE MATERIAL OXIDE_NM [options]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import DEFAULT_RANGE_NM, find_flakes


def build_parser():
    p = argparse.ArgumentParser(
        prog="find_flakes",
        description="Identify uniform-contrast 2D flakes in stitched optical overview images "
                    "and estimate their thickness from the optical contrast on SiO2/Si.")
    p.add_argument("image", nargs="+",
                   help="image file(s): .jpg/.png/.tif/.vsi, or a Micro-Manager tile acquisition folder")
    p.add_argument("material", help="exfoliated material: graphite | hbn | auto (a Micro-Manager scan by the scan program: "
                                    "the material entered at the microscope, from scan_params.json)")
    p.add_argument("oxide_nm", help="SiO2 thickness in nm (e.g. 90 or 285), or auto (as entered at the microscope)")
    p.add_argument("--range", nargs=2, type=float, metavar=("MIN", "MAX"),
                   help="thickness window of interest in nm (default: graphite 0.3-25, hbn 0-100)")
    p.add_argument("--min-area", type=float, default=100.0, help="minimum flake area in um^2 (default 100)")
    p.add_argument("--pixel-size", type=float, help="um per pixel (default: from .vsi metadata or 1.005)")
    p.add_argument("--scale", type=float, default=0.5,
                   help="processing resolution relative to the image (default 0.5; 1.0 = full res, slower)")
    p.add_argument("--profile", help="camera calibration profile (name in flakeid/profiles or JSON path)")
    p.add_argument("--no-refine-oxide", action="store_true",
                   help="use the given oxide thickness as-is instead of refining it (+/-10%%) from the image")
    p.add_argument("--max-residual", type=float, default=0.20, help="max colour-fit residual; the limit used is max(this, 0.25*|contrast|) (default 0.20)")
    p.add_argument("--max-edge-width", type=float, default=5.0,
                   help="max 20-80%% edge width in um; rejects diffuse smudges/defocus (default 5)")
    p.add_argument("--min-contrast", type=float,
                   help="min |RGB contrast| for a reliable flake (default graphite 0.06, hbn 0.15 ~ 3.5 nm)")
    p.add_argument("--min-solidity", type=float, default=0.6,
                   help="min area/convex-hull ratio; rejects ragged debris (default 0.6)")
    p.add_argument("--flat", help="flat field (.npz from 'python -m flakeid.mm flat') for Micro-Manager tiles")
    p.add_argument("--stage-origin", nargs=2, type=float, metavar=("X", "Y"),
                   help="stage coordinates (um) of the image top-left pixel (default: from .vsi, if present)")
    p.add_argument("--stage-sign", nargs=2, type=float, default=(1.0, 1.0), metavar=("SX", "SY"),
                   help="+1/-1: direction of stage X/Y relative to image +x (right) / +y (down)")
    p.add_argument("--all", action="store_true",
                   help="also list rejected regions (out of range, poor fit, ...) with their flags")
    p.add_argument("-o", "--out-dir", help="output directory (default: next to the image)")
    p.add_argument("--viewer", action="store_true",
                   help="write a zoomable multi-resolution viewer (<name>_viewer/index.html) with a summary JPEG")
    p.add_argument("--annotate", action="store_true", help="write an annotated overview JPEG")
    p.add_argument("--annotate-scale", type=float, default=0.5, help="size of annotated image (default 0.5)")
    p.add_argument("--crops", action="store_true", help="write a full-resolution crop of every flake")
    p.add_argument("--json", action="store_true", help="also write a JSON file with metadata + flakes")
    p.add_argument("--keep-tiles", action="store_true",
                   help="Micro-Manager folders: keep the tile TIFFs.  By default they are deleted once every output and "
                        "the --viewer pyramid are written (metadata.txt files are kept; without --viewer nothing is "
                        "deleted).  Keep them while the models are being developed (a new material, calibration work): "
                        "deleted tiles cannot be analysed again")
    p.add_argument("--delete-tiles", action="store_true", help=argparse.SUPPRESS)      # (the default now; still accepted)
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def delete_tiles(folder: Path, written, viewer_dir: Path, quiet=False):
    """Remove the tile TIFFs of a Micro-Manager acquisition folder after checking the outputs exist."""
    from .mm import is_dataset
    if not (folder.is_dir() and is_dataset(folder)):
        return                     # (not a Micro-Manager tile folder: nothing to delete)
        return
    missing = [str(w) for w in written if not (Path(w).is_file() and Path(w).stat().st_size > 0)]
    if not (viewer_dir / "index.html").is_file() or next(viewer_dir.glob("tiles/*/*/*.jpg"), None) is None:
        missing.append(f"{viewer_dir} pyramid")
    if missing:
        print(f"tile TIFFs kept: outputs missing ({', '.join(missing)})", file=sys.stderr)
        return
    tifs = [t for t in folder.glob("*/*.tif") if (t.parent / "metadata.txt").is_file()]
    size = sum(t.stat().st_size for t in tifs)
    for t in tifs:
        t.unlink()
    (folder / "TILES_DELETED.txt").write_text(
        f"{len(tifs)} tile TIFFs ({size / 1e9:.2f} GB) deleted after analysis (use --keep-tiles to keep them).\n"
        f"Results and viewer: {viewer_dir.parent}\nmetadata.txt files (stage positions, Z, camera) are kept.\n")
    if not quiet:
        print(f"  deleted {len(tifs)} tile TIFFs ({size / 1e9:.2f} GB) from {folder}")


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    args.delete_tiles = not args.keep_tiles and args.viewer      # the viewer pyramid is what is kept of the images
    rc = 0
    for img_path in args.image:
        img_path = Path(img_path)
        if img_path.stem.endswith("_flakes"):      # our own annotated output (e.g. from a glob)
            continue
        out_dir = Path(args.out_dir) if args.out_dir else img_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = img_path.name if img_path.is_dir() else img_path.stem + ("_vsi" if img_path.suffix.lower() == ".vsi" else "")
        try:
            from .io import load_image
            from .mm import resolve_material
            material, oxide_nm = resolve_material(img_path, args.material, args.oxide_nm)
            from .thickness import model_oxide_nm
            oxide_nm = model_oxide_nm(oxide_nm, log=print)        # nominal wafer oxide -> modelled (oxide_nominal.json)
            if "auto" in (args.material, args.oxide_nm):
                print(f"{img_path.name}: {material} on {oxide_nm:g} nm oxide (from the scan parameters)")
            img = load_image(img_path, pixel_um=args.pixel_size, flat=args.flat)
            res = find_flakes(
                img, material, oxide_nm,
                thickness_range_nm=tuple(args.range) if args.range else None,
                min_area_um2=args.min_area, work_scale=args.scale, profile=args.profile,
                refine_oxide_thickness=not args.no_refine_oxide,
                max_residual=args.max_residual, max_edge_width_um=args.max_edge_width,
                min_contrast=args.min_contrast, min_solidity=args.min_solidity,
                stage_origin_um=tuple(args.stage_origin) if args.stage_origin else None,
                stage_axis_sign=tuple(args.stage_sign), include_rejected=args.all,
                keep_contrast=args.viewer,
                verbose=not args.quiet)
        except Exception as exc:  # keep going with the other images
            print(f"ERROR processing {img_path}: {exc}", file=sys.stderr)
            rc = 1
            continue

        csv_path = out_dir / f"{stem}_flakes.csv"
        tpath = res.to_csv(csv_path)
        written = [csv_path, tpath]
        if args.json:
            res.to_json(out_dir / f"{stem}_flakes.json")
            written.append(out_dir / f"{stem}_flakes.json")
        if args.annotate or args.crops:
            from .annotate import annotate
            ann = out_dir / f"{stem}_flakes.jpg"
            annotate(img.to_uint8(), res, ann, scale=args.annotate_scale,
                     crops_dir=(out_dir / f"{stem}_flake_crops") if args.crops else None)
            if args.annotate:
                written.append(ann)
        if args.viewer:
            from .zoomview import build_viewer
            vdir = out_dir / f"{stem}_viewer"
            build_viewer(img, res, vdir, verbose=False)
            written.append(vdir / "index.html")
        if not args.quiet:
            n_good = sum(f.score >= 0.5 for f in res.flakes if f.score == f.score)
            print(f"{img_path.name}: {len(res.flakes)} flakes, {len(res.terraces)} terraces ({n_good} flakes scored >= 0.5); "
                  f"oxide used {res.oxide_nm_used:.1f} nm")
            for w in written:
                print(f"  wrote {w}")
        if args.delete_tiles:
            delete_tiles(img_path, written, out_dir / f"{stem}_viewer", quiet=args.quiet)
        del res, img
    return rc


if __name__ == "__main__":
    sys.exit(main())
