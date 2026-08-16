"""Rendering a SMILES to an image.

Lived in `pca_traversal_analysis.py` in the old pipeline, which is why four
plotting modules imported from the analysis layer. It is a renderer; it belongs
here. The analysis decides *which* molecules to draw and writes their SMILES to
`data/`; this turns one into pixels.
"""
from __future__ import annotations

import io
import math
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from rdkit import Chem
from rdkit.Chem import Draw, rdCoordGen
from rdkit.Chem.Draw import rdMolDraw2D


def mol_to_image(smiles: str, size: int = 300) -> Optional[np.ndarray]:
    """RGBA array for `smiles`, or None if RDKit cannot parse it.

    Returning None rather than raising is deliberate: a filmstrip with one
    unparseable step should still be drawn, with that panel labelled, instead of
    the whole figure failing.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2.5
    opts.padding = 0.12
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return np.array(Image.open(io.BytesIO(drawer.GetDrawingText())))


def smiles_to_svg(smiles: str, image_size: Tuple[int, int] = (300, 300)) -> Optional[str]:
    """SVG for one molecule, laid out with rdCoordGen.

    rdCoordGen gives noticeably more uniform 2D coordinates than RDKit's
    default, which matters when a grid puts many molecules side by side.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Warning: could not parse SMILES: {smiles}")
        return None
    try:
        rdCoordGen.AddCoords(mol)
    except Exception as exc:
        print(f"Warning: rdCoordGen failed ({exc}); using default coordinates.")

    drawer = rdMolDraw2D.MolDraw2DSVG(image_size[0], image_size[1])
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _svg_to_pil(svg_data: str, size: Tuple[int, int]) -> Optional[Image.Image]:
    """Best-effort SVG -> PIL. Returns None when no rasterizer is installed, so
    callers can fall back rather than fail."""
    try:
        import cairosvg  # optional dependency
        png = cairosvg.svg2png(bytestring=svg_data.encode("utf-8"),
                               output_width=size[0], output_height=size[1])
        return Image.open(io.BytesIO(png)).convert("RGBA")
    except Exception:
        return None


def render_mol_grid(smiles_list: List[str], mols_per_row: int = 3,
                    sub_img_size: Tuple[int, int] = (180, 180)) -> Image.Image:
    """A grid of molecules as one image.

    Prefers the per-molecule rdCoordGen + SVG path; if SVG rasterization is
    unavailable in this environment (no cairosvg), falls back to RDKit's own
    raster grid for the whole set.
    """
    tiles = []
    for smi in smiles_list:
        svg = smiles_to_svg(smi, image_size=sub_img_size)
        pil = _svg_to_pil(svg, sub_img_size) if svg is not None else None
        if pil is None:
            tiles = []
            break
        tiles.append(pil)

    if tiles:
        ncols = mols_per_row
        nrows = math.ceil(len(tiles) / ncols)
        w, h = sub_img_size
        canvas = Image.new("RGBA", (ncols * w, nrows * h), (255, 255, 255, 255))
        for i, tile in enumerate(tiles):
            r, c = divmod(i, ncols)
            canvas.paste(tile, (c * w, r * h), tile)
        return canvas

    mols = [m for m in (Chem.MolFromSmiles(s) for s in smiles_list) if m is not None]
    img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=sub_img_size,
                               returnPNG=False)
    return img.convert("RGBA") if hasattr(img, "convert") else img
