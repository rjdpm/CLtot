import math, os
import numpy as np
from collections import OrderedDict
from ordered_set import OrderedSet
from typing import List, Dict, Tuple, Union, Callable, Optional

import io
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from mordred import Calculator, descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Draw import rdMolDraw2D

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import warnings
warnings.filterwarnings("ignore", module="mordred")

__all__ = [
    'create_descriptor_functions',
    'compute_descriptors_from_name',
    'plot_smiles_grid'
    ]

def create_folder(folder_name: str) -> str:
    
    if len(folder_name):
        if not os.path.isdir(folder_name):
            print(f'Creating Folder:{folder_name}')
            os.makedirs(folder_name)
        
    return folder_name

# Descriptor Setup
def create_descriptor_functions(descriptor_names: List[str]) -> Tuple[List[Callable], Calculator]:
    """Creates RDKit and Mordred descriptor function sets."""
    RDKit_descriptor_funcs = OrderedDict({
        k: v for k, v in Descriptors.__dict__.items()
        if k in descriptor_names and callable(v)
    })

    mordred_calc_all = Calculator(descriptors, ignore_3D=True)
    mordred_names = list(OrderedSet(descriptor_names) - OrderedSet(RDKit_descriptor_funcs.keys()))
    mordred_selected = [d for d in mordred_calc_all.descriptors if str(d) in mordred_names]
    mordred_calc = Calculator(mordred_selected, ignore_3D=True)

    return RDKit_descriptor_funcs, mordred_calc


def compute_descriptors_from_name(smiles: str,
                        descriptor_names: List[str]
                        ) -> Dict[str, float] | None:
    """Computes descriptors for a single SMILES string."""
    try:
        RDKit_descriptor_funcs, mordred_calc = create_descriptor_functions(descriptor_names)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f'Invalid SMILES Error...')
            return None
        rdkit_vals = {name: func(mol) for name, func in RDKit_descriptor_funcs.items()}
        mordred_vals = mordred_calc(mol) if mordred_calc.descriptors else {}
        mordred_vals = {str(desc): val for desc, val in mordred_vals.items()}
        return {'SMILES': smiles, **rdkit_vals, **mordred_vals}
    except Exception as e:
        print(f'Exception Triggered: {e}')
        return None

def fig_to_image(fig, fmt="png", dpi=300, return_pil=True):
    """
    Convert a matplotlib figure to PNG or SVG.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to export.
    fmt : str
        Output format: "png" or "svg".
    dpi : int
        Resolution used for raster export.
    return_pil : bool
        If True and format is PNG, returns a PIL Image.
        Otherwise returns raw bytes.

    Returns
    -------
    PIL.Image | bytes | str
        PNG -> PIL Image or bytes
        SVG -> SVG string
    """

    if fmt.lower() == "png":

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)

        if return_pil:
            img = Image.open(buf)
            if hasattr(img, "info") and isinstance(img.info, dict):
                img.info = {k: str(v) for k, v in img.info.items()}
            return img
        else:
            return buf.getvalue()

    elif fmt.lower() == "svg":
        buf = io.StringIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        svg_data = buf.getvalue()
        buf.close()
        return svg_data
    else:
        raise ValueError("Unsupported format. Use 'png' or 'svg'.")

def mol_to_image_with_font(mol, size=(400, 400), atom_font_size=18, 
                           bond_line_width=4, dots_per_angstrom=None, 
                           hlt_atm_indcs = None, addAtomIndices=False):
    """
    Return a PIL.Image of `mol` drawn with RDKit's MolDraw2DCairo while setting atom label font size.
    atom_font_size should be an integer (point size).
    """
    try:
        w, h = size
        drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
        draw_options = drawer.drawOptions()

        # Important settings
        drawer.SetFontSize(int(atom_font_size))
        draw_options.addAtomIndices = addAtomIndices
        draw_options.highlightRadius = 0.4
        # draw_options.atomLabelFontSize = int(atom_font_size)   
        draw_options.bondLineWidth = bond_line_width           
        draw_options.minFontSize = atom_font_size
        draw_options.maxFontSize = atom_font_size+5
        draw_options.scaleBondWidth = True
        draw_options.annotationFontScale = 1.0

        if dots_per_angstrom is not None:
            try:
                draw_options.dotsPerAngstrom = float(dots_per_angstrom)
            except Exception:
                pass

        ## Highlight indices
        if isinstance(hlt_atm_indcs, int):
            hlt_atm_indcs = [hlt_atm_indcs]
        highlight_atom_indices = list(hlt_atm_indcs) if hlt_atm_indcs is not None else []
        
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atom_indices)
        drawer.FinishDrawing()
        png = drawer.GetDrawingText()
        if isinstance(png, str):
            png = png.encode("utf-8")
        return Image.open(io.BytesIO(png))
    except Exception as exc:
        print("Warning: MolDraw2DCairo unavailable/failed (falling back). Error:", exc)
        return Draw.MolToImage(mol, size=size)
    

def plot_smiles_grid(smiles_list: List[str],
                     legends: List[str]=None,
                     titles: List[str]=None,
                     cols: int = 5,
                     image_size: Tuple[int,int] = (400, 400),
                     figsize: Tuple[int,int] = None,
                     savepath: str = '',
                     suptitle: str = '',
                     caption: str = '',
                     legendfontsize: int = 0,
                     row_lines: bool = True,
                     atom_font_size: int = 18,
                     dots_per_angstrom: float = None,
                     bond_line_width: float = 4,
                     separate_first_column: bool = False,
                     highlight_atom_indices = None,
                     addAtomIndices=False,
                     show=True,
                     fmt='svg'
                     ) -> None:
    """
    Create a grid plot of molecules from SMILES strings.

    Parameters:
        smiles_list (list of str): List of SMILES strings.
        legends (list of str, optional): Captions below each molecule.
        titles (list of str, optional): Titles above each molecule.
        cols (int): Number of columns in the grid.
        image_size (tuple): Size of each image (width, height).
        figsize (tuple): Overall size of the grid figure (width, height).
        suptitle (str): Title for the entire figure.
        caption (str): Caption for the entire figure.
    """
    if figsize is None:
        n_row = (len(smiles_list)//cols)+1
        figsize=(3.3*cols, 4 * n_row)
    if legendfontsize == 0:
        legendfontsize = figsize[0]-int(figsize[0]/cols)-1
        
    titlefontsize = legendfontsize + 2
    num_molecules = len(smiles_list)
    rows = math.ceil(num_molecules / cols)
    fig, axs = plt.subplots(rows, cols, figsize=figsize)

    # Normalize axs to a flat list
    axs = np.atleast_1d(axs).reshape(-1)

    if highlight_atom_indices is not None:
        if isinstance(highlight_atom_indices, list):
            assert (len(highlight_atom_indices) == len(smiles_list), 
                    f'length mismatch between highlight_atom_indices ({len(highlight_atom_indices)}) and smiles_list ({len(smiles_list)})'
                    )
        if isinstance(highlight_atom_indices, int):
            highlight_atom_indices = [[highlight_atom_indices]]*len(smiles_list)

    for i, ax in enumerate(axs):
        if i < num_molecules:
            smiles = smiles_list[i]
            if highlight_atom_indices is not None:
                hlt_atm_indcs = highlight_atom_indices[i]
            else:
                hlt_atm_indcs = None

            if isinstance(smiles, str):
                mol = Chem.MolFromSmiles(smiles)
            else:
                mol = smiles
            if mol:
                img = mol_to_image_with_font(mol,
                                             size=image_size,
                                             atom_font_size=atom_font_size,
                                             dots_per_angstrom=dots_per_angstrom,
                                             bond_line_width=bond_line_width,
                                             hlt_atm_indcs=hlt_atm_indcs,
                                             addAtomIndices=addAtomIndices
                                             )
                ax.imshow(img)
                # Titles and legends
                if titles and i < len(titles) and titles[i]:
                    ax.set_title(titles[i], fontsize=titlefontsize, fontweight="bold")
                if legends and i < len(legends) and legends[i]:
                    ax.text(0.5, -0.12, legends[i], fontsize=legendfontsize,
                            ha="center", va="center", transform=ax.transAxes, fontweight="bold")
            else:
                ax.text(0.5, 0.5, 'Invalid SMILES',
                        ha='center', va='center', fontsize=legendfontsize)
        ax.axis('off')

    # Hide unused axes
    for j in range(num_molecules, rows * cols):
        axs[j].axis('off')

    if suptitle:
        plt.suptitle(suptitle, fontsize=titlefontsize+2*int(figsize[0]/cols), fontweight='bold')

    if caption:
        plt.figtext(0.5, -0.02, caption, wrap=True, ha="center", fontsize=legendfontsize)

    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.tight_layout()
    left = min(ax.get_position().x0 for ax in axs)
    right = max(ax.get_position().x1 for ax in axs)

    if row_lines and rows > 1:
        for r in range(rows - 1):
            row_axes = axs[r * cols:(r + 1) * cols]
            y_bottom = min(ax.get_position().y0 for ax in row_axes)
            row_height = max(ax.get_position().height for ax in row_axes)
            y = y_bottom - 0.25 * row_height
            fig.add_artist(plt.Line2D([left, right], [y, y], transform=fig.transFigure, color="black", lw=1, alpha=0.5))
            
    if cols > 1 and separate_first_column:
        x = axs[0].get_position().x1
        y_bottom = axs[(rows - 1) * cols].get_position().y0
        y_top = axs[0].get_position().y1
        fig.add_artist(plt.Line2D([x, x], [y_bottom, y_top], transform=fig.transFigure, color="black", lw=1, alpha=0.5))

    if savepath:
        parent_folder = os.path.dirname(os.path.abspath(savepath))
        create_folder(parent_folder)
        print(savepath)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f'Figure saved in: {savepath}')
    else:
        if show:
            plt.show()
        else:
            plt.close()
            return fig_to_image(fig, dpi=600, fmt=fmt)