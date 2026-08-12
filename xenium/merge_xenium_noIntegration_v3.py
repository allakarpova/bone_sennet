import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np
from matplotlib.path import Path as MplPath


def read_sample_sheet(sample_file):
    """Read a sample sheet that may be comma-separated or tab-separated.

    Required columns are Sample_ID and Xenium. The optional Sample_boundaries
    column should contain a polygon coordinate CSV path for samples that need
    cropping before merge.
    """
    samples = pd.read_csv(
        sample_file,
        sep=None,
        engine="python"
    )

    required = {"Sample_ID", "Xenium"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"Sample sheet is missing required columns: {sorted(missing)}")

    if "Sample_boundaries" not in samples.columns:
        samples["Sample_boundaries"] = ""

    samples["Sample_boundaries"] = samples["Sample_boundaries"].fillna("").astype(str).str.strip()
    return samples


def read_boundary_polygon(boundary_file):
    """Read QuPath/Xenium-style polygon coordinates from a CSV file.

    The expected coordinate columns are X and Y. Lines beginning with # are
    ignored, which handles files exported with selection metadata headers.
    """
    boundary_file = Path(boundary_file)
    if not boundary_file.exists():
        raise FileNotFoundError(f"Boundary coordinate file does not exist: {boundary_file}")

    polygon = pd.read_csv(
        boundary_file,
        comment="#"
    )
    polygon.columns = [c.strip() for c in polygon.columns]

    lower_to_real = {c.lower(): c for c in polygon.columns}
    if "x" not in lower_to_real or "y" not in lower_to_real:
        raise ValueError(
            f"Boundary file {boundary_file} must contain X and Y columns. "
            f"Found columns: {polygon.columns.tolist()}"
        )

    xy = polygon[[lower_to_real["x"], lower_to_real["y"]]].dropna().to_numpy(float)
    if xy.shape[0] < 3:
        raise ValueError(f"Boundary file {boundary_file} has fewer than 3 polygon vertices")

    return xy


def read_cell_centroids(xenium_dir):
    """Read Xenium cell centroid coordinates used for boundary-based subsetting."""
    cells_csv = xenium_dir / "cells.csv.gz"
    cells_parquet = xenium_dir / "cells.parquet"

    if cells_csv.exists():
        cells = pd.read_csv(
            cells_csv,
            usecols=["cell_id", "x_centroid", "y_centroid"]
        )
    elif cells_parquet.exists():
        cells = pd.read_parquet(
            cells_parquet,
            columns=["cell_id", "x_centroid", "y_centroid"]
        )
    else:
        raise FileNotFoundError(
            f"Could not find cells.csv.gz or cells.parquet in Xenium output: {xenium_dir}"
        )

    return cells


def subset_to_boundary(adata, xenium_dir, boundary_file):
    """Keep only cells whose Xenium centroids fall inside a sample boundary polygon."""
    polygon_xy = read_boundary_polygon(boundary_file)
    cells = read_cell_centroids(xenium_dir)

    # Match the raw 10x/Xenium cell barcodes before Sample_ID prefixes are added.
    coord = cells.set_index("cell_id").reindex(adata.obs_names)
    missing_coord = coord["x_centroid"].isna().sum()
    if missing_coord:
        print(f"Warning: {missing_coord} cells lacked centroid coordinates in {xenium_dir}")

    valid = coord[["x_centroid", "y_centroid"]].notna().all(axis=1).to_numpy()
    points = coord[["x_centroid", "y_centroid"]].to_numpy(float)
    inside = np.zeros(adata.n_obs, dtype=bool)
    inside[valid] = MplPath(polygon_xy).contains_points(points[valid])

    return adata[inside].copy(), int(inside.sum()), int(adata.n_obs)


def main(tsv_file, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------
    # Read sample sheet
    # ----------------------------

    samples = read_sample_sheet(tsv_file)

    # ----------------------------
    # Load each sample
    # ----------------------------

    adata_list = []
    merge_summary = []

    for _, row in samples.iterrows():

        sample = row["Sample_ID"]
        xenium_dir = Path(row["Xenium"])
        boundary_file = row["Sample_boundaries"]

        adata = sc.read_10x_h5(
            xenium_dir / "cell_feature_matrix.h5"
        )

        adata.var_names_make_unique()

        n_cells_loaded = adata.n_obs
        n_cells_after_boundary = adata.n_obs
        boundary_applied = bool(boundary_file)

        if boundary_applied:
            print(f"{sample}: subsetting by boundary {boundary_file}")
            adata, n_cells_after_boundary, n_cells_loaded = subset_to_boundary(
                adata,
                xenium_dir,
                boundary_file
            )
            if adata.n_obs == 0:
                raise ValueError(
                    f"Boundary {boundary_file} selected 0 cells for sample {sample}. "
                    "Check that coordinate units and the Xenium output path match."
                )
        else:
            print(f"{sample}: no boundary provided; using full Xenium output")

        adata.obs["Sample_ID"] = sample
        adata.obs["Xenium"] = str(xenium_dir)
        adata.obs["Sample_boundaries"] = boundary_file

        # Prefix cell barcodes by sample_id so obs_names are unique across samples
        adata.obs_names = pd.Index([f"{sample}_{n}" for n in adata.obs_names])

        sc.pp.filter_cells(
            adata,
            min_counts=10
        )

        sc.pp.filter_genes(
            adata,
            min_counts=10
        )

        merge_summary.append(
            {
                "Sample_ID": sample,
                "Xenium": str(xenium_dir),
                "Sample_boundaries": boundary_file,
                "boundary_applied": boundary_applied,
                "n_cells_loaded": n_cells_loaded,
                "n_cells_after_boundary": n_cells_after_boundary,
                "n_cells_after_filter": adata.n_obs,
            }
        )

        adata_list.append(adata)

    # ----------------------------
    # Merge
    # ----------------------------

    adata = ad.concat(
        adata_list,
        join="inner"
    )

    pd.DataFrame(merge_summary).to_csv(
        output_dir / "merge_boundary_summary.tsv",
        sep="\t",
        index=False
    )

    sc.pp.filter_cells(
        adata,
        min_counts=10
    )

    sc.pp.filter_genes(
        adata,
        min_counts=10
    )

    # Keep filtered raw counts for SCVI/scANVI later on.
    adata.layers["counts"] = adata.X.copy()

    # ----------------------------
    # Normalize with pearson residuals aka SCTransform
    # ----------------------------
    print("NaNs before pearson:", np.isnan(adata.X.data).sum())
    
    sc.experimental.pp.normalize_pearson_residuals(
            adata,
            theta=100
        )
    print("NaNs after pearson:", np.isnan(adata.X.data).sum())
    # ----------------------------
    # HVG by residual variance (top 2000)
    # ----------------------------

    X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    gene_var = np.var(X, axis=0)
    top_idx = np.argsort(gene_var)[::-1][:2000]
    hvg_mask = np.zeros(adata.shape[1], dtype=bool)
    hvg_mask[top_idx] = True
    adata.var['highly_variable'] = hvg_mask
    print(f'HVGs: {hvg_mask.sum()} / {adata.shape[1]}')

    # ----------------------------
    # PCA / clustering / UMAP
    # ----------------------------

    sc.tl.pca(
        adata,
        n_comps=30,
        mask_var='highly_variable'
    )

    sc.pp.neighbors(
        adata,
        n_neighbors=30,
        n_pcs=30
    )

    sc.tl.umap(
        adata
    )

    sc.tl.leiden(
        adata,
        n_iterations=2,
        flavor="igraph",
        resolution=0.5,
        key_added="leiden"
    )

    # ----------------------------
    # Save
    # ----------------------------

    adata.write(
        output_dir / "merged_xenium_pearson_umap.h5ad"
    )

    sc.settings.figdir = str(output_dir)

    sc.pl.umap(
        adata,
        color="leiden",
        save="_clusters.png"
    )

    sc.pl.umap(
        adata,
        color="Sample_ID",
        save="_sampleID.png"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help=(
            "Input sample sheet with Sample_ID and Xenium columns. "
            "CSV and TSV are both accepted. Optional Sample_boundaries values "
            "crop Xenium outputs to polygon coordinate CSV files before merge."
        )
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output directory"
    )

    args = parser.parse_args()

    main(
        args.input,
        args.output
    )
