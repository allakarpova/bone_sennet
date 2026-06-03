import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np


def main(tsv_file, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------
    # Read sample sheet
    # ----------------------------

    samples = pd.read_csv(
        tsv_file,
        sep="\t"
    )

    # ----------------------------
    # Load + normalize each sample
    # ----------------------------

    adata_list = []

    for _, row in samples.iterrows():

        sample = row["Sample_ID"]
        xenium_dir = Path(row["Xenium"])

        adata = sc.read_10x_h5(
            xenium_dir / "cell_feature_matrix.h5"
        )

        adata.var_names_make_unique()

        adata.obs["Sample_ID"] = sample

        sc.pp.filter_cells(
            adata,
            min_counts=1
        )

        sc.pp.filter_genes(
            adata,
            min_counts=10
        )

        adata_list.append(adata)

    # ----------------------------
    # Merge
    # ----------------------------

    adata = ad.concat(
        adata_list,
        join="inner",
        index_unique="-"
    )

    # ----------------------------
    # Normalize with pearson residuals aka SCTransform
    # ----------------------------

    sc.experimental.pp.normalize_pearson_residuals(
            adata,
            theta=100
        )

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
        n_neighbors=15,
        n_pcs=30
    )

    sc.tl.umap(
        adata
    )

    sc.tl.leiden(
        adata,
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
        help="Input TSV with Sample_ID and Xenium columns"
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