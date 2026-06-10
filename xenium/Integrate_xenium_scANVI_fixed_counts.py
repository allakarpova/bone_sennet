import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np
import re
import logging
import scvi

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

        # Prefix cell barcodes by sample_id so obs_names are unique across samples
        adata.obs_names = pd.Index([f"{sample}_{n}" for n in adata.obs_names])
        
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

    sc.pp.filter_cells(
        adata,
        min_counts=1
    )

    sc.pp.filter_genes(
        adata,
        min_counts=10
    )

    # Keep filtered raw counts for SCVI/scANVI. Pearson residuals are useful for
    # PCA/clustering, but feeding residualized values into SCVI can create NaNs.
    adata.layers["counts"] = adata.X.copy()
    logging.info('saved raw counts')
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

    sc.tl.leiden(
        adata,
        resolution=0.5,
        key_added="leiden"
    )

    # ----------------------------
    # filter out small clusters
    # ----------------------------

    # I also filtered small clusters to shrink scANVI classifier. Feel free to experiment with it as well
    _MIN_CLUSTER_SIZE = 50
    _vc = adata.obs['leiden'].value_counts()
    _small = _vc[_vc < _MIN_CLUSTER_SIZE].index
    _filt = adata.obs['leiden'].astype(str).copy()
    _filt[adata.obs['leiden'].isin(_small)] = 'Unknown'
    adata.obs['leiden_init'] = _filt.values
    _kept = (~adata.obs['leiden'].isin(_small)).sum()

    # ----------------------------
    # scANVI integration
    # ----------------------------

    adata_scvi = adata.copy()
    adata_scvi.X = adata_scvi.layers['counts'].copy()

    if hasattr(adata_scvi.X, "data"):
        if not np.isfinite(adata_scvi.X.data).all():
            raise ValueError("Raw count layer contains NaN or infinite values before SCVI setup.")
    elif not np.isfinite(adata_scvi.X).all():
        raise ValueError("Raw count layer contains NaN or infinite values before SCVI setup.")

    scvi.model.SCVI.setup_anndata(adata_scvi, batch_key='Sample_ID')
    vae = scvi.model.SCVI(adata_scvi, n_latent=30, n_layers=2)
    vae.train(max_epochs=10, batch_size=512, accelerator='gpu') # if you have GPU which should be 10-15x faster, you need to specify it: vae.train(max_epochs=10, batch_size=512, accelerator='gpu')

    adata_scvi.obs['leiden_init'] = adata.obs['leiden_init'].values
    scanvi_model = scvi.model.SCANVI.from_scvi_model(vae, labels_key='leiden_init', unlabeled_category='Unknown')
    scanvi_model.train(max_epochs=10, batch_size=512, accelerator='gpu')

    adata.obsm['X_scANVI'] = scanvi_model.get_latent_representation()

    sc.pp.neighbors(adata, use_rep='X_scANVI', n_neighbors=15, metric='euclidean')
    sc.tl.umap(adata, min_dist=0.05)
    sc.tl.leiden(adata, resolution=1.0, key_added='leiden_scanvi')

    adata.obsm['X_umap_scanvi'] = adata.obsm['X_umap'].copy()
    logging.info('scANVI done')
    # ----------------------------
    # Save
    # ----------------------------

    adata.write(
        output_dir / "scANVI_integrated_xenium_pearson.h5ad"
    )

    sc.settings.figdir = str(output_dir)

    sc.pl.umap(
        adata,
        color="leiden_scanvi",
        save="_leiden_scanvi.png"
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
