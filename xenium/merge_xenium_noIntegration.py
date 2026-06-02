import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np

# ----------------------------
# Read sample sheet
# ----------------------------

samples = pd.read_csv(
    "sample_sheet.tsv",
    sep="\t"
)

# ----------------------------
# Load + normalize each sample
# ----------------------------

adata_list = []

for _, row in samples.iterrows():

    sample = row["Sample_ID"]
    xenium_path = row["Xenium"]

    adata = sc.read_10x_h5(xenium_path)

    adata.var_names_make_unique()

    adata.obs["Sample_ID"] = sample

    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.filter_genes(adata, min_counts=1)

    sc.experimental.pp.normalize_pearson_residuals(
        adata,
        theta=100
    )

    adata_list.append(adata)

# ----------------------------
# Merge
# ----------------------------

adata = ad.concat(
    adata_list,
    join="inner",
    label="Sample_ID",
    keys=samples["Sample_ID"].tolist(),
    index_unique="-"
)

# ----------------------------
# PCA / clustering / UMAP
# ----------------------------

sc.pp.pca(
    adata,
    n_comps=30
)

sc.pp.neighbors(
    adata,
    n_neighbors=15,
    n_pcs=30
)

sc.tl.umap(adata)

sc.tl.leiden(
    adata,
    resolution=0.5,
    key_added="leiden"
)

# ----------------------------
# Save merged object
# ----------------------------

adata.write(
    "merged_xenium_pearson.h5ad"
)

# ----------------------------
# Plot
# ----------------------------

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