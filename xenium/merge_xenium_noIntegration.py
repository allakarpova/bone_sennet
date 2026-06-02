import argparse
import pandas as pd
import scanpy as sc
import anndata as ad


def main(tsv_file):

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
        xenium_path = row["Xenium"]

        adata = sc.read_10x_h5(
            xenium_path
        )

        adata.var_names_make_unique()

        adata.obs["Sample_ID"] = sample

        sc.pp.filter_cells(
            adata,
            min_counts=1
        )

        sc.pp.filter_genes(
            adata,
            min_counts=1
        )

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

    sc.tl.umap(
        adata
    )

    sc.tl.leiden(
        adata,
        resolution=0.5,
        key_added="cluster"
    )

    # ----------------------------
    # Save outputs
    # ----------------------------

    adata.write(
        "merged_xenium_pearson_umap.h5ad"
    )

    sc.pl.umap(
        adata,
        color="cluster",
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
        "tsv",
        help="TSV with Sample_ID and Xenium columns"
    )

    args = parser.parse_args()

    main(
        args.tsv
    )