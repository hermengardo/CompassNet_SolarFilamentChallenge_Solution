# Dataset Documentation

## Important

This document describes the datasets used during the supervised and self-supervised training stages.

Due to their size, the image datasets could not be uploaded to GitHub. The `supervised/` directory contained the MAGFiLO dataset used for supervised training, while the `ssl/` directory contained the images collected for self-supervised training, organized by data source and observation date.

`images_table.csv` lists all images used during pretraining.

## Organization

- `supervised/`: directory containing the data used for supervised training.
- `ssl/`: directory containing the data collected for self-supervised training.
- `ssl/<source>/<year>/<month>/`: one image representation per observation, without segmentation masks.
- `images_table.csv`: list of images used for self-supervised training, including their associated metadata.
- `ssl_post_prep`: preprocessing script used to prepare the images before self-supervised training.

## Collection Summary

As of the October 27, 2026 audit, the unlabeled collection contained **6,501 images**: 5,000 GONG, 1,000 BBSO, and 501 KSO. The entire `not_related` directory occupied 7.41 GB on disk, including supervised data and model files.

Although images from GONG, BBSO, and KSO were collected, **only GONG images were used in the final self-supervised learning dataset**.

## Temporal Separation

The entire period from **2011-01-01 to 2022-12-31** was reserved exclusively for the existing supervised dataset, including its training, validation, and test sets. No image used for self-supervised training was allowed to fall within this interval, regardless of its observatory of origin.

| Source | Exclusive permitted intervals |
| --- | --- |
| BBSO Full Disk Hα | 2002-01-01 to 2004-12-31 |
| KSO | 2006-01-01 to 2009-12-31 |
| GONG | 2010-06-01 to 2010-09-30; 2023-04-01 to 2025-12-31 |

There were more than 90 days between any consecutive blocks, including the protected interval. These restrictions prevented temporal overlap between the collected datasets, but did not imply absolute statistical independence from recurring solar phenomena.

Within each source, observations were separated by at least four hours. For GONG, this constraint applied across the entire network rather than independently to each station.

## Solar Activity

The catalog stored monthly sunspot numbers obtained from **SILSO, Royal Observatory of Belgium**, using the version 2.0 monthly total sunspot number series. Only the monthly values required for the collection were stored in the `sunspots` table; a complete copy of the historical dataset was not maintained.

The `low` (<30), `moderate` (30–99.9), and `high` (≥100) activity groups were used exclusively as sampling descriptors. They were not filament labels, nor did they represent official dates marking solar cycle extrema.

The selected periods covered the 2008–2009 solar minimum and periods of high activity during solar cycles 23 and 25. The 2010 GONG observations further increased the diversity of solar activity represented in the collected dataset.

## Images and Provenance

- **[NSO/GONG Hα](https://nso.edu/data/nisp-data/h-alpha/):** `hag` products from stations B, C, L, M, T, and U.
- **[KSO](https://cesar.kso.ac.at/info/ftp.php):** synoptic Hα FITS images with flat-field correction (HA1 and HA2).
- **[BBSO Full Disk Hα](https://www.bbso.njit.edu/Research/FDHA/):** `bbso_halph` products.
- **[MLSO/PICS](https://www2.hao.ucar.edu/mlso/mlso-data-and-movies):** corrected solar disk JPEG images (`dpm.asl`). DOI: **10.5065/D65719TR**.

## Excluded Sources

MLSO was removed from the collection because its images did not meet the minimum resolution requirement of 512×512.
