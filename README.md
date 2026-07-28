# Subsurface ML Platform

An end-to-end machine learning project for predicting subsurface lithology from well-log measurements.

## Goal

Build a reproducible multiclass classification system that predicts lithology for previously unseen wells.

## Key Constraint

Samples from the same well must not appear in both training and test sets.
The project will use group-based splitting with the well identifier as the grouping variable.

## Dataset

This project uses the FORCE 2020 well-log and lithofacies dataset.

The complete public dataset contains:

- 118 wells
- Approximately 2.34 million depth samples
- 12 lithology classes
- Well logs including GR, RHOB, NPHI, DTC, RDEP, RMED, SP, and others

Raw LAS files are not stored in this repository. Download the dataset separately and extract it under:

```text
data/raw/force2020_las/
