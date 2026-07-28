# Subsurface ML Platform

An end-to-end machine learning project for predicting subsurface lithology from well-log measurements.

## Goal

Build a reproducible multiclass classification system that predicts lithology for previously unseen wells.

## Key Constraint

Samples from the same well must not appear in both training and test sets.
The project will use group-based splitting with the well identifier as the grouping variable.