# Table Detection Benchmark Report

Generated: 2025-08-07 10:43:40

## Summary

- Total detection runs: 1
- Models tested: gpt-4.1-nano
- Sheets evaluated: 10

## Model Performance

| Model        | Runs | Avg Precision | Avg Recall | Avg F1 | Avg IoU | Avg Tokens | Avg Cost |
| ------------ | ---- | ------------- | ---------- | ------ | ------- | ---------- | -------- |
| gpt-4.1-nano | 1    | 0.000         | 0.000      | 0.000  | 0.000   | 5961       | $0.0000  |

## Sheet-by-Sheet Analysis

### Sheet 0: Yiriden Transactions 2025

- Expected tables: 2
- Notes: Small dataset with transaction records and a summary totals row. May have two tables if there's a clear separation between data and totals.

| Model        | Tables Found | Precision | Recall | F1    | Avg IoU |
| ------------ | ------------ | --------- | ------ | ----- | ------- |
| gpt-4.1-nano | 2            | 0.000     | 0.000  | 0.000 | 0.000   |

### Sheet 1: Yiriden Transactions 2023

- Expected tables: 1
- Notes: Main data in columns 0-6, columns 7-10 have headers but minimal data (totals only)

*No detection runs for this sheet*

### Sheet 2: Yiriden 2023 Loans

- Expected tables: 2
- Notes: Classic side-by-side layout with empty column 7 as separator

*No detection runs for this sheet*

### Sheet 3: Sanoun Transactions 2024

- Expected tables: 2
- Notes: Has bottom summary section with totals

*No detection runs for this sheet*

### Sheet 4: Sanoun Transactions 2025

- Expected tables: 1
- Notes: Standard transaction table

*No detection runs for this sheet*

### Sheet 5: 2024 Shea butter shipping

- Expected tables: 2
- Notes: Two adjacent tables - shipping data and currency conversion

*No detection runs for this sheet*

### Sheet 6: Yiriden mileages

- Expected tables: 1
- Notes: Single mileage tracking table, may have summary rows at bottom

*No detection runs for this sheet*

### Sheet 7: Truck Revenue Projections

- Expected tables: 1
- Notes: Complex projection model, may have multiple sub-sections

*No detection runs for this sheet*

### Sheet 8: Yiriden 2022

- Expected tables: 1
- Notes: Large historical transaction dataset

*No detection runs for this sheet*

### Sheet 9: Real Estate - Horton Rd

- Expected tables: 1
- Notes: Need to verify actual structure and dimensions

*No detection runs for this sheet*

## Common Issues

- Over-detection (detected more tables than expected): 0 cases
- Under-detection (missed tables): 0 cases
- Missed side-by-side tables: 0 cases
