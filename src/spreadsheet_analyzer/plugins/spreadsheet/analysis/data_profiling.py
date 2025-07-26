"""Advanced data profiling utilities for spreadsheet analysis.

This module provides integration with YData-Profiling (formerly pandas-profiling)
for comprehensive data quality assessment and automated insights generation.
"""

from typing import TypedDict

from structlog import get_logger

logger = get_logger(__name__)


class ProfilingConfig(TypedDict, total=False):
    """Configuration for profiling operations."""

    minimal: bool  # Minimal mode for large datasets
    samples_per_column: int  # Number of samples for correlation
    correlations: dict[str, bool]  # Which correlations to compute
    missing_diagrams: dict[str, bool]  # Which missing value diagrams to show
    explorative: bool  # Enable explorative analysis
    sensitive: bool  # Mark if data contains sensitive info


def generate_profiling_code(sheet_name: str, config: ProfilingConfig | None = None) -> str:
    """Generate Python code for YData-Profiling analysis.

    Args:
        sheet_name: Name of the sheet being analyzed
        config: Optional profiling configuration

    Returns:
        Python code string for profiling analysis
    """
    # Default config
    if config is None:
        config = ProfilingConfig(minimal=False, explorative=True, sensitive=False)

    minimal = config.get("minimal", False)
    explorative = config.get("explorative", True)

    code = f"""# Advanced Data Profiling using YData-Profiling
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Check if ydata-profiling is available
try:
    from ydata_profiling import ProfileReport
    PROFILING_AVAILABLE = True
except ImportError:
    try:
        # Try legacy import
        from pandas_profiling import ProfileReport
        PROFILING_AVAILABLE = True
    except ImportError:
        PROFILING_AVAILABLE = False
        print("⚠️ YData-Profiling not available. Install with: pip install ydata-profiling")
        print("Falling back to basic profiling...")

if PROFILING_AVAILABLE and 'df' in globals():
    print("\\n=== ADVANCED DATA PROFILING ===")
    print(f"Analyzing sheet: {sheet_name}")
    print(f"Data shape: {{df.shape}}")

    # Configure profiling based on data size
    row_count = len(df)
    col_count = len(df.columns)

    # Auto-detect if minimal mode needed
    use_minimal = {str(minimal).lower()}
    if row_count > 10000 or col_count > 50:
        use_minimal = True
        print(f"Large dataset detected. Using minimal mode for performance.")

    # Create profile configuration
    profile_config = {{
        "samples": {{
            "head": 10,
            "tail": 10
        }},
        "correlations": {{
            "auto": True,
            "pearson": not use_minimal,
            "spearman": not use_minimal,
            "kendall": False,  # Expensive
            "phi_k": False,  # Very expensive
            "cramers": not use_minimal,
        }},
        "missing_diagrams": {{
            "matrix": not use_minimal,
            "bar": True,
            "heatmap": not use_minimal,
        }},
        "interactions": {{
            "continuous": not use_minimal
        }},
        "explorative": {str(explorative).lower()},
    }}

    try:
        # Generate profile
        profile = ProfileReport(
            df,
            title=f"Data Profile: {{sheet_name}}",
            minimal=use_minimal,
            config_file=None,
            **profile_config
        )

        # Extract key insights
        print("\\n📊 KEY INSIGHTS FROM PROFILING:")

        # Variable types
        var_types = profile.get_description()["table"]["types"]
        print(f"\\nVariable Types:")
        for vtype, count in var_types.items():
            if count > 0:
                print(f"  - {{vtype}}: {{count}} columns")

        # Missing values summary
        missing_count = profile.get_description()["table"]["n_missing"]
        if missing_count > 0:
            missing_pct = (missing_count / (row_count * col_count)) * 100
            print(f"\\nMissing Values: {{missing_count:,}} ({{missing_pct:.1f}}% of all cells)")

        # Duplicate rows
        duplicates = profile.get_description()["table"]["n_duplicates"]
        if duplicates > 0:
            dup_pct = (duplicates / row_count) * 100
            print(f"\\nDuplicate Rows: {{duplicates:,}} ({{dup_pct:.1f}}%)")

        # High correlation warnings
        if not use_minimal:
            alerts = profile.get_description().get("alerts", [])
            high_corr = [a for a in alerts if a.get("type") == "HIGH_CORRELATION"]
            if high_corr:
                print(f"\\n⚠️ High Correlations Detected: {{len(high_corr)}} pairs")
                for alert in high_corr[:5]:  # Show top 5
                    vars = alert.get("variables", [])
                    if len(vars) >= 2:
                        print(f"  - {{vars[0]}} ↔ {{vars[1]}}")

        # Constant/unique columns
        const_cols = [col for col, desc in profile.get_description()["variables"].items()
                      if desc.get("n_distinct") == 1]
        if const_cols:
            print(f"\\nConstant Columns (single value): {{', '.join(const_cols[:5])}}")
            if len(const_cols) > 5:
                print(f"  ... and {{len(const_cols) - 5}} more")

        unique_cols = [col for col, desc in profile.get_description()["variables"].items()
                       if desc.get("n_distinct") == desc.get("n")]
        if unique_cols:
            print(f"\\nUnique Columns (all different values): {{', '.join(unique_cols[:5])}}")
            if len(unique_cols) > 5:
                print(f"  ... and {{len(unique_cols) - 5}} more")

        # Save report if not too large
        if row_count < 50000:
            output_dir = Path("analysis_results") / Path(excel_file).stem
            output_dir.mkdir(parents=True, exist_ok=True)
            report_path = output_dir / f"{{sheet_name.replace(' ', '_').lower()}}_profile.html"
            profile.to_file(report_path)
            print(f"\\n✅ Full profiling report saved to: {{report_path}}")
        else:
            print("\\n📝 Dataset too large for HTML report generation")

    except Exception as e:
        print(f"\\n❌ Profiling failed: {{type(e).__name__}}: {{e}}")
        print("Falling back to basic profiling...")
        PROFILING_AVAILABLE = False

# Fallback: Basic profiling if YData not available
if not PROFILING_AVAILABLE and 'df' in globals():
    print("\\n=== BASIC DATA PROFILING ===")

    # Data types summary
    dtype_counts = df.dtypes.value_counts()
    print("\\nData Types Summary:")
    for dtype, count in dtype_counts.items():
        print(f"  - {{dtype}}: {{count}} columns")

    # Missing values analysis
    missing_df = pd.DataFrame({{
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percent': (df.isnull().sum() / len(df)) * 100
    }})
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

    if len(missing_df) > 0:
        print(f"\\nColumns with Missing Values ({{len(missing_df)}} / {{len(df.columns)}}):")
        print(missing_df.head(10).to_string(index=False))
    else:
        print("\\n✅ No missing values found!")

    # Basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\\nNumeric Columns Summary ({{len(numeric_cols)}} columns):")
        desc_stats = df[numeric_cols].describe()
        print(desc_stats)

    # Cardinality analysis for object columns
    object_cols = df.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        print(f"\\nCategorical Columns Cardinality ({{len(object_cols)}} columns):")
        cardinality = pd.DataFrame({{
            'Column': object_cols,
            'Unique_Values': [df[col].nunique() for col in object_cols],
            'Unique_Ratio': [df[col].nunique() / len(df) for col in object_cols]
        }}).sort_values('Unique_Values', ascending=False)
        print(cardinality.head(10).to_string(index=False))

    # Memory usage
    memory_usage = df.memory_usage(deep=True)
    total_memory_mb = memory_usage.sum() / (1024 * 1024)
    print(f"\\nMemory Usage: {{total_memory_mb:.2f}} MB")

    # Correlation matrix for numeric columns (if not too many)
    if len(numeric_cols) > 1 and len(numeric_cols) <= 20:
        print(f"\\nCorrelation Matrix (top correlations):")
        corr_matrix = df[numeric_cols].corr()

        # Find high correlations
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))

        if high_corr:
            high_corr.sort(key=lambda x: abs(x[2]), reverse=True)
            print("\\nHigh Correlations (|r| > 0.7):")
            for col1, col2, corr in high_corr[:10]:
                print(f"  {{col1}} ↔ {{col2}}: {{corr:.3f}}")
"""

    return code


def generate_outlier_detection_code() -> str:
    """Generate code for comprehensive outlier detection."""

    return """# Outlier Detection and Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print("\\n=== OUTLIER DETECTION ===")

if 'df' in globals():
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) > 0:
        outlier_summary = []

        # Configure plot
        n_cols = min(len(numeric_cols), 6)  # Max 6 plots
        if n_cols > 0:
            fig, axes = plt.subplots(2, min(3, n_cols), figsize=(15, 10))
            fig.suptitle('Outlier Analysis', fontsize=16)
            axes = axes.flatten() if n_cols > 1 else [axes]

        for idx, col in enumerate(numeric_cols[:6]):
            data = df[col].dropna()
            if len(data) < 4:  # Need at least 4 points
                continue

            # Method 1: IQR Method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]

            # Method 2: Z-Score Method (for normally distributed data)
            z_scores = np.abs(stats.zscore(data))
            z_outliers = data[z_scores > 3]

            # Method 3: Modified Z-Score (more robust)
            median = data.median()
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad if mad != 0 else np.zeros_like(data)
            mz_outliers = data[np.abs(modified_z_scores) > 3.5]

            # Store results
            outlier_summary.append({
                'Column': col,
                'Total_Values': len(data),
                'IQR_Outliers': len(iqr_outliers),
                'IQR_Percent': (len(iqr_outliers) / len(data)) * 100,
                'Z_Score_Outliers': len(z_outliers),
                'Modified_Z_Outliers': len(mz_outliers),
                'Min': data.min(),
                'Q1': Q1,
                'Median': median,
                'Q3': Q3,
                'Max': data.max(),
                'Mean': data.mean(),
                'Std': data.std()
            })

            # Plot box plot
            if idx < len(axes):
                ax = axes[idx]
                ax.boxplot(data, vert=True)
                ax.set_title(f'{col}\\n{len(iqr_outliers)} outliers ({(len(iqr_outliers)/len(data))*100:.1f}%)')
                ax.set_ylabel('Value')

                # Mark outliers
                outlier_y = iqr_outliers.values
                outlier_x = np.ones_like(outlier_y)
                ax.scatter(outlier_x, outlier_y, color='red', marker='o', s=50, alpha=0.5)

        # Clean up unused subplots
        for idx in range(len(numeric_cols[:6]), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show()

        # Summary table
        outlier_df = pd.DataFrame(outlier_summary)
        print("\\nOutlier Summary by Column:")
        print(outlier_df[['Column', 'Total_Values', 'IQR_Outliers', 'IQR_Percent']].to_string(index=False))

        # Identify rows with multiple outliers
        print("\\n🎯 ROWS WITH MULTIPLE OUTLIERS:")

        # Create outlier mask for each column
        outlier_masks = {}
        for col in numeric_cols:
            data = df[col]
            if data.notna().sum() > 3:  # Need enough data
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                outlier_masks[col] = (data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)

        # Count outliers per row
        if outlier_masks:
            outlier_counts = pd.DataFrame(outlier_masks).sum(axis=1)
            multi_outlier_rows = outlier_counts[outlier_counts >= 2]

            if len(multi_outlier_rows) > 0:
                print(f"Found {len(multi_outlier_rows)} rows with 2+ outliers")

                # Show top rows with most outliers
                top_outlier_indices = multi_outlier_rows.nlargest(5).index
                for idx in top_outlier_indices:
                    outlier_cols = [col for col, mask in outlier_masks.items() if mask.iloc[idx]]
                    print(f"\\nRow {idx}: {len(outlier_cols)} outliers in columns: {', '.join(outlier_cols)}")

                    # Show the actual values
                    row_data = df.loc[idx, outlier_cols]
                    for col in outlier_cols:
                        value = df.loc[idx, col]
                        col_data = df[col].dropna()
                        percentile = stats.percentileofscore(col_data, value)
                        print(f"  {col}: {value:.2f} (percentile: {percentile:.1f}%)")
            else:
                print("No rows found with multiple outliers")

        # Context-aware outlier analysis
        print("\\n💡 CONTEXTUAL OUTLIER INSIGHTS:")

        # Check for potential data entry errors (e.g., extra zeros)
        for col in numeric_cols[:10]:  # Check first 10 numeric columns
            data = df[col].dropna()
            if len(data) > 10:
                # Check for values that are 10x or 100x larger than median
                median = data.median()
                if median != 0:
                    large_multiples = data[data > median * 10]
                    if len(large_multiples) > 0:
                        # Check if they're round multiples
                        for val in large_multiples.head(3):
                            if val % (median * 10) < median or val % (median * 100) < median:
                                print(f"\\n⚠️ Possible data entry error in '{col}':")
                                print(f"   Value {val} might be {median} with extra zeros")

    else:
        print("No numeric columns found for outlier analysis")
"""


def generate_semantic_validation_code() -> str:
    """Generate code for semantic validation of spreadsheet data."""

    return """# Semantic Data Validation
import pandas as pd
import numpy as np
import re
from datetime import datetime

print("\\n=== SEMANTIC VALIDATION ===")

validation_issues = []

if 'df' in globals():
    # Email validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    for col in df.columns:
        if 'email' in col.lower() or 'mail' in col.lower():
            if df[col].dtype == 'object':
                invalid_emails = df[col].dropna()[~df[col].dropna().str.match(email_pattern)]
                if len(invalid_emails) > 0:
                    validation_issues.append({
                        'Column': col,
                        'Type': 'Invalid Email',
                        'Count': len(invalid_emails),
                        'Examples': list(invalid_emails.head(3))
                    })

    # Phone number validation (basic)
    phone_pattern = r'^[\\+]?[(]?[0-9]{3}[)]?[-\\s\\.]?[(]?[0-9]{3}[)]?[-\\s\\.]?[0-9]{4,6}$'
    for col in df.columns:
        if 'phone' in col.lower() or 'tel' in col.lower() or 'mobile' in col.lower():
            if df[col].dtype == 'object':
                # Clean and check
                cleaned = df[col].dropna().astype(str).str.replace(r'[^0-9+]', '', regex=True)
                invalid_phones = cleaned[~cleaned.str.match(r'^[+]?[0-9]{10,15}$')]
                if len(invalid_phones) > 0:
                    validation_issues.append({
                        'Column': col,
                        'Type': 'Invalid Phone',
                        'Count': len(invalid_phones),
                        'Examples': list(df[col][invalid_phones.index].head(3))
                    })

    # Date validation and consistency
    date_columns = []
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            date_columns.append(col)
        elif df[col].dtype == 'object' and df[col].notna().any():
            # Try to parse as date
            sample = df[col].dropna().head(5)
            try:
                pd.to_datetime(sample, errors='coerce')
                if pd.to_datetime(sample, errors='coerce').notna().all():
                    date_columns.append(col)
            except:
                pass

    if date_columns:
        print(f"\\nDate columns found: {', '.join(date_columns)}")
        for col in date_columns:
            try:
                dates = pd.to_datetime(df[col], errors='coerce')
                invalid_dates = df[col][dates.isna() & df[col].notna()]
                if len(invalid_dates) > 0:
                    validation_issues.append({
                        'Column': col,
                        'Type': 'Invalid Date Format',
                        'Count': len(invalid_dates),
                        'Examples': list(invalid_dates.head(3))
                    })

                # Check for future dates if suspicious
                valid_dates = dates.dropna()
                if len(valid_dates) > 0:
                    future_dates = valid_dates[valid_dates > datetime.now()]
                    if len(future_dates) > 0 and 'future' not in col.lower():
                        validation_issues.append({
                            'Column': col,
                            'Type': 'Future Date',
                            'Count': len(future_dates),
                            'Examples': list(future_dates.head(3).astype(str))
                        })
            except:
                pass

    # Numeric range validation
    for col in df.select_dtypes(include=[np.number]).columns:
        # Percentage columns should be 0-100
        if 'percent' in col.lower() or 'pct' in col.lower() or '%' in col:
            out_of_range = df[col][(df[col] < 0) | (df[col] > 100)]
            if len(out_of_range) > 0:
                validation_issues.append({
                    'Column': col,
                    'Type': 'Percentage Out of Range',
                    'Count': len(out_of_range),
                    'Examples': list(out_of_range.head(3))
                })

        # Age columns should be reasonable
        if 'age' in col.lower():
            unreasonable = df[col][(df[col] < 0) | (df[col] > 150)]
            if len(unreasonable) > 0:
                validation_issues.append({
                    'Column': col,
                    'Type': 'Unreasonable Age',
                    'Count': len(unreasonable),
                    'Examples': list(unreasonable.head(3))
                })

        # Price/amount columns shouldn't be negative (usually)
        if any(keyword in col.lower() for keyword in ['price', 'amount', 'cost', 'revenue', 'sales']):
            negative_values = df[col][df[col] < 0]
            if len(negative_values) > 0 and 'refund' not in col.lower() and 'discount' not in col.lower():
                validation_issues.append({
                    'Column': col,
                    'Type': 'Negative Amount',
                    'Count': len(negative_values),
                    'Examples': list(negative_values.head(3))
                })

    # Cross-column validation
    print("\\n🔗 CROSS-COLUMN VALIDATION:")

    # Start date should be before end date
    date_pairs = []
    for i, col1 in enumerate(date_columns):
        for col2 in date_columns[i+1:]:
            if ('start' in col1.lower() and 'end' in col2.lower()) or \\
               ('begin' in col1.lower() and 'end' in col2.lower()) or \\
               ('from' in col1.lower() and 'to' in col2.lower()):
                date_pairs.append((col1, col2))

    for start_col, end_col in date_pairs:
        try:
            start_dates = pd.to_datetime(df[start_col], errors='coerce')
            end_dates = pd.to_datetime(df[end_col], errors='coerce')

            invalid_ranges = df[(start_dates > end_dates) & start_dates.notna() & end_dates.notna()]
            if len(invalid_ranges) > 0:
                validation_issues.append({
                    'Column': f'{start_col} > {end_col}',
                    'Type': 'Invalid Date Range',
                    'Count': len(invalid_ranges),
                    'Examples': []
                })
                print(f"⚠️ Found {len(invalid_ranges)} rows where {start_col} > {end_col}")
        except:
            pass

    # Summary of validation issues
    if validation_issues:
        print(f"\\n❌ VALIDATION ISSUES FOUND: {len(validation_issues)}")
        issue_df = pd.DataFrame(validation_issues)

        # Group by type
        by_type = issue_df.groupby('Type')['Count'].sum().sort_values(ascending=False)
        print("\\nIssues by Type:")
        for issue_type, count in by_type.items():
            print(f"  {issue_type}: {count} records")

        print("\\nDetailed Issues:")
        for issue in validation_issues[:10]:  # Show first 10
            print(f"\\n{issue['Type']} in '{issue['Column']}': {issue['Count']} records")
            if issue['Examples']:
                print(f"  Examples: {issue['Examples']}")
    else:
        print("\\n✅ No semantic validation issues found!")

    # Business rule validation suggestions
    print("\\n💼 SUGGESTED BUSINESS RULES TO VALIDATE:")

    # Total/sum columns
    total_cols = [col for col in df.columns if 'total' in col.lower() or 'sum' in col.lower()]
    if total_cols:
        print(f"\\n1. Total/Sum columns found: {', '.join(total_cols[:3])}")
        print("   → Consider validating these equal the sum of their components")

    # ID columns
    id_cols = [col for col in df.columns if 'id' in col.lower() and df[col].dtype in ['object', 'int64']]
    if id_cols:
        for col in id_cols[:3]:
            duplicates = df[col].duplicated().sum()
            if duplicates > 0:
                print(f"\\n2. Duplicate IDs in '{col}': {duplicates} duplicates found")
                print("   → IDs should typically be unique")

    # Status/category columns
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        unique_vals = df[col].nunique()
        if 2 <= unique_vals <= 10:  # Likely status/category
            print(f"\\n3. Possible status/category column '{col}' with {unique_vals} values:")
            print(f"   Values: {df[col].value_counts().index.tolist()}")
            print("   → Define allowed values and check for typos/inconsistencies")
"""


async def check_profiling_installed() -> bool:
    """Check if YData-Profiling is installed."""
    try:
        import ydata_profiling

        return True
    except ImportError:
        try:
            import pandas_profiling  # Legacy name

            return True
        except ImportError:
            return False


def generate_comprehensive_quality_code(
    sheet_name: str, include_profiling: bool = True, include_outliers: bool = True, include_validation: bool = True
) -> str:
    """Generate comprehensive quality analysis code.

    This combines profiling, outlier detection, and semantic validation
    into a single analysis flow.
    """
    sections = []

    if include_profiling:
        sections.append(generate_profiling_code(sheet_name))

    if include_outliers:
        sections.append(generate_outlier_detection_code())

    if include_validation:
        sections.append(generate_semantic_validation_code())

    # Add summary section
    sections.append("""
# === QUALITY ANALYSIS SUMMARY ===
print("\\n" + "="*50)
print("SPREADSHEET QUALITY REPORT CARD")
print("="*50)

quality_score = 100  # Start with perfect score

# Data completeness
if 'df' in globals():
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    if missing_pct > 20:
        quality_score -= 20
        print(f"\\n❌ Data Completeness: POOR ({100-missing_pct:.1f}% complete)")
    elif missing_pct > 10:
        quality_score -= 10
        print(f"\\n⚠️  Data Completeness: FAIR ({100-missing_pct:.1f}% complete)")
    else:
        print(f"\\n✅ Data Completeness: GOOD ({100-missing_pct:.1f}% complete)")

# Formula consistency (if checked)
if 'all_inconsistencies' in globals() and all_inconsistencies:
    quality_score -= min(20, len(all_inconsistencies))
    print(f"\\n❌ Formula Consistency: {len(all_inconsistencies)} issues found")
else:
    print("\\n✅ Formula Consistency: No issues detected")

# Data validation issues
if 'validation_issues' in globals() and validation_issues:
    total_issues = sum(issue['Count'] for issue in validation_issues)
    quality_score -= min(20, total_issues // 10)
    print(f"\\n⚠️  Data Validation: {total_issues} issues across {len(validation_issues)} checks")

# Outliers
if 'outlier_df' in globals():
    high_outlier_cols = outlier_df[outlier_df['IQR_Percent'] > 5]
    if len(high_outlier_cols) > 0:
        quality_score -= min(10, len(high_outlier_cols) * 2)
        print(f"\\n⚠️  Outlier Analysis: {len(high_outlier_cols)} columns with >5% outliers")

print(f"\\n\\n📊 OVERALL QUALITY SCORE: {max(0, quality_score)}/100")

if quality_score >= 80:
    print("Grade: A - Excellent data quality!")
elif quality_score >= 70:
    print("Grade: B - Good quality with minor issues")
elif quality_score >= 60:
    print("Grade: C - Acceptable quality, needs attention")
elif quality_score >= 50:
    print("Grade: D - Poor quality, significant issues")
else:
    print("Grade: F - Critical quality issues requiring immediate attention")

print("\\n" + "="*50)
""")

    return "\n\n".join(sections)
