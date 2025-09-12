# EDA Summary

## Dataset Overview
- **Total Records**: 13,862 monthly observations (raw data)
- **Processed Records**: 12,266 observations (after preprocessing pipeline)
- **Time Period**: January 2008 to April 2017 (9.3 years, 112 months)
- **Features**: 58 total (after engineering)
- **Agencies**: 5 MTA agencies
- **KPIs**: 130 unique performance indicators

## Agency Distribution
| Agency | Records | % of Total |
|--------|---------|------------|
| NYC Transit | 7,443 | 60.7% |
| Long Island Rail Road | 1,781 | 14.5% |
| MTA Bus | 1,478 | 12.1% |
| Metro-North Railroad | 1,267 | 10.3% |
| Bridges and Tunnels | 297 | 2.4% |

## Key Performance Metrics Analysis

### Monthly Actual Values Distribution
- **Mean**: 1,989,277 (highly variable across different KPI types)
- **Median**: 95.2 (suggests many percentage-based metrics)
- **Range**: 0 to 161,124,042 (extreme variation due to different KPI scales)
- **Standard Deviation**: 13,669,551 (high variability confirming scale heterogeneity)
- **Quartiles**: 25th (77.8), 75th (99.16) showing concentration around percentage values

### Most Frequent Indicators
1. **Employee Lost Time and Restricted Duty Rate** (199 records)
2. **Elevator Availability** (186 records)
3. **Escalator Availability** (186 records)
4. **West Farms Depot - % of Completed Trips** (100 records)
5. **Queens Village Depot - % of Completed Trips** (100 records)

## Category Analysis
- **Service Indicators**: 11,004+ records (~79%) - Focus on operational performance
- **Safety Indicators**: 1,262+ records (~9%) - Focus on safety metrics
- **Mixed Types**: Percentage-based (77.7%) vs Raw counts (22.3%)

## Data Quality Assessment
- **Missing Values**: 0% - No missing data in raw dataset
- **Data Completeness**: 100% after cleaning and imputation
- **Zero Target Issues Identified**:
  - **Percentage Indicators**: 8.26% have zero targets (890 out of 10,769 records)
  - **Raw Value Indicators**: 12.58% have zero targets (389 out of 3,093 records)
- **Temporal Coverage**: Consistent monthly data, ramp-up period 2008-2009

## Key Insights

### 1. **Scale Heterogeneity**
The dataset contains KPIs with vastly different scales:
- Percentage metrics (0-100): On-time performance, availability rates
- Count metrics (1,000s-100,000s): Ridership, incidents
- Rate metrics (0-10): Injury rates, failure rates

### 2. **Agency Focus Areas**
- **NYC Transit**: Largest dataset, covers subway/bus operations
- **LIRR/Metro-North**: Rail-specific metrics (on-time performance, safety)
- **MTA Bus**: Bus depot operations and completion rates
- **Bridges & Tunnels**: Traffic and safety metrics

### 3. **Temporal Patterns**
- **Data Evolution**: Clear ramp-up from 76 records/month (2008) to stable 130+ records/month (2009-2017)
- **Monthly frequency** appropriate for operational planning
- **Seasonal Stability**: Performance averages consistent across months (85-90% for percentage indicators)
- **Reporting Consistency**: Steady data collection maintained across 9+ years

### 4. **Feature Engineering Success**
- **Calendar features**: year, month, quarter for seasonality
- **Lag features**: 1, 3, 12-month lags for trend analysis
- **Rolling features**: 3, 6, 12-month averages for smoothing
- **Categorical encoding**: Agency and indicator one-hot encoding

## Modeling Implications

### Strengths for ML Modeling:
- Large dataset (12K+ records) suitable for machine learning
- Rich feature set (58 features) after engineering
- No missing data issues
- Cross-agency patterns for learning

### Challenges Addressed:
- **Scale normalization**: Log transformation applied where appropriate
- **Leakage prevention**: All lag/rolling features use only past data
- **Categorical handling**: One-hot encoding for agencies and top indicators
- **Temporal validation**: Time series split for proper evaluation

## Business Value Insights
- **Operational Focus**: ~79% service indicators suggest operational optimization priority
- **Multi-modal Coverage**: Complete MTA system representation across all transport modes
- **Safety Monitoring**: Dedicated safety metrics (~9%) across all agencies
- **Performance Benchmarking**: Consistent metrics enable cross-agency comparison
- **Target-Setting Issues**: 8-13% of records have zero targets requiring imputation strategy

## Validation Notes
*EDA findings validated through comprehensive notebook analysis (notebooks/eda.ipynb) including statistical summaries, visualizations, and data quality assessments. All figures confirmed through direct data analysis.*


FYP Supervisor meeting:
1. dashboard is ok, choose top 5 kpi and explain on them
2. add context on the shapes/patterns on graphs. people might not know what the pattern means
3. documentation explain the graphs also
4. why only forecasting 36 months, why not try for longer (could be a question in presentation later on)