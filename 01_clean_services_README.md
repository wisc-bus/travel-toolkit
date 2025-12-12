# travel-toolkit

## 01_clean_services_from_csv.py

This script cleans and deduplicates service/business data from a CSV file. It's designed to work with any dataset that contains business or service location data.

### Data Requirements

The input CSV file must contain the following columns:
- `GEOID`: Geographic identifier (string)
- `name`: Business/service name (string)
- `rating`: Rating value (float)
- `lat`: Latitude (float)
- `lon`: Longitude (float)
- `category`: Service category (string)

### Example: Milwaukee Dataset

To use this script with the Milwaukee Yelp services dataset:

1. **Obtain the data**: The Milwaukee dataset (`milwaukee_yelp_services.csv`) contains Yelp business listings for the Milwaukee area. This data typically comes from:
   - Yelp API/Fusion API
   - Yelp Dataset (if available for research)
   - Web scraping (with appropriate permissions and rate limiting)
   - Third-party data providers

2. **Run the script**:
   ```bash
   python 01_clean_services_from_csv.py --input milwaukee_yelp_services.csv --output outputs/milwaukee_yelp_services_clean.csv
   ```
   
   Or using environment variable:
   ```bash
   export SERVICES_RAW=milwaukee_yelp_services.csv
   python 01_clean_services_from_csv.py
   ```

3. **What the script does**:
   - Reads the CSV and validates required columns
   - Removes rows with missing essential data (GEOID, name, lat, lon)
   - Deduplicates businesses by name and location coordinates
   - When duplicates exist, keeps the maximum rating (best representation of business quality)
   - Aggregates GEOIDs and categories for businesses that appear in multiple geographic areas
   - Outputs a cleaned CSV with one row per unique business

### Usage

```bash
python 01_clean_services_from_csv.py --input <input_file> --output <output_file>
```

Arguments:
- `--input`: Path to input CSV file (default: `SERVICES_RAW` env var or `services.csv`)
- `--output`: Path to output CSV file (default: `outputs/<input_basename>_clean.csv`)