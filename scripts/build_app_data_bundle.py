"""Build the compact dashboard data bundle used by Streamlit."""

import argparse
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data import build_lookup_tables, get_aggregate_data, get_officer_trends, load_data


def main():
    parser = argparse.ArgumentParser(
        description='Replace the incident-level runtime CSV with compact dashboard summaries.'
    )
    parser.add_argument('--data-path', type=Path, default=Path('Models/combined_data.csv'))
    parser.add_argument('--output-path', type=Path, default=Path('Models/app_data_bundle.pkl'))
    parser.add_argument('--model-version', default='legacy-1.0')
    args = parser.parse_args()

    raw = load_data.__wrapped__(args.data_path)
    daily = get_aggregate_data.__wrapped__(raw)
    officer_trends = get_officer_trends.__wrapped__(raw)
    lookups = build_lookup_tables.__wrapped__(raw, include_legacy=True)
    crime_distribution = (
        raw.groupby(['city', 'year', 'offense_category_name'], observed=True)
        .size()
        .rename('count')
        .reset_index()
    )

    bundle = {
        'schema_version': '2.0',
        'model_version': args.model_version,
        'daily_city': daily,
        'officer_trends': officer_trends,
        'crime_distribution': crime_distribution,
        'location_areas': lookups['loc_cats'],
        'years': sorted(int(year) for year in raw['year'].unique()),
        'legacy_lookups': {
            key: lookups[key]
            for key in [
                'loc_total_lookup',
                'hour_typical_lookup',
                'avg_div_lookup',
                'avg_loc_lookup',
                'htc_cats',
            ]
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output_path, compress=3)
    print(f'Output: {args.output_path} ({args.output_path.stat().st_size / 1024**2:.1f} MB)')
    print(
        'Upload with: hf upload NoVaxiion/project360-assets '
        f'{args.output_path} app_data_bundle.pkl --repo-type dataset'
    )


if __name__ == '__main__':
    main()
