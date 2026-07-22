"""Split legacy per-city model dictionaries into lazy deployment artifacts."""

import argparse
import gc
import hashlib
import json
import re
from pathlib import Path

import joblib


def city_filename(city):
    slug = re.sub(r'[^a-z0-9]+', '-', str(city).lower()).strip('-') or 'city'
    digest = hashlib.sha1(str(city).encode('utf-8')).hexdigest()[:8]
    return f'{slug}-{digest}.pkl'


def split_models(source_path, output_dir, remote_group):
    models = joblib.load(source_path)
    if not isinstance(models, dict):
        raise TypeError(f'{source_path} must contain a city-to-model dictionary.')

    output_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for city, model in sorted(models.items(), key=lambda item: str(item[0])):
        filename = city_filename(city)
        destination = output_dir / filename
        joblib.dump(model, destination, compress=3)
        index[str(city)] = f'per_city/{remote_group}/{filename}'
    del models
    gc.collect()
    return index


def main():
    parser = argparse.ArgumentParser(
        description='Prepare city models for on-demand Hugging Face downloads.'
    )
    parser.add_argument('--models-dir', type=Path, default=Path('Models'))
    parser.add_argument('--output-dir', type=Path, default=Path('Models/per_city'))
    args = parser.parse_args()

    forecasters = split_models(
        args.models_dir / 'per_city_forecasters.pkl',
        args.output_dir / 'forecasters',
        'forecasters',
    )
    classifiers = split_models(
        args.models_dir / 'per_city_models.pkl',
        args.output_dir / 'classifiers',
        'classifiers',
    )
    forecast_features = joblib.load(args.models_dir / 'per_city_forecast_features.pkl')
    manifest = {
        'schema_version': '1.0',
        'forecasters': forecasters,
        'classifiers': classifiers,
        'forecast_features': list(forecast_features),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / 'index.json'
    index_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')

    total_bytes = sum(path.stat().st_size for path in args.output_dir.rglob('*') if path.is_file())
    print(f'Prepared {len(forecasters)} forecasters and {len(classifiers)} classifiers.')
    print(f'Output: {args.output_dir} ({total_bytes / 1024**2:.1f} MB)')
    print(
        'Upload with: hf upload NoVaxiion/project360-assets '
        f'{args.output_dir} per_city --repo-type dataset'
    )


if __name__ == '__main__':
    main()
