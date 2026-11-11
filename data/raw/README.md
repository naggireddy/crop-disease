# Raw Dataset Mount Directory

Place downloaded external benchmark datasets in their respective subdirectories:

- `PlantVillage/` — Extracted from official PlantVillage repository.
- `PlantDoc/` — Extracted from PlantDoc repository.
- `PlantWild/` — In-the-wild agricultural imagery.
- `FieldData/` — Field-scouted imagery from local agronomist networks.

Alternatively, configure exact directory locations using environment variables:

```bash
export PLANTVILLAGE_ROOT=/path/to/PlantVillage
export PLANTDOC_ROOT=/path/to/PlantDoc
export PLANTWILD_ROOT=/path/to/PlantWild
export FIELD_DATA_ROOT=/path/to/FieldData
```

Run the ingestion and deduplication scanner:

```bash
python 01_Dataset/merge_external_datasets.py
```

