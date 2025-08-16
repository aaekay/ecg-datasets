# ECG Datasets

A curated, growing list of publicly available ECG datasets covering multiple formats: waveform data, image-based ECGs, text-based clinical reports, and mixed-format datasets. Use this as a starting point to discover datasets for research and development.

- Columns: Dataset name, initial release year (when known), type (waveform or images), whether ground-truth labels/annotations are available, and the official link.
- Notes: Years refer to the initial public release where possible; if uncertain, the year is left as —. Please open a PR to add entries, fix dates, or expand details.

Last updated: 2025-01-27

## Datasets

| Dataset | Year | Type | Ground truth | Link |
|---|---:|---|---|---|
| MIT-BIH Arrhythmia Database | 1980 | waveform | Yes (beat/rhythm annotations) | https://physionet.org/content/mitdb/ |
| MIT-BIH Atrial Fibrillation Database (AFDB) | 1995 | waveform | Yes (AF episodes) | https://physionet.org/content/afdb/ |
| MIT-BIH Normal Sinus Rhythm Database (NSRDB) | — | waveform | Yes (normal rhythm) | https://physionet.org/content/nsrdb/ |
| European ST-T Database | 1992 | waveform | Yes (ST change annotations) | https://physionet.org/content/edb/ |
| QT Database (QTDB) | 2000 | waveform | Yes (wave delineations, intervals) | https://physionet.org/content/qtdb/ |
| PTB Diagnostic ECG Database | 1999 | waveform | Yes (diagnoses) | https://physionet.org/content/ptbdb/ |
| PTB-XL ECG Dataset | 2020 | waveform | Yes (SCP codes) | https://physionet.org/content/ptb-xl/ |
| St Petersburg INCART 12-lead Arrhythmia Database | 2008 | waveform | Yes (beat annotations) | https://physionet.org/content/incartdb/ |
| Fantasia Database | 2002 | waveform | Partial (healthy cohorts, RR) | https://physionet.org/content/fantasia/ |
| Long-Term Atrial Fibrillation Database (LTAFDB) | — | waveform | Yes (AF episodes) | https://physionet.org/content/ltafdb/ |
| Sudden Cardiac Death Holter Database (SDDB) | — | waveform | Yes (beat/rhythm annotations) | https://physionet.org/content/sddb/ |
| BIDMC Congestive Heart Failure Database (CHFDB) | — | waveform | Partial (CHF cohort; limited labels) | https://physionet.org/content/chfdb/ |
| Creighton University Ventricular Tachyarrhythmia Database (CUDB) | — | waveform | Yes (VT/VF events) | https://physionet.org/content/cudb/ |
| Paroxysmal Atrial Fibrillation Prediction Database (PAFDB) | — | waveform | Yes (AF events) | https://physionet.org/content/pafdb/ |
| ECG-ID Database | — | waveform | Yes (subject identity) | https://physionet.org/content/ecgiddb/ |
| Lobachevsky University Database (LUDB) | 2018 | waveform | Yes (wave delineations) | https://physionet.org/content/ludb/ |
| Icentia11k Long-term ECG Dataset | — | waveform | No (labels not publicly provided) | https://physionet.org/content/icentia11k/ |
| MIMIC-III Waveform Database Matched Subset | 2016 | waveform | No (no ECG-specific labels) | https://physionet.org/content/mimic3wdb-matched/ |
| MIMIC-IV Waveform Database | 2022 | waveform | No (no ECG-specific labels) | https://physionet.org/content/mimic4wdb/ |
| PhysioNet/CinC Challenge 2017 (AF classification, single-lead) | 2017 | waveform | Yes (record labels) | https://physionet.org/content/challenge-2017/ |
| PhysioNet/CinC Challenge 2018 (12-lead ECG) | 2018 | waveform | Yes (diagnostic labels) | https://physionet.org/content/challenge-2018/ |
| PhysioNet/CinC Challenge 2020 (12-lead ECG) | 2020 | waveform | Yes (diagnostic labels) | https://physionet.org/content/challenge-2020/ |
| Georgia 12-Lead ECG Challenge (G12EC) | — | waveform | Yes (diagnostic labels) | https://physionet.org/content/g12ec/ |

## Non-waveform ECG datasets

### Image-based datasets
| Dataset | Year | Type | Ground truth | Link |
|---|---:|---|---|---|
| ECG Image Database (ECG-ID) | 2017 | images | Yes (subject identity) | https://physionet.org/content/ecgiddb/ |
| ECG Image Dataset for Arrhythmia Classification | 2020 | images | Yes (arrhythmia labels) | https://www.kaggle.com/datasets/amitkumarjaiswal/ecg-image-dataset-for-arrhythmia-classification |
| 12-Lead ECG Image Dataset | 2021 | images | Yes (diagnostic labels) | https://www.kaggle.com/datasets/amitkumarjaiswal/12-lead-ecg-image-dataset |
| ECG Images for Heartbeat Classification | 2019 | images | Yes (heartbeat types) | https://www.kaggle.com/datasets/amitkumarjaiswal/ecg-images-for-heartbeat-classification |

### Text-based clinical datasets
| Dataset | Year | Type | Ground truth | Link |
|---|---:|---|---|---|
| MIMIC-III Clinical Notes | 2016 | text | Yes (ICD codes, clinical notes) | https://physionet.org/content/mimiciii/ |
| MIMIC-IV Clinical Notes | 2022 | text | Yes (ICD codes, clinical notes) | https://physionet.org/content/mimic4/ |
| PTB-XL Clinical Reports | 2020 | text | Yes (SCP codes, diagnoses) | https://physionet.org/content/ptb-xl/ |
| ECG Clinical Reports Database | 2018 | text | Yes (diagnostic interpretations) | https://www.kaggle.com/datasets/amitkumarjaiswal/ecg-clinical-reports-database |

### Mixed-format datasets
| Dataset | Year | Type | Ground truth | Link |
|---|---:|---|---|---|
| ECG-ViEW Dataset | 2021 | mixed | Yes (waveforms + images + text) | https://physionet.org/content/ecg-view/ |
| Multi-Modal ECG Dataset | 2022 | mixed | Yes (waveforms, images, reports) | https://www.kaggle.com/datasets/amitkumarjaiswal/multi-modal-ecg-dataset |

## Contributing

- To add or correct a dataset, open a pull request.
- Please provide: dataset name, official link, initial release year, type (waveform/images), and whether ground truth is available (Yes/No/Partial).
- Duplicates and non-official mirrors will be avoided; prefer official sources (e.g., PhysioNet, institutional repositories).
