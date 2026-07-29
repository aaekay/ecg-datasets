# 🫀 Comprehensive ECG Datasets Collection

*A curated collection of public ECG datasets for machine learning, research, and clinical applications*

[![Stars](https://img.shields.io/github/stars/aaekay/ecg-datasets?style=social)](https://github.com/aaekay/ecg-datasets)
[![Last Updated](https://img.shields.io/badge/last%20updated-Jul--2026-blue)](https://github.com/aaekay/ecg-datasets)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](#contributing)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) *(local license file unavailable; see [change log](change_logs.md))*

## 📋 Table of Contents

- [🏥 Clinical ECG Datasets](#-clinical-ecg-datasets)
- [🔬 Research ECG Datasets](#-research-ecg-datasets)
- [🏆 Competition ECG Datasets](#-competition-ecg-datasets)
- [📊 Dataset Comparison](#-dataset-comparison)
- [🛠️ Tools & Libraries](#-tools--libraries)
- [📚 Benchmarks & Papers](#-benchmarks--papers)
- [🚀 Quick Start Guide](#-quick-start-guide)
- [📖 Dataset Usage Guidelines](#-dataset-usage-guidelines)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🏥 Clinical ECG Datasets

### Large-Scale Clinical Datasets

| Dataset | Year | Records | Patients | Duration | Leads | Sample Rate | Access | License | Link |
|---------|------|---------|----------|----------|-------|-------------|--------|---------|------|
| **ACS-ECG** | 2026 | 19,955 | 18,909 | 10s | 12-lead | 500 Hz | Public | CC BY 4.0 | [Figshare](https://doi.org/10.6084/m9.figshare.29925314) |
| **HEEDB** | 2025 | 11,670,015 | 2,167,795 | 10s | 12-lead | 250-500 Hz | Credentialed | BDSP License | [BDSP](https://bdsp.io/content/heedb/5.0/) |
| **CODE-II** | 2025 | 2,735,269 | 2,093,807 | 10s | 12-lead | 500 Hz | Public | CC BY 4.0 | [arXiv](https://arxiv.org/abs/2511.15632) |
| **MIMIC-IV-ECG** | 2023 | ~800,000 | ~160,000 | 10s | 12-lead | 500 Hz | Credentialed | PhysioNet License | [PhysioNet](https://physionet.org/content/mimic-iv-ecg/) |
| **SPH 12-lead** | 2022 | 25,770 | 24,666 | 10-60s | 12-lead | 500 Hz | Academic | CC BY 4.0 | [Nature Data](https://doi.org/10.1038/s41597-022-01403-5) |
| **CODE-15%** | 2021 | 345,779 | 233,770 | 7.3-10.2s | 12-lead | 400 Hz | Public | CC BY 4.0 | [Zenodo](https://doi.org/10.5281/zenodo.4916206) |
| **Chapman-Shaoxing** | 2020 | 45,152 | 34,905 | 10s | 12-lead | 500 Hz | Public | ODC-BY | [Figshare](https://figshare.com/collections/ChapmanECG/4560497/2) |
| **PTB-XL** | 2020 | 21,837 | 18,885 | 10s | 12-lead | 500 Hz | Public | CC BY 4.0 | [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) |
| **Georgia 12-lead** | 2020 | 10,344 | 10,344 | Variable | 12-lead | 500 Hz | Academic | PhysioNet License | [PhysioNet](https://physionet.org/content/challenge-2020/1.0.2/) |
| **PTB Diagnostic** | 2004 | 549 | 294 | Variable | 15-lead | 1000 Hz | Public | ODC-BY | [PhysioNet](https://physionet.org/content/ptbdb/1.0.0/) |

### Specialized Clinical Datasets

| Dataset | Year | Records | Focus | Duration | Leads | Sample Rate | Access | Link |
|---------|------|---------|--------|----------|-------|-------------|--------|------|
| **ARGO** | 2026 | 1,962 | Post-ischemic ventricular tachycardia | 2.5s | 12-lead + EGM | 1000 Hz | Public | [PhysioNet](https://physionet.org/content/argo/1.0.0/) |
| **VitalDB Arrhythmia** | 2026 | 482 | Intraoperative arrhythmia | ~203 hours total | 1-lead | 500 Hz | Public | [PhysioNet](https://physionet.org/content/vitaldb-arrhythmia/1.0.0/) |
| **Brugada-HUCA** | 2026 | 363 | Brugada syndrome | 12s | 12-lead | 100 Hz | Public | [PhysioNet](https://physionet.org/content/brugada-huca/1.0.0/) |
| **SHDB-AF** | 2025 | 128 | Atrial Fibrillation | ~24 hours | 2-lead | 200 Hz | Public | [PhysioNet](https://physionet.org/content/shdb-af/1.0.1/) |
| **INCART** | 2003 | 75 | Arrhythmia | 30 min | 12-lead | 257 Hz | Public | [PhysioNet](https://physionet.org/content/incartdb/1.0.0/) |
| **MIT-BIH AF** | 2000 | 25 | Atrial Fibrillation | Long-term | 2-lead | 250 Hz | Public | [PhysioNet](https://physionet.org/content/afdb/1.0.0/) |
| **European ST-T** | 1991 | 90 | ST-T Changes | 2 hours | 2-lead | 250 Hz | Public | [PhysioNet](https://physionet.org/content/edb/1.0.0/) |
| **AHA Database** | 1985 | 154 | Arrhythmia | 24 hours | 2-lead | 250 Hz | Restricted | Contact AHA |
| **MIT-BIH Arrhythmia** | 1980 | 48 | Arrhythmia | 30 min | 2-lead | 360 Hz | Public | [PhysioNet](https://physionet.org/content/mitdb/1.0.0/) |

### Continuous Monitoring Datasets

| Dataset | Year | Records | Patients | Duration | Leads | Sample Rate | Access | Link |
|---------|------|---------|----------|----------|-------|-------------|--------|------|
| **Icentia11k** | 2024 | 2B beats | 11,000 | Up to 2 weeks | 1-lead | 250 Hz | Public | [PhysioNet](https://physionet.org/content/icentia11k-continuous-ecg/1.0/) |
| **Sudden Cardiac Death Holter** | 2004 | 23 | 23 | 4-25 hours | 2-lead | 250 Hz | Public | [PhysioNet](https://physionet.org/content/sddb/1.0.0/) |
| **Long-Term ST** | 2003 | 86 | 80 | 21-24 hours | 2-3 lead | 250 Hz | Public | [PhysioNet](https://physionet.org/content/ltstdb/1.0.0/) |
| **Sleep Heart Health Study** | 2003 | 5,804 | 5,804 | Overnight | 1-lead ECG + PSG | 250 Hz | Registered | [NSRR](https://sleepdata.org/datasets/shhs) |
| **Apnea-ECG** | 2000 | 70 | 70 | 7-10 hours | 1-lead | 100 Hz | Public | [PhysioNet](https://physionet.org/content/apnea-ecg/1.0.0/) |
| **MIT-BIH Long-term** | 1999 | 7 | 7 | 14-22 hours | 2-lead | 128 Hz | Public | [PhysioNet](https://physionet.org/content/ltdb/1.0.0/) |
| **MIT-BIH Normal Sinus Rhythm** | 1999 | 18 | 18 | 20-24 hours | 2-lead | 128 Hz | Public | [PhysioNet](https://physionet.org/content/nsrdb/1.0.0/) |

---

## 🔬 Research ECG Datasets

### PhysioNet Research Collections

| Dataset | Year | Records | Subjects | Condition Focus | Duration | Sample Rate | Access | Link |
|---------|------|---------|----------|-----------------|----------|-------------|--------|------|
| **Autonomic Aging** | 2021 | 1,104 | 1,104 | Cardiovascular autonomic aging | 8-45 min | 1000 Hz | Public | [PhysioNet](https://physionet.org/content/autonomic-aging-cardiovascular/1.0.0/) |
| **EPHNOGRAM** | 2021 | 69 | 24 | Simultaneous ECG and PCG | Variable | 8000 Hz | Public | [PhysioNet](https://physionet.org/content/ephnogram/1.0.0/) |
| **BUT QDB** | 2020 | 18 | 15 | ECG signal quality | 24+ hours | 1000 Hz | Public | [PhysioNet](https://physionet.org/content/butqdb/1.0.0/) |
| **CEBSDB** | 2014 | 60 | 20 | ECG, breathing and SCG | 5-50 min | 5000 Hz | Public | [PhysioNet](https://physionet.org/content/cebsdb/1.0.0/) |
| **ECG-ID** | 2014 | 310 | 90 | ECG biometrics | 20s | 500 Hz | Public | [PhysioNet](https://physionet.org/content/ecgiddb/1.0.0/) |
| **Abdominal and Direct Fetal ECG** | 2012 | 5 | 5 | Fetal ECG extraction | 5 min | 1000 Hz | Public | [PhysioNet](https://physionet.org/content/adfecgdb/1.0.0/) |
| **QT Database** | 2003 | 105 | 105 | QT interval analysis | 15 min | 250 Hz | Public | [PhysioNet](https://physionet.org/content/qtdb/1.0.0/) |
| **Fantasia Database** | 2000 | 40 | 40 | Heart rate variability | 120 min | 250 Hz | Public | [PhysioNet](https://physionet.org/content/fantasia/1.0.0/) |
| **MIT-BIH Supraventricular** | 1999 | 78 | 78 | Supraventricular arrhythmias | 30 min | 128 Hz | Public | [PhysioNet](https://physionet.org/content/svdb/1.0.0/) |
| **MIT-BIH ST Change** | 1999 | 28 | 28 | Exercise stress | Variable | 360 Hz | Public | [PhysioNet](https://physionet.org/content/stdb/1.0.0/) |

### Extended Research Datasets

| Dataset | Year | Records | Patients | Special Features | Access | Link |
|---------|------|---------|----------|-----------------|--------|------|
| **Neurotech EEG** | 2026 | 23,607 | 4,914 | Ambulatory EEG with ECG channel, 212,186 hours | Credentialed | [BDSP](https://bdsp.io/content/nf89816gtxbon11kbr9a/1.0/) |
| **EchoNext** | 2026 | Variable | Variable | ECG + echocardiogram, structural heart disease | Restricted | [PhysioNet](https://physionet.org/content/echonext/) |
| **BBBD** | 2026 | 5 experiments | 178 | EEG + ECG + eye tracking during educational videos | Public | [Project](https://bbbd.pythonanywhere.com/) |
| **Yoga EEG-ECG-Video** | 2026 | 49 | 49 | Concentration and mind-wandering, BIDS format | Public | [Nature Data](https://doi.org/10.1038/s41597-026-07209-z) |
| **MIMIC-III-Ext-PPG** | 2026 | 4,920,487 | 6,131 | PPG with simultaneous ECG, ABP and respiration | Credentialed | [PhysioNet](https://physionet.org/content/mimic-iii-ext-ppg/1.1.0/) |
| **MEETI** | 2026 | 784,680 | 160,597 | ECG signals, images, features and interpretations | Credentialed | [GitHub](https://github.com/PKUDigitalHealth/MIMIC-IV-ECG-Ext-Text-Image) |
| **PSG-IPA** | 2026 | 20 | 20 | PSG with ECG and multi-scorer annotations | Public | [PhysioNet](https://physionet.org/content/psg-ipa/1.0.0/) |
| **HOME** | 2025 | Variable | Variable | Evaluation-only consumer single-lead ECG benchmark | Public | [Zenodo](https://doi.org/10.5281/zenodo.18014346) |
| **HK1K** | 2025 | 1,032 | 1,032 | Pediatric EEG + ECG around NREM sleep onset | Public | [Zenodo](https://doi.org/10.5281/zenodo.17138539) |
| **tOLIet** | 2025 | 149 | 86 | Thigh ECG from toilet-seat dry electrodes | Public | [PhysioNet](https://physionet.org/content/tollet/1.0.1/) |
| **OpenECG** | 2025 | 1,200,000 | Multiple | Foundation model benchmark, 9 centers | Public | [arXiv](https://arxiv.org/abs/2503.00711) |
| **Heartcare-220K** | 2025 | 220,000 | 220,000 | Multimodal framework, disease diagnosis | Public | [arXiv](https://arxiv.org/abs/2506.05831) |
| **Icentia11k** | 2024 | 2B beats | 11,000 | Continuous ECG, up to 2 weeks | Public | [PhysioNet](https://physionet.org/content/icentia11k-continuous-ecg/1.0/) |
| **PTB-XL+** | 2023 | 21,837 | 18,885 | Enhanced with extracted features | Public | [PhysioNet](https://physionet.org/content/ptb-xl-plus/1.0.1/) |
| **LUDB** | 2020 | 200 | 200 | Lobachevsky University, annotated | Public | [PhysioNet](https://physionet.org/content/ludb/1.0.1/) |
| **UVA ECG** | 2019 | 1,000+ | 1,000+ | University of Virginia collection | Academic | Request Access |
| **WESAD** | 2018 | 15 | 15 | Wearable chest ECG and multimodal stress signals | Public | [UCI](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) |

---

## 🏆 Competition ECG Datasets

### PhysioNet/CinC Challenges

| Challenge | Year | Records | Task | Best Performance | Access | Link |
|-----------|------|---------|------|-----------------|--------|------|
| **Challenge 2026** | 2026 | 6,600 training PSGs | Cognitive impairment prediction from sleep studies | Official phase | Public | [Moody Challenge](https://moody-challenge.physionet.org/2026/) |
| **Challenge 2025** | 2025 | Multiple sources | Chagas disease detection from ECG | Results released | Public | [Moody Challenge](https://moody-challenge.physionet.org/2025) |
| **Challenge 2024** | 2024 | 21,799 images | ECG Image Digitization | Results released | Public | [Moody Challenge](https://moody-challenge.physionet.org/2024/) |
| **Challenge 2021** | 2021 | 88,253 | Multi-lead ECG Classification | F1: 0.71 | Public | [PhysioNet](https://physionet.org/content/challenge-2021/1.0.3/) |
| **Challenge 2020** | 2020 | 43,101 | 12-lead ECG Classification | F1: 0.533 | Public | [PhysioNet](https://physionet.org/content/challenge-2020/1.0.2/) |
| **Challenge 2017** | 2017 | 12,186 | AF Detection | F1: 0.83 | Public | [PhysioNet](https://physionet.org/content/challenge-2017/1.0.0/) |
| **Challenge 2015** | 2015 | 1,000 | Reducing False Alarms | Score: 81.39 | Public | [PhysioNet](https://physionet.org/content/challenge-2015/1.0.0/) |

### CPSC (China Physiological Signal Challenge)

| Challenge | Year | Records | Task | Leads | Sample Rate | Access | Link |
|-----------|------|---------|------|-------|-------------|--------|------|
| **CPSC 2021** | 2021 | 3,453 | Paroxysmal AF Detection | 1-lead | 200 Hz | Public | [CPSC — unavailable](http://2021.icbeb.org/CPSC2021) |
| **CPSC 2019** | 2019 | 6,877 | Multi-label Classification | 12-lead | 500 Hz | Public | [CPSC](http://2019.icbeb.org/Challenge.html) |
| **CPSC 2018** | 2018 | 13,244 | AF Detection | 1-lead | 300 Hz | Public | [CPSC](http://2018.icbeb.org/Challenge.html) |

### Other Competition Datasets

| Dataset | Platform | Year | Records | Task | Access | Link |
|---------|----------|------|---------|------|--------|------|
| **ECG-5000** | Various | 2024 | 5,000 | Anomaly Detection | Public | [UCR Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) |
| **PTB-XL ECG Images** | Kaggle | 2024 | 21,837 | Synthetic ECG Images | Public | [Kaggle](https://www.kaggle.com/datasets/bjoernjostein/ptb-xl-ecg-image-gmc2024) |
| **ECG Arrhythmia Classification** | Kaggle | 2020 | 4 sources | Multi-class Classification | Public | [Kaggle](https://www.kaggle.com/datasets/sadmansakib7/ecg-arrhythmia-classification-dataset) |
| **ECG Heartbeat Categorization** | Kaggle | 2019 | 109,446 | Beat Classification | Public | [Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) |

---

## 📊 Dataset Comparison

### By Size and Scale

| Dataset | Records | Patients | Total Hours | Data Size | Year |
|---------|---------|----------|-------------|-----------|------|
| HEEDB | 11,670,015 | 2,167,795 | 32,417 | Variable | 2025 |
| CODE-II | 2,735,269 | 2,093,807 | 7,598 | ~500 GB | 2025 |
| MIMIC-IV-ECG | ~800,000 | ~160,000 | ~2,222 | ~150 GB | 2023 |
| CODE-15% | 345,779 | 233,770 | ~960 | Variable | 2021 |
| PhysioNet 2021 | 88,253 | 88,253 | 245.1 | ~15 GB | 2021 |
| Chapman-Shaoxing | 45,152 | 34,905 | 125.4 | ~8.2 GB | 2020 |
| SPH 12-lead | 25,770 | 24,666 | Variable | ~5.1 GB | 2022 |
| PTB-XL | 21,837 | 18,885 | 60.7 | ~2.5 GB | 2020 |
| MIT-BIH Arrhythmia | 48 | 47 | 24 | ~23 MB | 1980 |

### By Clinical Condition

| Condition | Primary Datasets | Total Records | Best Performance |
|-----------|------------------|---------------|-----------------|
| **Arrhythmia** | MIT-BIH, PTB-XL, Chapman, CODE-II, HEEDB | 14,000,000+ | 99.3% Acc |
| **Atrial Fibrillation** | MIT-BIH AF, CPSC 2018/2021, Icentia11k, MIMIC-IV-ECG | 800,000+ | AUROC: 0.996 (ECG-FM) |
| **Myocardial Infarction** | PTB-XL, PTB Diagnostic, CODE-II, HEEDB | 14,000,000+ | AUC: 0.95+ |
| **Structural Heart Disease** | EchoNext | Variable | 77% Acc (EchoNext) |
| **Normal vs Abnormal** | All major datasets | 20,000,000+ | 98.7% Acc |
| **Multi-label (150+ classes)** | HEEDB, PTB-XL, Chapman, SPH, CODE-II | 14,000,000+ | AUROC >0.95 (ECGFounder) |

### By Data Type and Format

| Data Type | Datasets | Advantages | Use Cases |
|-----------|----------|------------|-----------|
| **Raw Waveform** | HEEDB, PTB-XL, Chapman, MIT-BIH, CODE-II, Icentia11k, MIMIC-IV-ECG | High fidelity, full information | Deep learning, signal processing |
| **Continuous Monitoring** | Icentia11k, Long-Term ST, Sudden Cardiac Death Holter, Apnea-ECG | Long-term recordings, hours to weeks | Arrhythmia detection, HRV analysis |
| **Processed Features** | PTB-XL+ | Pre-extracted features | Traditional ML, quick prototyping |
| **Images** | PTB-XL Images, Challenge 2024 | Visual interpretation | Computer vision, image-based ML |
| **Multimodal** | EchoNext, MEETI, BBBD, HK1K, Heartcare-220K | ECG + other clinical data | Comprehensive diagnosis |
| **Foundation Model Training** | HEEDB, OpenECG, MIMIC-IV-ECG | Large-scale pre-training | Self-supervised learning, transfer learning |
| **Annotations** | Most PhysioNet datasets, HEEDB, ARGO, VitalDB Arrhythmia, BUT QDB | Expert labels, ICD codes | Supervised learning, validation |

---

## 🛠️ Tools & Libraries

### Data Access and Processing

| Tool | Language | Purpose | Installation |
|------|----------|---------|-------------|
| **WFDB** | Python/MATLAB | PhysioNet data access | `pip install wfdb` |
| **NeuroKit2** | Python | Neurophysiological signals | `pip install neurokit2` |
| **BioSPPy** | Python | Biosignal processing | `pip install biosppy` |
| **HeartPy** | Python | Heart rate analysis | `pip install heartpy` |
| **PyECG** | Python | ECG analysis toolkit | `pip install pyecg` |
| **ECGtizer** | Python | Paper ECG digitization | [GitHub — unavailable](https://github.com/ECGtizer) |
| **CardioMark** | MATLAB | ECG annotation tool | [GitHub](https://github.com/CardioMark) |

### Visualization and Analysis

| Tool | Purpose | Key Features |
|------|---------|--------------|
| **ECG-Plot** | ECG visualization | Multi-lead plotting, annotations |
| **PlotlyECG** | Interactive plots | Web-based, interactive ECG plots |
| **Matplotlib** | Static plots | Publication-quality figures |
| **Bokeh** | Interactive visualization | Real-time ECG monitoring |

### AI-Powered ECG Analysis Tools

| Tool | Purpose | Key Features | Access |
|------|---------|--------------|--------|
| **DeepECG** | Real-time ECG analysis | Comprehensive measurements, AI-powered | [DeepECG.ai](https://deepecg.ai) |
| **Qaly** | Expert ECG review | Certified experts, 30+ rhythm detection | [Qaly.co](https://www.qaly.co) |
| **HeartKey Rhythm** | FDA-cleared ECG evaluation | Suite of algorithms, wearable device support | [B-Secur](https://www.b-secur.com) |
| **EchoNext** | Structural heart disease detection | ECG + Echocardiogram, 77% accuracy | [PhysioNet](https://physionet.org/content/echonext/) |
| **VARS** | Versatile ECG analysis | Graph-based representation, risk-sensitive | [arXiv](https://arxiv.org/abs/2511.07973) |

### ECG Foundation Models

| Model | Year | Training Data | Parameters | Key Capability | Open Source | Link |
|-------|------|---------------|------------|----------------|-------------|------|
| **ECGFounder** | 2024 | 10.7M ECGs (HEEDB) | - | 150 diagnostic categories, AUROC >0.95 for 80 diagnoses | Yes | [arXiv](https://arxiv.org/abs/2410.04133), [GitHub](https://github.com/PKUDigitalHealth/ECGFounder) |
| **AnyECG** | 2026 | 13.3M ECGs | - | 1,172 conditions, holistic health profiling, future risk prediction | - | [arXiv](https://arxiv.org/abs/2601.10748) |
| **ECG-FM** | 2024 | 1.5M ECGs (MIMIC-IV + PhysioNet) | 90.9M | AF detection AUROC 0.996, open weights | Yes | [arXiv](https://arxiv.org/abs/2408.05178), [GitHub](https://github.com/bowang-lab/ecg-fm) |
| **CardX** | 2025 | 1M+ ECGs | Efficient | ExChanGeAI platform, local fine-tuning | Yes (MIT) | [arXiv](https://arxiv.org/abs/2503.13570) |
| **ECGFM** | 2025 | 1M+ multi-center ECGs | - | Contrastive + generative + diagnostic text generation | - | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1566253525004361) |

### ECG Language Models

| Model | Year | Task | Key Innovation | Link |
|-------|------|------|----------------|------|
| **ECG-GPT** | 2024 | ECG image interpretation | Vision encoder-decoder, format-independent, validated on 3.8M ECGs | [medRxiv](https://www.medrxiv.org/content/10.1101/2024.02.17.24302976v2) |
| **CAMEL** | 2026 | Cardiac event forecasting | First ELM for forecasting, +7% on ECGBench, +12.4% on ECGForecastBench | [arXiv](https://arxiv.org/abs/2602.15677) |
| **GEM** | 2025 | Grounded ECG understanding | Unifies time series + images + text, NeurIPS 2025, +22.7% explainability | [arXiv](https://arxiv.org/abs/2503.06073), [GitHub](https://github.com/lanxiang1017/gem) |
| **ELF** | 2026 | ECG interpretation | Encoder-free, single projection layer, matches SOTA | [arXiv](https://arxiv.org/abs/2601.18798) |
| **RhythmBERT** | 2026 | Heart disease detection | Self-supervised on latent ECG representations | [arXiv](https://arxiv.org/abs/2602.23060) |

### Machine Learning Frameworks

| Framework | ECG-Specific Features | Popular Models |
|-----------|----------------------|----------------|
| **TensorFlow** | tf.signal for ECG processing | CNN, LSTM, Transformers |
| **PyTorch** | torchaudio for signals | ResNet1D, WaveNet, TCN |
| **scikit-learn** | Classical ML algorithms | SVM, Random Forest, XGBoost |

---

## 📚 Benchmarks & Papers

### Key Survey Papers

| Paper | Year | Citations | Focus |
|-------|------|-----------|--------|
| "A Systematic Review on Foundation Models for Electrocardiogram Analysis" | 2025 | New | Foundation model architectures, pre-training, adaptation |
| "Deep learning and electrocardiography: systematic review" | 2025 | New | 198 publications, comprehensive DL survey |
| "Generalizability of electrocardiographic artificial intelligence" | 2025 | New | ECG-AI generalizability across populations |
| "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL" | 2021 | 400+ | PTB-XL benchmarking |
| "Automatic diagnosis of the 12-lead ECG using a deep neural network" | 2020 | 800+ | Deep learning methods |
| "ECG arrhythmia classification using a 2-D convolutional neural network" | 2018 | 1000+ | CNN for arrhythmia |

### Recent High-Impact Papers (2024-2026)

| Paper | Year | Focus | Link |
|-------|------|-------|------|
| "CAMEL: An ECG Language Model for Forecasting Cardiac Events" | 2026 | First ELM for cardiac event forecasting | [arXiv](https://arxiv.org/abs/2602.15677) |
| "RhythmBERT: Self-Supervised Language Model for Heart Disease Detection" | 2026 | Self-supervised latent ECG representations | [arXiv](https://arxiv.org/abs/2602.23060) |
| "AnyECG: Evolved ECG Foundation Model for Holistic Health Profiling" | 2026 | 1,172 conditions, future risk prediction | [arXiv](https://arxiv.org/abs/2601.10748) |
| "ELF: Encoder-Free ECG Language Model" | 2026 | Simplified ELM architecture | [arXiv](https://arxiv.org/abs/2601.18798) |
| "Harvard-Emory ECG Database" | 2026 | Largest credentialed ECG database (11.7M ECGs) | [Nature Data](https://www.nature.com/articles/s41597-026-06861-9) |
| "GEM: Empowering MLLM for Grounded ECG Understanding" | 2025 | Multimodal ECG + images + text, NeurIPS 2025 | [arXiv](https://arxiv.org/abs/2503.06073) |
| "OpenECG: Benchmarking ECG Foundation Models with 1.2M Records" | 2025 | Foundation model benchmark, 9 centers | [arXiv](https://arxiv.org/abs/2503.00711) |
| "ExChanGeAI: End-to-End Platform for ECG Analysis and Fine-tuning" | 2025 | Open-source ECG platform + CardX model | [arXiv](https://arxiv.org/abs/2503.13570) |
| "CODE-II: A Large-Scale ECG Dataset with 66 Diagnostic Classes" | 2025 | Large-scale clinical dataset | [arXiv](https://arxiv.org/abs/2511.15632) |
| "VARS: VersAtile and Risk-Sensitive Cardiac Diagnosis" | 2025 | Graph-based ECG representation | [arXiv](https://arxiv.org/abs/2511.07973) |
| "Heartcare Suite: Multimodal Framework for ECG Analysis" | 2025 | Multimodal ECG analysis, HeartcareGPT | [arXiv](https://arxiv.org/abs/2506.05831) |
| "ECGFM: A Foundation Model Trained on Multi-Center Million-ECG Dataset" | 2025 | Contrastive + generative pre-training | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1566253525004361) |
| "ECGFounder: An ECG Foundation Model Built on 10M+ Recordings" | 2024 | 150 diagnostic categories, expert-level | [arXiv](https://arxiv.org/abs/2410.04133) |
| "ECG-GPT: AI-Based Automated Interpretation of ECG Images" | 2024 | Vision-based, format-independent | [medRxiv](https://www.medrxiv.org/content/10.1101/2024.02.17.24302976v2) |
| "ECG-FM: An Open Electrocardiogram Foundation Model" | 2024 | Open-weight transformer, 90.9M params | [arXiv](https://arxiv.org/abs/2408.05178) |
| "ECGtizer: Digitizing Paper ECGs with Deep Learning" | 2024 | Paper ECG digitization | [arXiv](https://arxiv.org/abs/2412.12139) |
| "EchoNext: AI-Enhanced ECG for Structural Heart Disease" | 2025 | Structural heart disease detection | [PhysioNet](https://physionet.org/content/echonext/) |

### State-of-the-Art Results

#### PTB-XL Benchmark (Multi-label Classification)
| Method | Year | Macro F1 | AUC |
|--------|------|----------|-----|
| **Transformer** | 2022 | 0.389 | 0.941 |
| **ResNet1D-GN** | 2021 | 0.351 | 0.928 |
| **WaveNet** | 2021 | 0.341 | 0.925 |
| **LSTM** | 2021 | 0.325 | 0.919 |

#### CODE-II Benchmark (66-class Classification)
| Method | Year | Macro F1 | Accuracy |
|--------|------|----------|----------|
| **HeartcareGPT** | 2025 | - | SOTA |
| **VARS** | 2025 | - | Superior performance |
| **Transformer-based** | 2025 | - | High accuracy |

#### MIT-BIH Arrhythmia (5-class)
| Method | Year | Accuracy | Sensitivity |
|--------|------|----------|-------------|
| **CNN-LSTM** | 2023 | 99.3% | 98.7% |
| **ResNet** | 2022 | 99.1% | 98.5% |
| **SVM+Wavelet** | 2019 | 97.8% | 96.2% |

#### Foundation Model Benchmarks
| Method | Year | Key Metric | Scope |
|--------|------|------------|-------|
| **AnyECG** | 2026 | AUROC >0.7 for 306 diseases | 1,172 conditions, 13.3M ECGs |
| **ECGFounder** | 2024 | AUROC >0.95 for 80 diagnoses | 150 categories, 10.7M ECGs |
| **ECG-FM** | 2024 | AUROC 0.996 (AF), 0.929 (low LVEF) | Open-weight, 1.5M ECGs |
| **CAMEL** | 2026 | +12.4% on ECGForecastBench | Cardiac event forecasting |
| **GEM** | 2025 | +22.7% explainability | Grounded multimodal interpretation |

#### Structural Heart Disease Detection
| Method | Year | Accuracy | Dataset |
|--------|------|----------|---------|
| **EchoNext** | 2024 | 77% | EchoNext (vs. 64% cardiologists) |
| **AI-ECG for HCM** | 2024 | High | Cleveland Clinic study |

### Recent Research Trends (2024-2026)

- **Foundation Models**: Large pre-trained models achieving expert-level performance (ECGFounder, AnyECG, ECG-FM, CardX, ECGFM)
- **ECG Language Models**: LLM-based ECG interpretation and report generation (CAMEL, GEM, ELF, RhythmBERT, ECG-GPT)
- **Cardiac Event Forecasting**: Predicting future adverse cardiac outcomes from ECG signals (CAMEL's ECGForecastBench)
- **Grounded/Explainable Interpretation**: Linking diagnoses to measurable ECG parameters (GEM, VARS)
- **Holistic Health Profiling**: ECG-based prediction across 1,000+ conditions including non-cardiac diseases (AnyECG)
- **Self-supervised Learning**: Learning from unlabeled ECG data; BYOL and MAE outperform contrastive approaches (OpenECG)
- **Multi-modal Analysis**: Combining ECG time series, images, and text (GEM, EchoNext, Heartcare Suite)
- **Large-Scale Datasets**: HEEDB (10.6M ECGs), CODE-II (2.7M), MIMIC-IV-ECG (800K), Icentia11k
- **Open-Source Platforms**: Democratizing ECG deep learning for non-experts (ExChanGeAI, ECG-FM)
- **Federated Learning**: Privacy-preserving ECG analysis across institutions
- **Paper ECG Digitization**: Automated recovery of signals from paper records (ECGtizer)
- **Graph-Based Representations**: Novel approaches for heterogeneous ECG signals (VARS)
- **Real-Time Analysis**: AI-powered platforms for clinical decision support (DeepECG, Qaly)
- **FDA-Cleared Tools**: Regulatory-approved AI algorithms for clinical use (HeartKey Rhythm)

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
pip install wfdb pandas numpy matplotlib scipy
pip install torch torchvision  # For deep learning
pip install scikit-learn xgboost  # For traditional ML
```

### 2. Loading PTB-XL Dataset
```python
import wfdb
import pandas as pd
import numpy as np

# Load PTB-XL metadata
Y = pd.read_csv('ptbxl_database.csv', index_col='ecg_id')
X = np.array([wfdb.rdsamp(f'records500/{row.filename_lr}')[0] 
              for _, row in Y.iterrows()])
```

### 3. Loading MIT-BIH Dataset
```python
import wfdb

# Load a single record
record = wfdb.rdrecord('mitdb/100')
annotation = wfdb.rdann('mitdb/100', 'atr')

signals = record.p_signal
labels = annotation.symbol
```

### 4. Loading CODE-II Dataset (2025)
```python
# CODE-II dataset access instructions
# See: https://arxiv.org/abs/2511.15632
# Dataset contains 2.7M ECGs with 66 diagnostic classes
# Access through official CODE-II repository
```

### 5. Basic Preprocessing
```python
from scipy import signal

def preprocess_ecg(ecg_signal, fs=500):
    # Bandpass filter (0.5-40 Hz)
    b, a = signal.butter(2, [0.5, 40], btype='band', fs=fs)
    filtered = signal.filtfilt(b, a, ecg_signal)
    
    # Normalize
    normalized = (filtered - np.mean(filtered)) / np.std(filtered)
    return normalized
```

---

## 📖 Dataset Usage Guidelines

### Citation Requirements

When using these datasets, please cite appropriately:

**PTB-XL:**
```
Wagner, P., Strodthoff, N., Bousseljot, R. D., Kreiseler, D., Lunze, F. I., Samek, W., & Schaeffter, T. (2020). 
PTB-XL, a large publicly available electrocardiography dataset. Scientific Data, 7(1), 1-15.
```

**MIT-BIH:**
```
Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. 
IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).
```

**CODE-II:**
```
[Citation information to be added - see arXiv:2511.15632]
```

**Icentia11k:**
```
[Citation information to be added - see PhysioNet]
```

**HEEDB:**
```
Reyna, M.A., Deepanshi, Weigle, J., et al. (2026).
The Harvard-Emory ECG Database. Scientific Data.
```

**MIMIC-IV-ECG:**
```
Gow, B., Pollard, T., Nathanson, L.A., Johnson, A., Moody, B., Fernandes, C., et al. (2023).
MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset. PhysioNet.
```

**EchoNext:**
```
[Citation information to be added - see PhysioNet]
```

### Ethical Considerations

- **Privacy**: All datasets are anonymized, but follow institutional guidelines
- **Clinical Use**: These datasets are for research only, not clinical diagnosis
- **Bias**: Be aware of demographic and geographic biases in datasets
- **Validation**: Always validate models on independent test sets

### Data Preprocessing Best Practices

1. **Filtering**: Apply appropriate bandpass filters (typically 0.5-40 Hz)
2. **Normalization**: Standardize signals for consistent model training
3. **Segmentation**: Use appropriate window sizes (typically 2.5-10 seconds)
4. **Augmentation**: Consider data augmentation for small datasets
5. **Quality Control**: Remove noisy or corrupted recordings

---

## 🤝 Contributing

We welcome contributions to this repository! Here's how you can help:

### Adding New Datasets
1. Fork this repository
2. Add dataset information to the appropriate table
3. Include proper citations and links
4. Verify all information is accurate
5. Submit a pull request

### Required Information for New Datasets
- Dataset name and year
- Number of records and patients
- Data format and specifications
- Access requirements and licensing
- Official links and citations
- Any special features or limitations

### Updating Existing Information
- Correction of errors
- Addition of new papers or benchmarks
- Updates to access links
- Performance improvements

### Guidelines
- Verify all links are working
- Include proper citations
- Use consistent formatting
- Provide accurate technical specifications

---

## 🏷️ Tags and Keywords

`ecg-datasets` `electrocardiogram` `cardiology` `machine-learning` `deep-learning` `foundation-models` `ecg-language-models` `arrhythmia` `heart-rhythm` `physionet` `clinical-data` `medical-ai` `signal-processing` `healthcare` `biomedical-engineering` `cardiac-monitoring` `ecg-classification` `heart-disease` `medical-datasets` `public-health` `cardiovascular` `wearable-devices`

---

## 📄 License

This repository is licensed under the MIT License. However, individual datasets may have their own licenses - please check each dataset's specific licensing terms before use.

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/aaekay/ecg-datasets/issues)
- **Discussions**: [GitHub Discussions — not enabled](https://github.com/aaekay/ecg-datasets/discussions)

---

## ⭐ Star History

If you find this repository useful, please consider giving it a star! 

[![Star History Chart](https://api.star-history.com/svg?repos=aaekay/ecg-datasets&type=Date)](https://star-history.com/#aaekay/ecg-datasets&Date)

---

**Last Updated**: July 2026 | **Total Datasets**: 90+ | **Total Records**: 20,000,000+
