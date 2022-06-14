# README for Longformer Electra

This README quickly explains how to run the longformer_electra adaptation for pretraining with the RTD (replaced token detection) task.

## Setup
Make sure you have pip installed.

Install the requirements with ```pip install -e .```

Make sure you have the ProceduralPosture50K dataset available. If you want to exclude it to test on smaller data, please set the flag ```use_cuad_only``` to True.

## Run
Run the pretraining with ```python -m torch.distributed.launch examples/language_modeling/pretrain_electra.py```