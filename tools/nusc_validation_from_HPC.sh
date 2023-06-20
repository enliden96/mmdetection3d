#!/usr/bin/env bash

MODEL=centerpoint
RUN=third_runs
VERSION=baseline_with_fade_2x
CONFIGFILE=centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_fade_resume
CHECKPOINT=epoch_20

python ./tools/test.py runs/$MODEL/$RUN/$VERSION/$CONFIGFILE.py runs/$MODEL/$RUN/$VERSION/$CHECKPOINT.pth --format-only --eval-options 'jsonfile_prefix=results/HPC/baseline_with_fade_2x/'