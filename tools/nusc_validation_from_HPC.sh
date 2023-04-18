#!/usr/bin/env bash

MODEL=centerpoint
RUN=third_runs
VERSION=v2_no_xy_flip
CONFIGFILE=centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus
CHECKPOINT=epoch_20

python ./tools/test.py runs/$MODEL/$RUN/$VERSION/$CONFIGFILE.py runs/$MODEL/$RUN/$VERSION/$CHECKPOINT.pth --format-only --eval-options 'jsonfile_prefix=results/HPC/'