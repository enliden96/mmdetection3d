#!/usr/bin/env bash

MODEL=centerpoint
CONFIGFILE=centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus
CHECKPOINT=centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822

python ./tools/test.py configs/$MODEL/$CONFIGFILE.py checkpoints/$CHECKPOINT.pth --format-only --eval-options 'jsonfile_prefix=results'