#!/bin/zsh
# Packs what the P5 Colab run needs (code, preregistration, the P4c subspace, the results log) into colab/geomap_colab.zip.
set -e
cd "${0:A:h}/.."
rm -f colab/geomap_colab.zip
zip -qr colab/geomap_colab.zip src scripts docs pyproject.toml results/RESULTS.md \
  results/p4/gemma3-12b_L14_relation_confirm_k2.pt -x '*/__pycache__/*' -x '*.pyc'
ls -lh colab/geomap_colab.zip
