#!/bin/bash

rm -rf tmp/
find . -type d -name .ipynb_checkpoints -exec rm -rf {} \;