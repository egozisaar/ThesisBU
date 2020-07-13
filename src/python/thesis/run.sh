#!/bin/bash

for j in $(seq 0 0.001 1); do
  ts python main.py -n 10000 -k 1000 -o 2 -tc "$j" -es 1000
done
