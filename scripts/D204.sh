#!/bin/bash

for i in 1
do
	nohup matlab -nodisplay -r "main_socdl(80,204,1,-1,0.01,100,$i); exit;" > log/SP[$i]_D204_a8_15.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(90,204,1,-1,0.01,100,$i); exit;" > log/SP[$i]_D204_a9_15.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(61,204,1,1,0.01,0.01,$i); exit;" > log/SP[$i]_D204_a6_k1_11.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(62,204,1,4,0.01,0.01,$i); exit;" > log/SP[$i]_D204_a6_k5_11.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(63,204,1,6,0.01,0.01,$i); exit;" > log/SP[$i]_D204_a6_k10_11.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(64,204,1,8,0.01,0.01,$i); exit;" > log/SP[$i]_D204_a6_k20_11.log 2>&1 &

done