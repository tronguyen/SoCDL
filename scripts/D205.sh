#!/bin/bash

for i in 1
do
	nohup matlab -nodisplay -r "main_socdl(80,205,1,-1,0.01,100,$i); exit;" > log/SP[$i]_D205_a8_15.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(90,205,1,-1,0.01,100,$i); exit;" > log/SP[$i]_D205_a9_15.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(61,205,1,2,0.01,0.01,$i); exit;" > log/SP[$i]_D205_a6_k1_11.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(62,205,1,4,0.01,0.01,$i); exit;" > log/SP[$i]_D205_a6_k5_11.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(63,205,1,6,0.01,0.01,$i); exit;" > log/SP[$i]_D205_a6_k10_11.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(64,205,1,8,0.01,0.01,$i); exit;" > log/SP[$i]_D205_a6_k20_11.log 2>&1 &

done