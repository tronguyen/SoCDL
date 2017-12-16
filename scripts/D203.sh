#!/bin/bash

for i in 1
do
	nohup matlab -nodisplay -r "main_socdl(80,203,1,-1,0.1,100,$i); exit;" > log/SP[$i]_D203_a8_25.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(90,203,1,-1,0.1,100,$i); exit;" > log/SP[$i]_D203_a9_25.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(61,203,1,2,0.01,0.01,$i); exit;" > log/SP[$i]_D203_a6_k1_11.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(62,203,1,4,0.01,0.01,$i); exit;" > log/SP[$i]_D203_a6_k5_11.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(63,203,1,6,0.01,0.01,$i); exit;" > log/SP[$i]_D203_a6_k10_11.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(64,203,1,8,0.01,0.01,$i); exit;" > log/SP[$i]_D203_a6_k20_11.log 2>&1 &

done