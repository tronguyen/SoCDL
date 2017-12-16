#!/bin/bash

for i in 1
do
	nohup matlab -nodisplay -r "main_socdl(80,202,1,-1,0.1,100,$i); exit;" > log/SP[$i]_D202_a8_25.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(90,202,1,-1,0.1,100,$i); exit;" > log/SP[$i]_D202_a9_25.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(61,202,1,2,0.1,0.01,$i); exit;" > log/SP[$i]_D202_a6_k1_21.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(62,202,1,4,0.1,0.01,$i); exit;" > log/SP[$i]_D202_a6_k5_21.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(63,202,1,6,0.1,0.01,$i); exit;" > log/SP[$i]_D202_a6_k10_21.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(64,202,1,8,0.1,0.01,$i); exit;" > log/SP[$i]_D202_a6_k20_21.log 2>&1 &

done