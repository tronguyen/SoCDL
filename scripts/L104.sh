#!/bin/bash

for i in 1
do
	nohup matlab -nodisplay -r "main_socdl(80,104,1,-1,0.1,100,$i); exit;" > log/SP[$i]_L104_a8_25.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(90,104,1,-1,0.1,100,$i); exit;" > log/SP[$i]_L104_a9_25.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(61,104,1,1,0.1,100,$i); exit;" > log/SP[$i]_L104_a6_k1_25.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(62,104,1,3,0.1,100,$i); exit;" > log/SP[$i]_L104_a6_k5_25.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(63,104,1,5,0.1,100,$i); exit;" > log/SP[$i]_L104_a6_k10_25.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(64,104,1,7,0.1,100,$i); exit;" > log/SP[$i]_L104_a6_k20_25.log 2>&1 &

done