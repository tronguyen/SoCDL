#!/bin/bash

for i in 1
do
	nohup matlab -nodisplay -r "main_socdl(80,101,1,-1,1,100,$i); exit;" > log/SP[$i]_L101_a8_35.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(90,101,1,-1,1,100,$i); exit;" > log/SP[$i]_L101_a9_35.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(61,101,1,1,0.1,100,$i); exit;" > log/SP[$i]_L101_a6_k1_25.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(62,101,1,3,0.1,100,$i); exit;" > log/SP[$i]_L101_a6_k5_25.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(63,101,1,5,0.1,100,$i); exit;" > log/SP[$i]_L101_a6_k10_25.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(64,101,1,7,0.1,100,$i); exit;" > log/SP[$i]_L101_a6_k20_25.log 2>&1 &

done