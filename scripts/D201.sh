#!/bin/bash

for i in 1
do
	nohup matlab -nodisplay -r "main_socdl(80,201,1,-1,0.01,100,$i); exit;" >> log/SP[$i]_D201_a8_15.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(90,201,1,-1,0.01,100,$i); exit;" >> log/SP[$i]_D201_a9_15.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(61,201,1,2,1,0.01,$i); exit;" >> log/SP[$i]_D201_a6_k1_31.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(62,201,1,4,1,0.01,$i); exit;" >> log/SP[$i]_D201_a6_k5_31.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(63,201,1,6,1,0.01,$i); exit;" >> log/SP[$i]_D201_a6_k10_31.log 2>&1 &

	nohup matlab -nodisplay -r "main_socdl(64,201,1,8,1,0.01,$i); exit;" >> log/SP[$i]_D201_a6_k20_31.log 2>&1 &
done