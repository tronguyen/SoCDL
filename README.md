## SOCDL: Social Collaborative Deep Learning
-------------------------------------------
[Social network and Item content integration for recommendation]

### INTRODUCTION

- This is a raw code in matlab for the paper "Collaborative Topic Regression with Denoising AutoEncoder for Content and Community Co-Representation" (Nguyen & Lauw, CIKM 2017)
- The pakage includes a few algorithms in social recommendations: CDL, SoRec, PMF, CTR, CTR-SMF, and SoCDL(this paper)
- Feel free for any modification from this raw code


### Run

CMD: matlab ./main_socdl.m [alg_index]
, where alg_index:
- 1: SoCDL
- 2: CDL for item-side
- 3: CDL for user-side
- 4: PMF
- 5: SoRec
- 6/7: CDL-CTR combine
- 8: CTR
- 9: CTR-SMF

### Cite

@inproceedings{\
 Nguyen:2017:CTR:3132847.3133128,\
 author = {Nguyen, Trong T. and Lauw, Hady W.},\
 title = {Collaborative Topic Regression with Denoising AutoEncoder for Content and Community Co-Representation},\
 booktitle = {Proceedings of the 2017 ACM on Conference on Information and Knowledge Management},\
 series = {CIKM '17},\
 publisher = {ACM},\
 address = {New York, NY, USA}\
} 
