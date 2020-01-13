# Grand Challenge: 3-D Urban Objects Detection and Classification From Point Clouds

This is the code repository for the paper on https://dl.acm.org/doi/10.1145/3328905.3330298

This could be a possible solution for DEBS 19 grand challenge. http://debs2019.org/Calls/Call_for_Grand_Challenge_Solutions.html


This repository provides a sample HTTP-client to which detects and classifies LiDAR dataset of DEBS Grand Challenge's 2019 edition.
Note, the example is written in Python. However, some part of the solution is written in c++, the reason for that is that Python PCL does not provide all the functionalities of the PCL library.

as long as it is communicated through the HTTP/REST interface of the benchmark system which is available here https://github.com/debs-2019-challenge/debs-2019-challenge
