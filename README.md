![](logo.jpeg)

# Zero-shot Lidar DeBlur (ZSLDB)

This repository contains the code for the paper A. Montanaro, D. Valsesia - "A novel method and dataset for depth-guided image deblurring from smartphone Lidar", ICIP 2025.

## Dataset

The method was tested on the novel dataset of low-light smartphone images with associated Lidar depth maps available at [https://github.com/diegovalsesia/licam-dataset](https://github.com/diegovalsesia/licam-dataset). In particular the "resized_45" data were used.


## Code

The code can be launched with the launcher.sh script. Notice that the script provides two examples of usage in order to test the method with or without depth map guidance.
``
./launcher.sh
``

The results in the paper have been obtained by initializing the blur kernels with the estimations provided by J-MKPD. Refer to [their repository](https://github.com/GuillermoCarbajal/J-MKPD) for implementation. The kernels are saved as npy files in the directory specified in the "labels" variable in the launcher.

Anaconda environment dependendecies are list in the requirements.yml file.


## Acknowledgment

The LICAM -“AI-powered LIDAR fusion for next-generation smartphone cameras (LICAM)” project is funded by European Union – Next Generation EU within the PRIN 2022 program (D.D. 104 - 02/02/2022 Ministero dell’Università e della Ricerca). The contents of this website reflect only the authors' views and opinions and the Ministry cannot be considered responsible for them.
