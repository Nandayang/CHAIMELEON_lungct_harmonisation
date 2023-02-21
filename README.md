# CHAIMELEON_lungct_harmonisation

## How to use this docker  
Step1: Download the reliable weights through  
https://drive.google.com/file/d/1v-zjlJvEtvOpQ8NGVHkpcm3v17QlMwWx/view?usp=sharing     
Place the downloaded files at the same layer as the 'weight.pth'

Step2: Build the docker through the following command
```sudo docker build -t harmonisation_lungct .```    

Step3: Run the docker through    
```sudo docker run --rm -v [input_datapath]:/workspace/inputs/ -v [output_datapath]:/workspace/outputs/ harmonisation_lungct:latest /bin/bash -c "sh predict.sh"```
