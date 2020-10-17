# Neural-transfer
This is an application of neural transfer(style + content) on 2 dog images.

The initial dog images look like this:
<p align="middle">
  <img src="https://github.com/Shuyi-bomi/Neural-transfer/blob/main/initial%20picture/pomeranian-900212_1280.jpg" width="200" />
  <img src="https://github.com/Shuyi-bomi/Neural-transfer/blob/main/initial%20picture/australian-shepherd-3237735_1280.jpg" width="200" /> 
</p>

## Detection

We set hyperparameter upsample\_num\_times =0 in detector after tuning. We then draw bounding box after obtaining coordinate of dog's face in both images. Here are the results:
<p align="middle">
  <img src="https://github.com/Shuyi-bomi/Neural-transfer/blob/main/result/1imgde.png" width="200" />
  <img src="https://github.com/Shuyi-bomi/Neural-transfer/blob/main/result/1imgde2.png" width="200" /> 
</p>

## Crop

Then we crop to the same size:
<p align="middle">
  <img src=https://github.com/Shuyi-bomi/Neural-transfer/blob/main/result/imac.jpg width="200" />
  <img src=https://github.com/Shuyi-bomi/Neural-transfer/blob/main/result/imac2.jpg width="200" /> 
</p>


![equation](http://latex.codecogs.com/gif.latex?Concentration%3D%5Cfrac%7BTotalTemplate%7D%7BTotalVolume%7D)  

