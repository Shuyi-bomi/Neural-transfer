# Neural-transfer
This is an application of neural transfer(style + content) on 2 dog images.

The initial dog images look like this:
<p align="middle">
  <img src="https://github.com/Shuyi-bomi/Neural-transfer/blob/main/initial%20picture/pomeranian-900212_1280.jpg" width="200" />
  <img src="https://github.com/Shuyi-bomi/Neural-transfer/blob/main/initial%20picture/australian-shepherd-3237735_1280.jpg" width="200" /> 
</p>
We finish the following 2 steps in preprocess.py.

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


## Neural Transfer
### Neural Style Transfer
Neural style transfer is a method which could generate images in the style of another image. Thus, the neural-style algorithm needs two inputs.  One is a content-image which will be took as input.  The other is a style image.  And the neural-style algorithm will return the content image as if it were paintedusing the artistic style of the style image.  We set image ’australian-shepherd’ as our content image, and image ’pomeranian’ as artistic style.  Which means that we intend to see result of adding pomeranian’s style to dog australian-shepherd.

### Neural Content Transfer
Neural content transfer is similar to style transfer, it's just the mixture of the content(feathers) of 2 images. So basically, we should modify the neural-style algorithm to neural content algorithm in order to record content loss for 2 images.

For both methods, we need to initialize the third image as a exact copy of content image and do backpropogation on the third image.

<p align="middle">
  <img src=https://github.com/Shuyi-bomi/Neural-transfer/blob/main/result/3finalcompare.jpg width="200" />
</p>

