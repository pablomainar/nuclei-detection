# nuclei-detection

### How to use
Run "fast_test.py" to test the program on some of the test images of the competition. The images are under the directory "stage1_test", the trained model is under the directory "models".

### About the project
This project is my solution to the 2018 Data Science Bowl held in Kaggle. You can see the competition rules and data in this link: https://www.kaggle.com/c/data-science-bowl-2018. The goal was to build a program to automatically segment nuclei of cells in microscopy images. It is a segmentation problem, so it is crucial not only to distinguish between background and nuclei, but also to separate between adjacent nuclei. For more details, please refer to the competition webpage.

This was my first competition in Kaggle. I started the competition quite late, when there were only 9 days to finish. Therefore I only had time to build a very simple CNN and I finished in position 638 out of 3634 (top 18%) with a score of 0.235. However, since my goal was to learn and not to compete, I spent a few weeks more trying to improve my solution after the competition had finished. With the final solution that can be found here I obtained a score of 0.358 that would have raised me to position 521 (top 14%).

My main limitation was my available hardware. I could only use my old laptop to do everything, so I did not have acces to a GPU and the RAM was very limited. This meant that training was painfully slow and uncomfortable, the CNN was forced to be relatively shallow and I could not use any pre-trained net like VGG or ResNet to initialize it, so I had to train it from scratch.

### Main aspects of solution

- It is based on U-Net (https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).
- Since this is an instance segmentation problem the training is done end-to-end and there is no need for fully connected layers. This allows meto use any size of image as input/output, which is an advantage given that both the train and test images have very different sizes. To simplify things, training is done with random crops of 256x256 pixels, while the test images are processed "as they are", without resizing them.
- Heavy data augmentation: We only had 670 images to train with, so to generalize better and avoid overfitting it was crucial to do data augmentation. As said before, random crops of 256x256 pixels are used from the training images. To this crops I also apply rotation, vertical/horizontal flipping, random cropping, rgb channel shuffle, gaussian noise and average/gaussian blurring.
- In a first solution I trained end-to-end with the masks given by the competition. However I soon realized that this was a problem because some nuclei were touching each other and since this is an instance segmentation task, they should be labeled as different objects. My net was unable to split nuclei that were very close to each other. To solve this I did two things:
  - First, I eroded all the masks of the nuclei to make sure that there was always a background space between them. In a post-processing step I dilate again the predicted masks, but they have been already labeled so they are split even if the nuclei are touching each other.
  - Secondly, I found the contours for each mask and added a second decoding branch to the U-Net to predict them. This way the encoding part of the U-Net is shared by both outputs, while the decoding part is formed by one branch that predicts the labels and the other that predicts the contours. I combine these two outputs to create the final prediction. A pixel in the final prediction will be labeled as 1 (nuclei) if the label probability is higher than a threshold (I found that 0.6 was the best) and at the same time the contour probability is smaller than a threshold (0.5). If not, the pixel is labeld as 0 (background).
- In postprocessing I only dilate the predicted masks as explained before. I also tried some other morphology actions (opening and closing) amd watershed segmentation but the results were similar or even worse, so I discarted them.
