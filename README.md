# A Scene Text Recognition System of Both Sideways and Upright Text in Arbitrary Orientation
We implement the proposed method under the TensorFlow framework with version 1.6. The CUDA 9.0 and cuDNN v7.0
## **Abstract**
Research of scene text recognition done to date has focused on sideways text recognition. However, it is common that both sideways and upright text appear in one scene. In some Asian countries like China, you may see as much upright text as sideways text in street views. Under such circumstance, it is necessary for a scene text recognition system to recognize both types of text simultaneously. Most scene text recognizers expect the text in all input image to be arranged in the same direction. However, once the text lines in an image can be arbitrarily oriented sideways and upright, it is hard to make sure all detector output images have the same character direction which would cause false recognition. In this paper, we develop a system for scene text recognition of both sideways and upright text in arbitrary orientation. A text orientation estimation module is further proposed to capture the orientation angle information and make sure the character direction is correct for the recognizer. Experimenting on benchmark sideways datasets, our model demonstrates competitive performance compared to state-of-the-arts, with the additional functionality of handling text in different direction and automatically recognizing both sideways and upright text in the same time.

## **Text Recognition System**
<div align=center><img width="500" src="https://github.com/R06942112/OCR/blob/master/system.jpg"/></div>

## **Network Architecture**
<div align=center><img width="500" src="https://github.com/R06942112/OCR/blob/master/architecture.jpg"/></div>

## **Submitted Paper**
[A Scene Text Recognition System of Both Sideways and Upright Text in Arbitrary Orientation](https://drive.google.com/open?id=1jJks6Yot7e1P1XisPjol721vsg4pd__4)

## **Authors**

* **Chia-Lin Chang** - *NTU* - 

* **Prof. Pei-Yuan Wu** - *NTU*- 




