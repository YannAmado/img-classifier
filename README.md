<!--Template taken from https://github.com/othneildrew/Best-README-Template/blob/master/README.md -->
<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">Easy-Img-Classifer-Pipeline</h3>

  <p align="center">
    An awesome image classifier pipeline to make classifying images easy! 
    <br />
    <a href="https://github.com/YannAmado/img-classifier"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/YannAmado/img-classifier">View Demo</a>
    ·
    <a href="https://github.com/YannAmado/img-classifier/issues">Report Bug</a>
    ·
    <a href="https://github.com/YannAmado/img-classifier/issues">Request Feature</a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#how-to-use">How to Use</a></li>
    <li><a href="#demos">Demos</a></li>
    <li><a href="#tips">Tips</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

[![Image Classifier][product-screenshot]](https://example.com)

Making image classifiers is a very troublesome task, having very repetitive steps that are simple but also boring  

Some of the things you always need to do are:
* Data Augment your dataset
* Create your models
* Train and Test your models
* Compare the models trained

Everything is simple enough if you have ever used TensorFlow or Keras before, but tracking all of the possible configurations and repeating everything was often a pain to me.
That's why I decided to make this pipeline to make everything extremely easy!

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/YannAmado/img-classifier.git
   ```
2. Install PIP packages
   ```sh
   pip install keras
   pip install numpy
   pip install pandas
   pip install scikit-learn
   pip install split_folders
   pip install tqdm
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- HOW TO USE -->
## How to use

### Setting config.py

config.py is where the code takes all the necessary settings to do everything, change the necessary parameters as desired for your own project.

### To run

Just run main.py as normal and everything will automatically be done as you setup on the previous step!

### To add new models

Currently, the models supported are MobileNetV2, Residual Networks (ResNet50) and EfficientNetB4, if you wish to add more or customize your own just define it on create_model.py and select it on main.py

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- DEMOS -->
## Demos

All demos were made using only 10 classes because of computational power constraints

### Stanford cars

* MobileNetV2: 98.4% accuracy
* EfficientNetB4: 98.1% accuracy
* ResNet50: 96.5% accuracy

### Simpsons Characters

* MobileNetV2: 97.8% accuracy
* EfficientNetB4: 97.3% accuracy
* ResNet50: 96.1% accuracy

### CIFAR-10

* MobileNetV2: 97.1% accuracy
* EfficientNetB4: 98.3% accuracy
* ResNet50: 97.7% accuracy

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- TIPS -->
## Tips
* The bigger the augmented dataset, the better
* The more diverse the augmented dataset (meaning, choosing different augmenation parameters), the better
* All models used benefit from a larger image size, for training I used (224,224)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

[![LinkedIn][linkedin-shield]][linkedin-url] 

Email: yannamado.n@gmail.com

Project Link: [https://github.com/YannAmado/img-classifier](https://github.com/YannAmado/img-classifier)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/yannamado
[product-screenshot]: images/screenshot.png

