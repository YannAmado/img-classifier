<!--Template taken from https://github.com/othneildrew/Best-README-Template/blob/master/README.md -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


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
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#how-to-use">How to Use</a></li>
    <li><a href="#demos">Usage Demos</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Making image classifiers is a very troublesome task, having very repetitive steps that are simple but also boring  

Some of the things you always need to do is:
* Data Augment your dataset
* Create your models
* Train and Test your models
* Compare the models trained

Everything is simple enough if you have ever used TensorFlow or Keras before, but tracking all of the possible configurations and repeating everything was often a pain to me.
That's why I decided to make this pipeline to make everything easier!

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

Currently, the only models supported are MobileNetV2 and a simple neural network, if you wish to add more or customize your own just define it on create_model.py and select it on main.py

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- DEMOS -->
## Usage Demos

### Stanford cars

### Simpsons Characters 

### CIFAR-10


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

