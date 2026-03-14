<!--
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
-->

# Torch TEM








### Model Training

With the repo as working directory, we modified the code to align the GPU running environments, train a model by running
```sh
python run_main.py
```
Model parameters are specified in ```parameters.py```.

### Model Analysis

After training a model, analyse a model and plot analysis results by running
```sh
python test.py
```
You will need to specify the correct model run in ```test.py```.


