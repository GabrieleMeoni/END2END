# END2END
### END2END: End2end training on S2 raw data
# About the project 

## 🛰️ END2END: Onboard Detection of Thermal Anomalies from Raw Sentinel-2 Data

END2END is a library for onboard detection of thermal anomalies from raw Sentinel-2 data. The code base for this library is derived from the paper "MSMatch: Semi-Supervised Multispectral Scene Classification with Few Labels" (arXiv:2103.10368) by Pablo Gómez and Gabriele Meoni.

## 🌍🧑‍🔬 The Challenge of Supervised Learning in Remote Sensing

Supervised learning techniques are at the center of many tasks in remote sensing. However, these methods often require large amounts of labeled data for training. Labeling data can be tedious, expensive, and requires expert knowledge. END2END provides a solution to this problem by using semi-supervised learning techniques.

## 📈💻 MSMatch: Semi-Supervised Learning for Scene Classification

The MSMatch algorithm, which is implemented in END2END, is the first semi-supervised learning approach that is competitive with supervised methods on scene classification on the EuroSAT and UC Merced Land Use benchmark datasets. With just five labeled examples per class, the neural network can achieve up to 95.86% accuracy on the EuroSAT multispectral dataset, outperforming previous methods by up to 19.76%. On the UC Merced Land Use dataset, END2END outperforms previous works by up to 5.59% and reaches 90.71% accuracy with just five labeled examples.

## 🚀👨‍🚀 Easy Reproduction and Quick Adoption

END2END provides the source code of MSMatch online to enable easy reproduction and quick adoption. It hendles well to multispectral raw data and should enable various applications that are currently infeasible.

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About the Project</a></li>
    <li><a href="#content-of-the-repository">Content of the repository</a></li>
    <li><a href="#installation">Installation</a>
    <ul>
      <li><a href="#create-the-end2end-environment">Create the end2end environment</a></li>
      <li><a href="#set-up-for-the-embedded-hardware-implementation">Set-up for the embedded hardware implementation</a></li>
    </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## Installation
### Pre-requirements
Before all, clone this repository. We suggest using git from CLI, execute:

``` git clone https://gitlab.esa.int/Alix.DeBeusscher/PyRawS.git ```

### Create the end2end environment
To install the environment, we suggest to use [anaconda]("https://www.anaconda.com/products/distribution"). You can create a dedicated conda environment by using the `environment.yml` file by running the following command from the main directory: 

``` conda env create -f environment.yml ```

To activate your environment, please execute:

``` conda activate end2end ```


### Set-up for the embedded hardware implementation
We have implemented a prototype on the [Intel Neural Compute Stick 2 (NCS2)](https://www.intel.com/content/www/us/en/developer/articles/tool/neural-compute-stick.html). <br>
If you also want to use the files in the `ncs2` directory, you need to install [OPENVINOv 2022.1](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html). We proceeded as follow to install successfully the software needed to interface the NCS2 device on Windows: 

1. Select version 2022.1 and download the offline installer.
2. Install openvino through the downloaded offline installer. By default, it should produce the a directory called "Intel".
3. Install openvino in the `end2end` conda environment through: 

```pip install openvino==2022.1````

4. Install numpy==1.23.4 in the conda environment. 

5. Copy the content of `Intel\openvino_2022.1.0.643\runtime\bin\intel64\Release` into your 
   `$CONDA_PATH\envs\end2end\Lib\site-packages\openvino\libs`

6. Export `$CONDA_PATH\envs\end2end\Lib\site-packages\openvino\libs` to PATH 

Now, you should be able to work with the NCS2 device. 

## Workflow to implement a trained model on the edge device 
Once you have trained your model by using [MSMAtch](https://github.com/gomezzz/MSMatch), you can implement the trained model on the edge dvice as follows: 
<p align="center">
  <img src="resources/images/ncs2WorkFlow.drawio.png" alt="Sublime's custom image"/>
</p>


## Contributing
The ```end2end``` project is open to contributions. To discuss new ideas and applications, please, reach us via email (please, refer to [Contact](#contact)). To report a bug or request a new feature, please, open an [issue](https://github.com/GabrieleMeoni/END2END/issues) to report a bug or to request a new feature. 

If you want to contribute, please proceed as follow:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/NewFeature`)
3. Commit your Changes (`git commit -m 'Create NewFeature'`)
4. Push to the Branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## License
Distributed under the GPL-3.0 License.

## Contact
Created by the European Space Agency $\Phi$-[lab](https://phi.esa.int/).

* Gabriele Meoni - Currently with TU Delft: G.Meoni@tudelft.nl
* Roberto Del Prete - roberto.delprete at ext.esa.int
