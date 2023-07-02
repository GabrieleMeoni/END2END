# END2END
### END2END: End2end training on S-2 RAW data
# About the project 

## 🛰️ END2END: Onboard classification of Thermal Anomalies on Raw Sentinel-2 Data

The END2END project studies the problem of performing on-board satellite thermal anomalies classification on Sentinel-2 [Raw data](https://github.com/ESA-PhiLab/PyRawS#sentinel-2-raw-data).
It provides a complete codebase that enables users to replicate experiments, train and test an [EfficientNet-lite-0](https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html) architecture for the classification of thermal anomalies.


## 🚀👨‍🚀 Easy Reproduction and Quick Adoption

END2END is based on the source code of [MSMatch](https://github.com/gomezzz/MSMatch), which was appropriately retailored for our purposes. Indeed, to train the EfficientNet-lite-0 models both semi-supervised and fully-supervised learning where investigated. 

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About the Project</a>
    <ul>
      <li><a href="#END2END#%EF%B8%8F-end2end-onboard-detection-of-thermal-anomalies-on-raw-sentinel-2-data">END2END: Onboard classification of Thermal Anomalies on Raw Sentinel-2 Data</a></li>
      <li><a href="#easy-reproduction-and-quick-adoption">Easy Reproduction and Quick Adoption</a></li>
    </ul>
    </li>
    <li><a href="#content-of-the-repository">Content of the repository</a></li>
    <li><a href="#installation">Installation</a>
    <ul>
      <li><a href="#create-the-end2end-environment">Create the end2end environment</a></li>
      <li><a href="#set-up-for-the-embedded-hardware-implementation">Set-up for the embedded hardware implementation</a></li>
    </ul>
    </li>
    <li><a href="#end2end-dataset">END2END dataset </a></li>
    <li><a href="#workflow-to-implement-a-trained-model-on-the-edge-device">Workflow to implement a trained model on the edge device </a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## Installation
### Pre-requirements
Before all, clone this repository. We suggest using git from CLI, execute:

``` git clone https://github.com/ESA-PhiLab/PyRawS ```

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

## END2END dataset
The dataset used for END2END is based on [THRawS](https://arxiv.org/abs/2305.11891). `THRawS` (Thermal Hotspots on Raw Sentinel-2 data) is a Sentinel-2 dataset containing raw granules including annotated thermal anomalies. <br>
In particular: 
- to train the model, the bands [`B8A`, `B11`, `B12`] of the various granules of `THRawS` was preprocessed to extract 256x256 patches for thermal anomalies classification. To this aim, `ROBERTO TO COMPLETE`.
- for the [onboard payload prototype](#onboard-payload-prototype), the bands [`B8A`, `B11`, `B12`] of [Sentinel-2 Raw granules](https://github.com/ESA-PhiLab/PyRawS#sentinel-2-raw-granule) are grouped in a [TIF](https://en.wikipedia.org/wiki/TIFF) file, representing an easy-to-read version of the THRawS granules without metadata.

## Workflow to implement a trained model on the edge device 
Once you have trained your model, you can implement the trained model on the edge device as follows: 
<p align="center">
  <img src="resources/images/ncs2WorkFlow.drawio.png" alt="Sublime's custom image"/>
</p>

## Onboard payload prototype
<p align="center">
  <img src="resources/images/onboard_prototype.drawio.png" alt="Sublime's custom image"/>
</p>

In the frame of the END2END project, we aim to implement a mock-up of a full on-board payload processing chain from the sensor to the classification with minimal pre-processing. <br>
The processing chain includes: 

- **Coarse spatial bands registration**: it is a simple but coarse bands registration technique based on the solution described in the[THRawS](https://arxiv.org/abs/2305.11891) paper. The coregistration technique is lightweight and consists of a simple spatial shift that compensates the average [along-track, across-track] displacements between each couple of bands. The coarse band registratio is performed in an onboard processor.

- **Demosaicking**: this steps simply splits an entire [Sentinel-2 Raw granule](https://github.com/ESA-PhiLab/PyRawS#sentinel-2-raw-granule) into 256x256 patches. The `patch engine` is responsible for performing the granule demosaicking and is implemented in the onboard processor

- **AI inference**: the `AI engine` consists of a trained AI EfficientNet-lite-0 model that processes the cropped 256x256 patches. After being compiled through the dedicated [workflow](#workflow-to-implement-a-trained-model-on-the-edge-device), the dedicated `OpenVino IR` file can be deployed on the `Intel NCS2` or `CogniSat` board.

- **Mosaicking**: the results are, then, mosaicked to allineate each prediction to each corresponding patch. 

The onboard payload prototype processing chain can be now profiled to measure the total processing time. 

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
