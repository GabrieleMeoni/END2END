# END2END
End2end training on S2 raw data
## About the project 
This project investigates the training on Sentinel-2 RAW data. 

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
If you also want to use the files in the `nc2` directory, you also need to quantize your model to infer your model on the NCS2.
To this aim, you can use the script `quantize.py` located in `ncs2\quantization`, which implements `INT8` quantization. 
If you use "Windows", proceed as follows: 

 *  install the `Build Tools for Visual Studio 2022`, including `Desktop development with C++` from [Visual Studio 2022](https://visualstudio.microsoft.com/it/downloads/). Theoretically, the use of other compilers should be possible, despite we did not manage to make it run without it. 
 * When you launch `quantize.py`, please, make use to specify the path including the "cl.exe" file through the flag "--cl_exe_path" (i.e., ```python quantize.py --cl_exe_path path_to_dir_including_cl_exe other flags```).

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

* Gabriele Meoni - gabriele.meoni at esa.int
* Roberto Del Prete - roberto.delprete at ext.esa.int
