# AutoQSOFit
This repository is a wrapper over ([PyQSOFit](https://github.com/legolason/PyQSOFit/tree/master)) in which the user can input a catalog that has a redshift and fits file and recieve an output catalog of the corresponding emission line fit. 

In order to do this, we created a notebook `pre_notebook.ipynb` *(title in progress)* that allows the user to read in their catalog and spits out an "input catalog" that is compatible with the fitting program `autoqsofit.py`.

We will also be providing a notebook `post_notebook.ipynb` *(title in progress)* that shows the user how the fitting program works. This notebook will also go into breif detail on what types data analysis can be done for the resulting fits.

# Instructions:
## 1. Installation
To begin working with AutoQSOFit, you must install it.To do so you can use pip. You can perform the following lines in the terminal:

`$ git clone https://github.com/astrohanp/AutoQSOFit.git`

`$ cd AutoQSOFit`

`$ python -m pip install .`

*It may be most useful to create an enviornment designed to work with this program.* 

## 2. Run `Pre-Notebook.ipynb` *(title in progress)*
As mentioned in the overview, this notebook is designed to take an input catalog and output a catalog that is formatted to be compatible with the fitting program `autoqsofit.py`

<b>This notebook is divided into 3 sections:</b>

### 2.1 Catalog Formatting (Required)
This section is the only <b>required section</b> of the notebook, it requires the user to read in their catalog and will output a catalog that is formatted to work with the fitting code. 

An example is done for the user using a subsection of the DEIMOS Catalog (obtained by [Cosmos Team](https://cosmos.astro.caltech.edu/news/65)).

### 2.2 Setting up a Configuration File (Optional)
This section is optional as we have our default configuration file `qsopar.fits` located in the `pyqsofit` folder. If the user wants to make edits to this configuration file, they can run through the arguments found here. This section was primarily written by the PyQSOFit Team.

### 2.3 Creating an Arguments File (Optional) (WIP)
This section is optional as well, it is designed to allow the user to make edits to the fitting arguments without having to make changes to the `autoqsofit.py` code itself. This section is currently a work in progress.

## 3. Use Command Line to Run `autoqsofit.py`
Once you have a formatted catalog, you can run in the terminal:

`$ python autoqsofit.py catalogs/input_catalog.csv`

This will output a catalog `output_catalog.csv` in the `catalogs` folder that holds all the emission line fitted data. This command will also provide individual folders for the galaxies in their catalog found in the `results` folder. If there is an error, it will skip the modeling for said galaxy and output its id in the `results/bad_ids/` folder as a .txt file.    
