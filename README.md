[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23262986.svg)](https://doi.org/10.5281/zenodo.23262986)

## A Python script for generating figures about population structure of *Notorynchus cepedianus* and their growth curves

This repository contains the Python code required to replicate the statistical analyses and generate Figures 2 and 3 (length-frequency distributions and non-linear growth curves) from the manuscript: "***Wild population size structure and female-specific growth of the broadnose sevengill shark, _Notorynchus cepedianus_, in the Southwestern Atlantic***". To estimate growth parameters from tag-recapture data, three non-linear candidate models were fitted: Gompertz, Logistic, and Fabens' reparameterization of the von Bertalanffy Growth Function (VBGF).


🔬 Overview

We investigated the broadnose sevengill shark [*_Notorynchus cepedianus_*](https://www.fishbase.se/summary/Notorynchus-cepedianus) population structure and growth rates in coastal waters of the Southwestern Atlantic (~34.6ºS to ~49.4ºS). Our findings reveal distinct regional variation in size structure across the latitudinal range.

- The filtered recapture dataset used for growth modeling is provided in the manuscript, whereas the broader nominal dataset is available upon reasonable request, subject to data-sharing agreements.

- To get a complete description of each variable and performed analyses take a look at the [manuscript](https://doi.org/10.1016/j.fishres.2026.107896).


📁 Repository Structure

   - `shark_growth_Ncepedianus_code.py`: contains the code to perform the statistical analyses and get the outputs.
   
   - `shark_growth_Ncepedianus_functions.py`: Auxiliary Python functions used throughout the analysis.


🛠️ Reproducibility

To replicate this research locally, you will need Python 3.10+ installed.

Clone the repository: git clone https://github.com/jcompaire/FishGrowth_Ncepedianus.git

Open the project in Python in your preferred IDE (Spyder was used for this project).
      

👤 Authors

This work was conducted by [Jesus C. Compaire](https://www.researchgate.net/profile/Jesus-Compaire), [Gastón A. Trobbiani](https://www.researchgate.net/profile/Gaston-Trobbiani), Federico Más, Juan Martín Cuevas, Carolina Pantano, Martín Laporta, Lucas Albornoz, Cristian Lagger,  and [Alejo J. Irigoyen](https://www.researchgate.net/profile/Alejo-Irigoyen). Jesus C. Compaire wrote and updates the Python script.

This work was supported in part by [Proyecto Arrecife](https://www.proyectoarrecife.com.ar/es/pez/gatopardo). 


## References

[Compaire et al. (2026)](https://doi.org/10.1016/j.fishres.2026.107896)

```
Compaire, J.C., Trobbiani, G.A., Mas, F., Cuevas, J.M., Pantano, C., 
Laporta, M., Albornoz, L., Lagger, C., Irigoyen, A.J. (2026). Wild population 
size structure and female-specific growth of the broadnose sevengill shark, 
Notorynchus cepedianus, in the Southwestern Atlantic. 
Fisheries Research, XX(X), XXX. https://doi.org/10.1016/j.fishres.2026.107896
```

# _Notorynchus cepedianus_
<img align="left" width="50%" src="https://www.proyectoarrecife.com.ar/sites/default/files/2021-01/29.359_Irigoyen%20Alejo_20160926_Alejo%20Irigoyen.jpg"/>

Credit: 

<img align="center" width="10%" src="https://www.proyectoarrecife.com.ar/sites/default/files/logo_proyecto_3.png"/>
