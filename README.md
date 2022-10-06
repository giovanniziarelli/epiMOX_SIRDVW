# Optimal control problems on a SIRDVW model for the SARS-CoV-2 on the vaccination campaign: epiMOX Research group

The following code has been developed for optimizing vaccination campaigns on a SIRDVW age-stratified model, conceived with SARS-CoV-2 transmission mechanism and with the Italian Vaccination Campaign. Details of the model can be found in *Articolo*.
This *python* code has been developed from the epiMOX research group at Politecnico di Milano (Milan, Italy) at MOX laboratory for Mathematical Modelling and Scientific Computing.

Code architecture
--
- **epi**: in this folder you will find the main methods and functions for the *Montecarlo Markov Chain* (MCMC) method adopted for calibrating the transmission rate of the model, together with the class SIRDVW implementing the methods for solving the state problem, the adjoint one and for solving the Optimal control problem (method `self.optimal_control()`) thorugh the *Projected Gradient Descent* method;
- **Tests**: this folder contains a template folder **SIRDVW_age_template**, which in turn contains necessary input file (*input.inp*) and parameters file (*parameters_latest.csv*). Those two files are necessary in order to run tests;
- **util**: it contains all the necessary dependancies for applying the code to the Italian 2020-2022 SARS-CoV-2 pandemic;
- *epiMOX_OC_class.py*: this is the main function for solving the optimal control problem. After setting the chosen flags in the *epi/SIRDVW.py* file, the code runs until the tolerance criterion has been fulfilled or the maximum number of iterations has been reached. Once in the major folder, the code can be run as `python3 epiMOX_OC_class.py Tests/[folder name]`;
- *epiMOX_direct_class.py*: this function hs to be run whenever one wants to solve only the direct problem or to calibrate the parameters. After adjusting properly the input and parameters files, it can be compiled and executed as `python3 epiMOX_class.py Tests/[folder name]`;
- *MCMC_postprocess_SIRDVW.py*: this 
- *plot_MCMC_SIRDVW_age.py*: this 
- *plot_single_scenario.py*


Mails and Contacts:
--
- Giovanni Ziarelli - giovanni.ziarelli@polimi.it
- Marco Verani - marco.verani@polimi.it
- Nicola Parolini - nicola.parolini@polimi.it
- Luca Dede' - luca.dede@polimi.it
- Alfio Quarteroni - alfio.quarteroni@polimi.it



