# Optimal control problems on a SIRDVW model for the SARS-CoV-2 on the vaccination campaign: epiMOX Research group

The following code has been developed for optimizing vaccination campaigns on a SIRDVW age-stratified model, conceived with SARS-CoV-2 transmission mechanism and with the Italian Vaccination Campaign. Details of the model can be found in .
This *python* code has been developed from the epiMOX research group at Politecnico di Milano (Milan, Italy) at MOX laboratory for Mathematical Modelling and Scientific Computing.

Code architecture
--
- **epi**: in this folder you will find the main methods and functions for the *Montecarlo Markov Chain* (MCMC) method adopted for calibrating the transmission rate of the model, together with the class SIRDVW implementing the methods for solving the state problem, the adjoint one and for solving the Optimal control problem (method `optimal_control`)
- **Tests**
- **util**
- *epiMOX_OC_class.py*
- *epiMOX_direct_class.py*
- *plot_MCMC_SIRDVW_age.py*
- *plot_single_scenario.py*
- *MCMC_postprocess_SIRDVW.py*

Mails and Contacts:
--
- Giovanni Ziarelli - giovanni.ziarelli@polimi.it
- Marco Verani - marco.verani@polimi.it
- Nicola Parolini - nicola.parolini@polimi.it
- Luca Dede' - luca.dede@polimi.it
- Alfio Quarteroni - alfio.quarteroni@polimi.it



