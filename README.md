Zeolites are essential materials with applications in catalysis, separation, and adsorption. However, their
traditional synthesis relies on organic structure-directing agents (OSDAs), which are expensive and
environmentally unfriendly. Seed-assisted zeolite synthesis provides a green and economical alternative by
reducing or eliminating the need for OSDAs. Despite its potential, identifying optimal synthesis conditions
remains a challenge due to the high-dimensional chemical space involved.

To address this challenge, your task is to develop a machine learning model that predicts the success of seed-
assisted zeolite synthesis experiments using a provided dataset. The dataset contains 385 historical records of

seed-assisted zeolite synthesis experiments conducted in a trial-and-error manner. Each experiment is
categorized into one of two classes:
• Class "0": Failed experiments resulting in amorphous, mixed, dense, or layered phases.
• Class "1": Successful experiments resulting in a pure zeolite phase.
The dataset includes the following parameters:
1. Seed properties:
o Seed amount (normalized to SiO2 weight = 1)
o Seed framework density (FD) in T/Å3
o Seed Si/Al molar ratio (measured using ICP-AES)

Click here to
download

2. Gel composition:
o SiO2 (normalized to 1)
o NaOH/SiO2 molar ratio
o B2O3/SiO2 molar ratio
o H2O/SiO2 molar ratio
o OTMAC/SiO2 molar ratio (SDA)
3. Crystallization conditions:
o Crystallization temperature (°C)
o Crystallization time (days)

Streamlit : https://d8dmpo3gdtunubwa9bgigr.streamlit.app/
