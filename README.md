# PoreVoyant

https://github.com/mehradans92/PoreVoyant/assets/51170839/55624963-d60e-4cf7-b377-e6502ea6f85d

## Overview

![image](https://github.com/mehradans92/PoreVoyant/assets/51170839/a48498f8-1a50-45e7-9a97-e37c001c88ed)

## Results
## Experiemnt 1. Using all the tools
Model's suggestion on MOFs with lower band gap with the domain knowledge extracted from scientifc literature.
![image](https://github.com/mehradans92/PoreVoyant/assets/51170839/52e45d8b-f9bf-447c-9016-3b396456d5bd)

## Experiemnt 2. Removing domain knowledge tool (Tool 1)
Model's suggestion on MOFs with lower band gap with the GPT-4 knowledge only. We observe manny chemically infeasible suggestion by the model, which confirms the importance of domain knowledge that the agent.
![Cd O- C(=O)c1ccc2c(c1)ccc(c2)C(=O) O- _unguided](https://github.com/mehradans92/PoreVoyant/assets/51170839/ec13e9d1-4da2-431e-b99d-fbfba5346b65)


## Working model

Please run the Jupyter notebook `transformer/workflow.ipynb` for the model and the `transformer/post_process.ipynb` for results visualization.

## Installation

```bash
conda create -n pore python=3.11.9
conda activate pore

pip install -r requirements.txt
conda install -c pytorch/label/nightly -c nvidia faiss-gpu=1.8.0
```

## Contributors
1. Mehrad Ansari (Acceleration Consortium)
2. Sartaaj Khan (University of Toronto)
3. Mahyar Rajabi (University of Toronto)
4. Seyed Mohamad Moosavi (University of Toronto)
5. Amro Aswad (University of Toronto)
