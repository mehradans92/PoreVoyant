# PoreVoyant
A Chemistry-informed AI Agent for Metal-organic Framework Property Prediction


https://github.com/mehradans92/PoreVoyant/assets/51170839/55624963-d60e-4cf7-b377-e6502ea6f85d

## Overview
Our AI agent looks up guidelines for designing low band gap MOFs from research papers and suggests a new MOF (likely with a lower band gap). It then checks validity of the new SMILES candidate and predicts band gap with uncertainty estimation using a surrogate ensemble of fine-tuned MOFormers.
![TOC](https://github.com/mehradans92/PoreVoyant/assets/51170839/f9fd068f-0a8c-4cff-8bac-a7053786ec15)


## Results

## Experiment 1. Using all the tools 
Model's suggestion on MOFs with lower band gap with the domain knowledge extracted from scientific literature.
![Agent_run](https://github.com/mehradans92/PoreVoyant/assets/51170839/851505d8-d1af-4a61-b478-e16ddbc02f64)


## Experiment 2. Removing domain knowledge tool (Tool 1)
Model's suggestion on MOFs with lower band gap with the GPT-4 knowledge only. We observe many chemically infeasible suggestions by the model, which confirms the importance of domain knowledge that the agent.
![Cd O- C(=O)c1ccc2c(c1)ccc(c2)C(=O) O- _unguided](https://github.com/mehradans92/PoreVoyant/assets/51170839/7d97b8f7-08bf-494c-9e07-f53b600d341b)






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
