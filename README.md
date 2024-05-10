# PoreVoyant

https://github.com/mehradans92/PoreVoyant/assets/51170839/55624963-d60e-4cf7-b377-e6502ea6f85d

## Overview
Our AI agent looks up guidelines for designing low band gap MOFs from research papers and suggests a new MOF (likely with a lower band gap). It then checks validity of the new SMILES candidate and predicts band gap with uncertainty estimation using a surrogate ensemble of fine-tuned MOFormers.
![image](https://github.com/mehradans92/PoreVoyant/assets/51170839/a48498f8-1a50-45e7-9a97-e37c001c88ed)

## Results

## Experiment 1. Using all the tools
Model's suggestion on MOFs with lower band gap with the domain knowledge extracted from scientific literature.
![image](https://github.com/mehradans92/PoreVoyant/assets/51170839/52e45d8b-f9bf-447c-9016-3b396456d5bd)

## Experiment 2. Removing domain knowledge tool (Tool 1)
Model's suggestion on MOFs with lower band gap with the GPT-4 knowledge only. We observe many chemically infeasible suggestions by the model, which confirms the importance of domain knowledge that the agent.

https://github.com/mehradans92/PoreVoyant/assets/51170839/b1de5885-73f9-45f1-87ac-10bde9fc77e7




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
