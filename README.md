# Rapid Artificial Intelligence-based Multi-physic CFD Simulator
Repository for the novel CFD approach called AI4CFD and no traniing of the network is needed. More detailed implementation and possible applications can be found in as follow: 
- **"Using AI libraries for incompressible Computational Fluid Dynamics"** 
https://www.imperial.ac.uk/people/boyang.chen16/document/10322/AI_for_CFD/?AI_for_CFD.pdf
- **"Using Graph Neural Networks for Incompressible CFD on Unstructured Meshes"** \
https://www.imperial.ac.uk/people/boyang.chen16
- **"Solving the Dsicretised Diffusion Equation in Neutron Transport using Neural Networks"**
https://drive.google.com/file/d/18ssthBBOi4zuY8oLRFKO9rXbjUg9ictl/view?usp=share_link
- **"Solving the Dsicrete Ordinate Equations in Neutron Transport using Neural Networks"**
https://drive.google.com/file/d/1dm4cXBBX-fcFS33Zi7IIFjTU7qywDKWK/view?usp=sharing

## Applications of Simulator:

- **Anisotropic Resistivity Tomography**: 
- **2D Kolmogorov Flow**: 
- **Flow Past a Bluff Body**: 
- **Urban Modelling**: 
- **Unstuctured Mesh**: 

## Requirements

To install requirements:

```setup
 $ conda env create -f environment.yml 
 $ conda activate ten
 $ python -m ipykernel install --user --name=python3 (optional)
```

Finally, start Jupyter with a local machine:

```start 
 $ jupyter notebook
```

Or, start Jupyter with a remote server:
```start 
 $ jupyter notebook --no-browser --port=<number>
```


