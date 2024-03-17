# Contract-Theoretic Apporach for Robot Allocation
This repo is for contract-theoretic robot allocation project.

## Requiremens
- Python 3.11 or higher

## Running Scripts
1. Create a python virtual envrionment with Python 3.11 and source the virtual environment:
```bash
$ python3.11 -m venv <your-virtual-env-name>
$ source /path-to-venv/bin/activate
```
2. `pip` install the requirements:
```bash
$ pip install -r requirements.txt
```
3. In the project directory, run scripts:
```bash
$ python experiments/run1.py    # run1.py as an example
```

## File Structures
- `config`: Store configuration/parameter files for different testing scenarios.
- `experiments`: Scripts for running the contract-theoretic algorithm and other comparing methods.
- `exp_data`: Store simulation results.
- `utils.py`: Implementations of all algorihtms, including robust allocaiton, contract-based allocation, random-max allocation, and random-sample allocation.
    - Algorithms are selected by `prob_option` in `['robust', 'contract', 'rand_max', 'rand_samp']`.

### Coding Specifications
- We use list to group service robots. For convenience, robot list index equals to robot id.
- We use dictionary to store robot neighbor information. The keys are specified in the code.
- We use numpy array to store:
    - user and robot positions.
    - user and robot types.
    - SP's user type probability.
- We store simulation results in a txt file. Simulation results include:
    - user related information.
    - robot position and control trajectory.
    - locational energy trajectory.