# CS 7649 Final Project

## Setup Training Environment
To setup your training environment, run the following commands.

```
git clone https://github.com/alanhesu/evacuation_MARL
cd evacuation_MARL
conda env create environment.yml
conda activate cs7649_proj
cd environment
```

## Training
To train your environment, set any relevant constants in the `evacuation/constants.py` and then run the following commands.
```
cd evacuation
python3 train_evacuation.py
```


## Visualizing
To visualize your environment, set the mode variable in the top of `evacuation/visualize_evacuation.py` to either `"human"` or `"none"` whether you want to show the pygame visualization or just run simulation respectively.
```
cd evacuation
python3 visualize_evacuation.py
```

We have included two working policies to visualize, one called `evac_policy.zip` which was our baseline policy, and one called `evac_policy_multi_exit.zip` which requirest setting `NUM_EXITS = 2` in `evacuation/constants.py` and `load = evac_policy_multi_exit.zip` in `evacuation/visualize_evacuation.py` to run a simulation with two exits. Both should work without any other modifications. 

To finish the current episode, press the `n` key. To stop all episodes, press the `q` key. This is useful if a single human or robot is not exiting and you want to continue the simulation without letting them exit. See notes if you have issues with the keyboard library.

Notes:
- If you have issues with using the `keyboard` library, you can comment the troublesome parts and it should run normally without the functionality of skipping episodes or exiting. `Ctrl+C` should still work normally to exit the program if you want to end it another way.

