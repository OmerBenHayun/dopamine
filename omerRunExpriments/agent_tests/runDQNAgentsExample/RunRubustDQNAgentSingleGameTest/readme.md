# how to run
## init the virtual envirument (only the first time)
follow the main turial from the base directory of this repo 
after that activate the virtual env via 
```
source dopamine-venv/bin/activate
```
and then execute:
```
pip install tf-nightly-gpu==2.5.0.dev20201028
```
## detremite the gpu fraction to use
edit the file
```
dopamine/discrete_domains/run_experiment.py
```
comment out line 
```
    #config.gpu_options.allow_growth = True 
```
and add after it the following
```
    config.gpu_options.allow_growth = False
    config.gpu_options.per_process_gpu_memory_fraction = 0.5

```
when `per_process_gpu_memory_fraction` is the amount of GPU to use for the expiriment

for more information one can look at the following [link](https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory)
## run an expiriment
activete virtual env
```
source dopamine-venv/bin/activate
```
run the code
```
cd dopamine
export PYTHONPATH=$PYTHONPATH:$PWD
python -m omerRunExpriments.agent_tests.runDQNAgentsExample.RunRubustDQNAgentSingleGameTest.runExpiriments
```
