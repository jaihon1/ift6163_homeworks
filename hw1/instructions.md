## Homework 1
Starting with the default configuration state of config.yaml, execute the following commands to run different parts of the assignment.

### Part 1

#### Run first environment (BC with Ant)
```
python run_hw1.py env.expert_policy_file=./ift6163/policies/experts/Ant.pkl env.expert_data=../../../ift6163/expert_data/expert_data_Ant-v2.pkl env.env_name=Ant-v2 env.max_episode_length=1000 alg.n_iter=1 alg.batch_size=5000 alg.eval_batch_size=5000 alg.data_ratio=1.0 alg.do_dagger=false
```

#### Run second environment (BC with Hopper)
```
python run_hw1.py env.expert_policy_file=./ift6163/policies/experts/Hopper.pkl env.expert_data=../../../ift6163/expert_data/expert_data_Hopper-v2.pkl env.env_name=Hopper-v2 env.max_episode_length=1000 alg.n_iter=1 alg.batch_size=5000 alg.eval_batch_size=5000 alg.data_ratio=1.0 alg.do_dagger=false
```

#### Run first environment with varying hyperparameter data_ratio
The following command runs the first environment with data_ratio=0.1

For my experiments, I ran the first environment with data_ratio = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

Please modify the data_ratio value to get the desired result.

```
python run_hw1.py env.expert_policy_file=./ift6163/policies/experts/Ant.pkl env.expert_data=../../../ift6163/expert_data/expert_data_Ant-v2.pkl env.env_name=Ant-v2 env.max_episode_length=1000 alg.n_iter=1 alg.batch_size=5000 alg.eval_batch_size=5000 alg.data_ratio=0.1 alg.do_dagger=false
```


### Part 2

#### Run first environment (Dagger with Ant)
```
python run_hw1.py env.expert_policy_file=./ift6163/policies/experts/Ant.pkl env.expert_data=../../../ift6163/expert_data/expert_data_Ant-v2.pkl env.env_name=Ant-v2 env.max_episode_length=1000 alg.n_iter=10 alg.batch_size=5000 alg.eval_batch_size=5000 alg.data_ratio=1.0 alg.do_dagger=true
```

#### Run second environment (Dagger with Hopper)

```
python run_hw1.py env.expert_policy_file=./ift6163/policies/experts/Hopper.pkl env.expert_data=../../../ift6163/expert_data/expert_data_Hopper-v2.pkl env.env_name=Hopper-v2 env.max_episode_length=1000 alg.n_iter=10 alg.batch_size=5000 alg.eval_batch_size=5000 alg.data_ratio=1.0 alg.do_dagger=true
```