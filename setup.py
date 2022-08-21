from setuptools import setup, find_packages

setup(
    name='omega',
    version='0.0.1',
    packages=find_packages(),

    author='Boris Yangel',
    long_description="A number of agents (PPO, MuZero) with a \
        Perceiver-based NN architecture that can be trained \
            to achieve goals in nethack/minihack environments.",
    url="https://github.com/hr0nix/omega",

    install_requires=[
        'numpy~=1.22.4',
        'flax~=0.5.1',
        'optax~=0.1.2',
        'rlax~=0.1.2',
        'attrs~=21.2.0',
        'tensorflow~=2.9.1',
        'gym~=0.24.1',
        'nle~=0.8.1',
        'minihack~=0.1.3',
        'dataclasses~=0.6',
        'PyYAML~=5.4.1',
        'tqdm~=4.62.0',
        'ray~=1.13.0',
        'pytest~=7.1.2',
        'wandb~=0.12.14',
        'array2gif~=1.0.4',
        'pygraphviz~=1.9',
    ],

    extras_require={
        # CPU-only
        # pip install omega[cpu]
        'cpu': ['jaxlib==0.3.10', 'jax[cpu]==0.3.13'],

        # GPU-only
        # pip install omega[cuda]
        'cuda': ["jaxlib[cuda]==0.3.10", "jax[cuda]==0.3.13"],
    },
)
