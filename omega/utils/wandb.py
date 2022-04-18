import os
import wandb.util


def get_wandb_id(path):
    if os.path.exists(path):
        with open(path) as id_file:
            id = id_file.read()
    else:
        id = wandb.util.generate_id()
        with open(path, 'w') as id_file:
            id_file.write(id)

    return id
