import os
from argparse import Namespace

import torch
from pytorch_lightning import Trainer

from constants import BATCH_SIZE, DATA_DIR, MODEL_DIR
from svae_model import Svae


dataset_file = os.path.join(DATA_DIR, 'final_structures6.txt')

model = Svae(BATCH_SIZE, dataset_file)

trainer = Trainer(
    default_save_path=MODEL_DIR,
    show_progress_bar=True,
    early_stop_callback=None,
    max_nb_epochs=300,
    gpus=1 if torch.cuda.is_available() else 0)

trainer.fit(model)
