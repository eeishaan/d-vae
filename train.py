import os
from argparse import Namespace

import torch
from pytorch_lightning import Trainer

from dvae.constants import BATCH_SIZE, DATA_DIR, MODEL_DIR
from dvae.models.dvae_model import Dvae
from dvae.models.svae_model import Svae

hparams = {
    "batch_size": BATCH_SIZE,
    "dataset_file": os.path.join(DATA_DIR, "final_structures6.txt"),
    "mod": None,
    "beta": 0.005,
    "bidirectional": True,
    "sgp": False,
}
hparams = Namespace(**hparams)

model = Svae(hparams)

trainer = Trainer(
    default_save_path=MODEL_DIR,
    show_progress_bar=True,
    early_stop_callback=None,
    max_nb_epochs=300,
    gpus=1 if torch.cuda.is_available() else 0,
)
trainer.fit(model)
