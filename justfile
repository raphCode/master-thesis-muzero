generate-plots:
    #!/usr/bin/env bash
    cd experiment_data
    python plot.py 100 "selfplay score00" latent_loss
