generate-plots:
    #!/usr/bin/env bash
    cd experiment_data
    python plot.py 100 "selfplay score00" latent_loss
    python plot.py 10 "selfplay score00" terminal_nodes
    python plot_carchess.py "carchess/selfplay score00.csv"
