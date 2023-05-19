# LifeCLEF2023-SampleSubmission

Challenge submission template with scripts required to evaluate methods for [FungiCLEF2023](https://huggingface.co/spaces/competitions/FungiCLEF2023) and [SnakeCLEF2023](https://huggingface.co/spaces/competitions/SnakeCLEF2023) challenges.

Instruction to add a custom method:
1. Update `predict.py` script with the custom classification model.
2. Set python dependencies in `requirements.txt` file.
3. Add fine-tuned checkpoint to the directory and replace "DF20-ViT_base_patch16_224-100E.pth" in `Dockerfile`.
4. Run `docker compose up --build` to evaluate the model.

Key files:
* `predict.py` - Prediction script that loads the fine-tuned model, runs inference, and creates `user_submission.csv` file.
* `requirements.txt` - File with python dependencies.
* `Dockerfile` - Dockerfile with instructions to install packages from `requirements.txt` and a command to run `start.sh`. 
* `docker-compose.yml` - Docker compose configuration to build docker image and run container. Contains instruction to mount volume with image data. 
* `start.sh` - Bash script that runs `predict.py` and `evaluate.py`.
* `evaluate.py` - Evaluation script that computes competition metrics.
