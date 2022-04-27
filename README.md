#Attention Loss for Vision Transformer Robustness

##Usage

Run the code on SOMA or Murugan.

Download the project and create a virtual environment.

Make sure you have access to /data/hugo/ImageNet/ on the server you are using.

```
# Download the project
git clone https://github.com/ErestorX/AttentionLoss.git
cd AttentionLoss

# Create a virtual environment
conda create --name AttentionLoss 
conda activate AttentionLoss
pip install -r requirements.txt

# Run the training
./distributed_train.sh [#of GPUs] /data/hugo/ImageNet/ -c output/train/args_t2t_t_base.yaml
./distributed_train.sh [#of GPUs] /data/hugo/ImageNet/ -c output/train/args_attention_t2t.yaml -b 150 --attention_loss_weight 0.1
```

You can change the batch size with the -b parameter.