# Pré-treino de um modelo baseado em Flan-t5 para Pt-Br usando GCP - TPU :brazil:

:uk: English [documentation here](README_en.md)

## Antes de começar

Este código é baseado no trabalho de https://github.com/gsarti. Mais informações e o código original no repositório 
https://github.com/gsarti/t5-flax-gcp.

## Configuração do ambiente

O primeiro passo é criar uma conta no Google Cloud e seguir as instruções deste manual: https://cloud.google.com/sdk/docs/install

Em seguida, crie um disco adicional para o cache dos datasets e dos checkpoints do modelo. O ambiente disponibilizado 
pelo Google com TPU possui apenas 100Gb de disco, o que é consumido rapidamente. Por fim, inicialize a TPU e monte
o disco a ser usado em seguida.

```shell
### Variáveis
export GCP_PROJECT="<YOUR_PROJECT_NAME>"
export GCP_ZONE="<YOUR_REGION>"
export GCP_TPU_NAME="<YOUR_TPU_NAME>"

# >>>>> Variáveis para criação do disco
export GCP_DISK_NAME="<YOUR_DISK_NAME>"
export GCP_DISK_SIZE_GB=1200
export GCP_DISK_TYPE=pd-standard

gcloud beta compute disks create $GCP_DISK_NAME \
    --project=$GCP_PROJECT \
    --type=$GCP_DISK_TYPE \
    --size="${GCP_DISK_SIZE_GB}GB" \
    --zone=$GCP_ZONE

# Criação da TPU VM
gcloud alpha compute tpus tpu-vm create $GCP_TPU_NAME \
    --zone $GCP_ZONE \
    --project $GCP_PROJECT \
    --accelerator-type v3-8 \
    --version v2-alpha \
    # Descomente aqui para usar o disco
    #--data-disk source="projects/${GCP_PROJECT}/zones/${GCP_ZONE}/disks/${GCP_DISK_NAME}"
```

### Dentro da TPU

Para logar na TPU, use o seguinte comando:

`gcloud alpha compute tpus tpu-vm ssh $GCP_TPU_NAME --zone $GCP_ZONE --project $GCP_PROJECT`

O ambiente é um Ubuntu, logo, os seguintes comandos devem funcionar corretamente. O objetivo aqui é identificar o disco,
formatá-lo e efetuar o "mount" em uma pasta para liberação do acesso.

```shell
# Verifique se o disco está visível
lsblk
# Efetue o mount
sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb
sudo mkdir -p data
sudo mount -o discard,defaults /dev/sdb data
sudo chmod a+w data

# Alteração da variável HF_DATASETS_CACHE que indica a localização da para a ser usada pelo Huggingface
echo "export HF_DATASETS_CACHE=data" >> .bashrc
source .bashrc
```

### Instalação do software básico

```shell
sudo apt update
sudo apt-get install -y python3.8-venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
sudo apt-get install git-lfs
git lfs install
git config --global credential.helper store
```

Agora que o ambiente possui o software básico, é possível iniciar o treinamento do modelo. Para isto, clone este
repositório e mude para a pasta onde está o código.

```shell
git clone https://github.com/the-good-fellas/flan-t5-ptbr.git
cd flan-t5-xl-ptbr
pip install -r requirements.txt
```

Faça o login no Huggingface com o comando `huggingface-cli login`.

Veja o arquivo `arg_parser.py` para saber todos os parâmetros possíveis.

## Considerações sobre o tokenizer

Este trabalho utilizou um tokenizer específicamente criado para a lingua portuguesa. Veja o 
repositório `thegoodfellas/tgf-sp-unigram-tokenizer-ptbr` em Huggingface para mais informações sobre a construção.

## Inicio do treinamento

Utilizamos aqui o https://wandb.ai para monitoramento do processo, sendo assim, a variável de ambiente com o token
de acesso precisa estar disponível: `export WANDB_API_KEY=[SUA CHAVE]`.

Para o treinamento usando o dataset Oscar, o seguinte comando foi utilizado:

```shell
nohup python3 -m tgft5 -mode t5 \
--hub_model_id thegoodfellas/tgf-flan-t5-base-ptbr \
--dataset_id oscar \
--dataset_subset unshuffled_deduplicated_pt \
--lm_name thegoodfellas/tgf-flan-t5-base-ptbr \
--wandb_run_id tgf-flan-t5-base-tpuv2-8-oscar \
--save_steps 10_000 \
--warmup_steps 2_000 \
--batch_size 32 \
--preprocessing_num_workers 36 \
--dtype bfloat16 \
--from_pretrained &
```

A descrição completa do modelo está disponível no card do Huggingface `thegoodfellas/tgf-flan-t5-base-ptbr`.

## Agradecimentos

O Google possui um programa de incentivo a pesquisa chamado TPU Research Cloud. Através dele foi possível realizar o
treinamento de diversos modelos a custo zero por 1 mês. Mais informações no site: https://sites.research.google/trc/about/


