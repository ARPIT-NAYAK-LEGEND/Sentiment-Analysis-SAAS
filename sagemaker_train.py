import os
import shutil

os.environ["AWS_DEFAULT_REGION"] = "ap-south-1"
os.environ["AWS_REGION"] = "ap-south-1"

from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig


def start_training():
    source_dir = 'sagemaker_source'

    # Clean source dir
    if os.path.exists(source_dir):
        shutil.rmtree(source_dir)

    os.makedirs(os.path.join(source_dir, 'training'))

    # Copy training code
    for item in os.listdir('training'):
        s = os.path.join('training', item)
        d = os.path.join(source_dir, 'training', item)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)

    # Requirements
    with open(os.path.join(source_dir, 'requirements.txt'), 'w') as f:
        f.write("transformers\n")
        f.write("torchaudio\n")
        f.write("torchvision\n")
        f.write("imageio-ffmpeg\n")
        f.write("opencv-python-headless\n")
        f.write("scipy\n")
        f.write("pandas\n")
        f.write("scikit-learn\n")
        f.write("numpy==1.26.4\n")

    # TensorBoard
    tensorboard_config = TensorBoardOutputConfig(
        s3_output_path='s3://sentiment-analyzer-arpit/tensorboard-logs',
        container_local_output_path='/opt/ml/output/tensorboard'
    )

    estimator = PyTorch(
        entry_point='training/train.py',
        source_dir=source_dir,
        role='arn:aws:iam::219967434603:role/sentiment-analyzer-execution',
        framework_version='2.5.1',
        py_version='py311',

        instance_count=1,
        instance_type='ml.g5.xlarge',

        max_run=88000,

        hyperparameters={
            'batch_size': 32,
            'epochs': 25
        },

        tensorboard_config=tensorboard_config,
        disable_profiler=True
    )

    estimator.fit({
        "train": "s3://sentiment-analyzer-arpit/dataset/train",
        "validation": "s3://sentiment-analyzer-arpit/dataset/dev",
        "test": "s3://sentiment-analyzer-arpit/dataset/test"
    })


if __name__ == "__main__":
    start_training()
