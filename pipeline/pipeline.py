import kfp
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model
from typing import Optional

def preprocess_op(input_path: str, output_x_path: str, output_y_path: str, output_preprocessing_state_path: str):
    return dsl.ContainerOp(
        name='Preprocess',
        image='us-east1-docker.pkg.dev/cmlproj2/ner-repo/preprocess:latest',
        arguments=[
            '--input-path', input_path,
            '--output-x-path', output_x_path,
            '--output-y-path', output_y_path,
            '--output-preprocessing-state-path', output_preprocessing_state_path,
        ],
        file_outputs={
            'output_x_path': output_x_path,
            'output_y_path': output_y_path,
            'output_preprocessing_state_path': output_preprocessing_state_path
        }
    )

def train_op(input_x_path: str, input_y_path: str, input_job_dir: str, input_tags: int, input_words: int, input_dropout: float, output_model_path: str):
    return dsl.ContainerOp(
        name='Train',
        image='us-east1-docker.pkg.dev/cmlproj2/ner-repo/train:latest',
        arguments=[
            '--input-x-path', input_x_path,
            '--input-y-path', input_y_path,
            '--input-job-dir', input_job_dir,
            '--input-tags', input_tags,
            '--input-words', input_words,
            '--input-dropout', input_dropout,
            '--output-model-path', output_model_path,
            '--output-model-path-file', output_model_path
        ],
        file_outputs={
            'output_model_path': output_model_path
        }
    )

def deploy_op(model_dir: str, project: str, region: str, endpoint_name: str, model_display_name: str):
    return dsl.ContainerOp(
        name='Deploy',
        image='us-east1-docker.pkg.dev/cmlproj2/ner-repo/deploy:latest',
        arguments=[
            '--model-dir', model_dir,
            '--project', project,
            '--region', region,
            '--endpoint-name', endpoint_name,
            '--model-display-name', model_display_name,
        ]
    )

@dsl.pipeline(
    name='NER Pipeline Vertex AI',
    description='An example pipeline for NER using Vertex AI Pipelines.'
)
def ner_pipeline(
    input_path: str = 'gs://my-kubeflow-bucket-1745602479/input/ner.csv',
    preprocessing_output_prefix: str = 'gs://my-kubeflow-bucket-1745602479/preprocess',
    training_output_prefix: str = 'gs://my-kubeflow-bucket-1745602479/training',
    project: str = 'cmlproj2',
    region: str = 'us-east1',
    dropout_rate: float = 0.5,
    input_tags: int = 17,
    input_words: int = 10000,
):
    preprocess = preprocess_op(
        input_path=input_path,
        output_x_path=f'{preprocessing_output_prefix}/X.pkl',
        output_y_path=f'{preprocessing_output_prefix}/y.pkl',
        output_preprocessing_state_path=preprocessing_output_prefix,
    )

    train = train_op(
        input_x_path=preprocess.output,
        input_y_path=preprocess.output,
        input_job_dir=training_output_prefix,
        input_tags=input_tags,
        input_words=input_words,
        input_dropout=dropout_rate,
        output_model_path=training_output_prefix,
    )

    deploy = deploy_op(
        model_dir=training_output_prefix,
        project=project,
        region=region,
        endpoint_name='ner-endpoint-{{workflow.uid}}',
        model_display_name='ner-model-{{workflow.uid}}',
    )

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(ner_pipeline, 'ner_pipeline_vertex_ai.json')
