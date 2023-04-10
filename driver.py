"""
TODO: use LightningCLI to automatically map model hyperparameters
"""
import argparse
from runner import CI_Runner


def cli_main():
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('-s', '--seed', default=42, type=int)      
    parser.add_argument('--overfit-batch', action=argparse.BooleanOptionalAction)      
    parser.add_argument('--logs', default=False)      
    parser.add_argument('--patience', default=3, type=int)      
    # defult is no experiment
    parser.add_argument('--logdir', default="lightning_logs")      
    parser.add_argument('-name', '--experiment-name', default="")      
    parser.add_argument('-v', '--experiment-version', default=None)      
    # 
    parser.add_argument('--eval-period-epochs', default=1, type=int)      
    parser.add_argument('--max-train-epochs', default=100, type=int)      
    #
    parser.add_argument('--disable-checkpoints', action=argparse.BooleanOptionalAction)
    # model
    parser.add_argument('-lr', '--learning-rate', default=5e-5, type=float)      
    parser.add_argument('--freeze-embedder', action=argparse.BooleanOptionalAction)      
    return parser


if __name__ == "__main__":
    # args = cli_main().parse_args()
    args = argparse.Namespace(
        seed=42, overfit_batch=None, 
        logs=False, patience=100, 
        logdir='lightning_logs', 
        experiment_name='CI_token', experiment_version='freeze_bert_5e5_patient_context_bert', 
        eval_period_epochs=1, max_train_epochs=200, 
        disable_checkpoints=True, learning_rate=5e-05, 
        freeze_embedder=True
    )

    CI_Runner(args).run()

