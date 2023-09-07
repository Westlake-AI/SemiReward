# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Ref: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/evaluation.py

import os
from .hook import Hook


class EvaluationHook(Hook):
    """
    Evaluation Hook for validation during training
    """
    
    def after_train_step(self, algorithm):
        if self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm):
            algorithm.print_fn("validating...")
            eval_dict = algorithm.evaluate('eval')
            algorithm.log_dict.update(eval_dict)
            task_type = getattr(algorithm, 'task_type', 'cls')

            # update best metrics
            if task_type == 'cls':
                if algorithm.log_dict['eval/top-1-acc'] > algorithm.best_eval_metric:
                    algorithm.best_eval_metric = algorithm.log_dict['eval/top-1-acc']
                    algorithm.best_it = algorithm.it
            else:
                if algorithm.log_dict['eval/mse'] < algorithm.best_eval_metric:
                    algorithm.best_eval_metric = algorithm.log_dict['eval/mse']
                    algorithm.best_it = algorithm.it
    
    def after_run(self, algorithm):
        task_type = getattr(algorithm, 'task_type', 'cls')

        if not algorithm.args.multiprocessing_distributed or (algorithm.args.multiprocessing_distributed and algorithm.args.rank % algorithm.ngpus_per_node == 0):
            save_path = os.path.join(algorithm.save_dir, algorithm.save_name)
            algorithm.save_model('latest_model.pth', save_path)

        if task_type == 'cls':
            results_dict = {'eval/best_acc': algorithm.best_eval_metric, 'eval/best_it': algorithm.best_it}
            if 'test' in algorithm.loader_dict:
                # load the best model and evaluate on test dataset
                best_model_path = os.path.join(algorithm.args.save_dir, algorithm.args.save_name, 'model_best.pth')
                algorithm.load_model(best_model_path)
                test_dict = algorithm.evaluate('test')
                results_dict['test/best_acc'] = test_dict['test/top-1-acc']
        else:
            results_dict = {'eval/best_mse': algorithm.best_eval_metric, 'eval/best_it': algorithm.best_it}
            if 'test' in algorithm.loader_dict:
                # load the best model and evaluate on test dataset
                best_model_path = os.path.join(algorithm.args.save_dir, algorithm.args.save_name, 'model_best.pth')
                algorithm.load_model(best_model_path)
                test_dict = algorithm.evaluate('test')
                results_dict['test/best_mse'] = test_dict['test/mse']
        algorithm.results_dict = results_dict
