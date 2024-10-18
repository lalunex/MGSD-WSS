'''
Author: Yidan Liu 1334252492@qq.com
Date: 2024-02-06 12:20:54
LastEditors: Yidan Liu 1334252492@qq.com
LastEditTime: 2024-10-18 18:45:52
FilePath: /add_noise/parameter_tuning.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from recbole.trainer import HyperTuning
from run_mgsd import main, get_parameter_dict
from utils.parser import parse_args


def customizer_objective_function(config_dict=None, config_file_list=None):
    mp_args = dict(
        world_size=4,
        nproc=4,
        offset=0,
        # ip='ssh.rde-ws.lanrui-ai.com',
        # port=32335
    )
    args = parse_args()
    dic = vars(args)
    dic.update(config_dict)
    # dic.update(mp_args)
    parameter_dict = get_parameter_dict(dic)

    result = main(
        model=dic['model_name'], dataset=dic['dataset'],
        config_dict=parameter_dict, config_file_list=config_file_list, saved=True
    )

    return result


hp = HyperTuning(objective_function=customizer_objective_function,
                 algo='random', early_stop=10, max_evals=200,
                 params_file='config/tuning.hyper',
                 fixed_config_file_list=['./config/text.yml'])

if __name__ == '__main__':
    # run
    hp.run()
    # export result to the file
    hp.export_result(output_file=r'./para_tune/mlk_bert4rec_alltrancl_hyper_example.result')
    # print best parameters
    print('best params: ', hp.best_params)
    # print best result
    print('best result: ')
    print(hp.params2result[hp.params2str(hp.best_params)])
