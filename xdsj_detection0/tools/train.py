import sys
from optparse import OptionParser
from importlib import import_module

sys.path.append('./')

# import yolo
from xdsj_detection.yolo.utils.process_config import process_config

parser = OptionParser()
parser.add_option("-c", "--conf", dest="configure",
                  default="E:/JY_detection/xdsj_detection0/conf/train.cfg",
                  help="configure filename")
(options, args) = parser.parse_args()
if options.configure:
    conf_file = str(options.configure)
else:
    print('please sspecify --conf configure filename')
    exit(0)

# 参数配置
common_params, dataset_params, net_params, solver_params = process_config(conf_file)

# print("%%%%%%%%%%%%%%%%%%%")
# print(common_params, dataset_params, net_params, solver_params)
# print("%%%%%%%%%%%%%%%%%%%")
# print(dataset_params['name'])
# print("$$$$$$$$$$$$$$$$$$$$$4")

# 数据初始化
datasetdot = dataset_params['name'].rindex('.')
datasetmodule, datasetname = dataset_params['name'][:datasetdot], dataset_params['name'][datasetdot + 1:]
datasetmod = import_module(datasetmodule)
datasetobj = getattr(datasetmod, datasetname)
dataset = datasetobj(common_params, dataset_params)

# 网络初始化
netdot = net_params['name'].rindex('.')
netmodule, netname = net_params['name'][:netdot], net_params['name'][netdot + 1:]
netmod = import_module(netmodule)
netobj = getattr(netmod, netname)
net = netobj(common_params, net_params)


# 训练参数初始化
solverdot = solver_params['name'].rindex('.')
solvermodule, solvername = solver_params['name'][:solverdot], solver_params['name'][solverdot + 1:]
solvermod = import_module(solvermodule)
solverobj = getattr(solvermod, solvername)
solver = solverobj(dataset, net, common_params, solver_params)

# dataset = eval(dataset_params['name'])(common_params, net_params)
# net = eval(net_params['name'])(common_params, net_params)
# solver = eval(solver_params['name'])(dataset, net, common_params, solver_params)
solver.solve()
