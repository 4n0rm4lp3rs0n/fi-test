# def nb201parser(inp_layers : list):
#     # exp list : [1,3,2,3,1,4]
#     op_dict = {0 : "none", 1 : "skip_connect", 2 : "nor_conv_1x1", 3 : "nor_conv_3x3", 4 : "avg_pool_3x3"}
#     idx = 0
#     query = []
#     for layer in range(3):
#         qp = []
#         for i in range(layer + 1):
#             op = op_dict[inp_layers[idx]]
#             # print(f"{op}, {i}")
#             qp.append(f"{op}~{i}")
#             idx += 1
#         query.append(qp)

#     subq = [[""]]
    
#     return query

def nb201parser(genes : list):
    # exp list : [1,3,2,3,1,4]
    op_dict = {0 : "none", 1 : "skip_connect", 2 : "nor_conv_1x1", 3 : "nor_conv_3x3", 4 : "avg_pool_3x3"}
    assert len(genes) == 6
    return (f"|{op_dict[genes[0]]}~0|+"
            f"|{op_dict[genes[1]]}~0|{op_dict[genes[2]]}~1|+"
            f"|{op_dict[genes[3]]}~0|{op_dict[genes[4]]}~1|{op_dict[genes[5]]}~2|"
    )

l = [1,3,2,3,1,4]
print(nb201parser(l))