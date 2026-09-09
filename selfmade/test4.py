def nb201parser(inp_layers : list):
    # exp list : [1,3,2,3,1,4]
    op_dict = {0 : "none", 1 : "skip_connect", 2 : "nor_conv_1x1", 3 : "nor_conv_3x3", 4 : "avg_pool_3x3"}
    idx = 0
    query = []
    for layer in range(3):
        qp = []
        for i in range(layer + 1):
        # for i in range(layer):
            op = op_dict[inp_layers[idx]]
            print(f"{op}, {i}")
            qp.append(f"{op}~{i}")
            idx += 1
        query.append(qp)
    return query

l = [1,3,2,3,1,4]
print(nb201parser(l))