import torch
import time
import pickle

def test_model(model, test_loader, device):
    correct, total = 0, 0
    model.eval().to(device)
    with torch.no_grad():
        for batch, labels in test_loader:
            y = model(batch.to(device))
            pred = torch.argmax(y.cpu(), dim = 1)
            total += labels.shape[0]
            correct += (pred == labels).sum().item()
    print("Accuracy on the test set is %s %%" % (100 * correct / total))
    return (100 * correct / total)

def get_data(num, loader, loader_size, PATH):
    data = torch.vstack([batch for batch, labels in loader])
    labels = torch.cat([labels for batch, labels in loader])
    # 1000 because there are 1000 instances of each class in the test dataset
    indices = torch.randperm(1000)[:(num // 10)]
    random_data = {k: torch.stack([data[i] for i in range(data.shape[0])
                                   if labels[i] == k])[indices, ...]
                   for k in range(10)}
    for k, v in random_data.items():
        v.requires_grad = True
    if PATH is not None:
        with open(PATH + "/images.pickle", "wb") as f:
            pickle.dump(random_data, f)
    return random_data

def get_data_cifar100(num, loader, loader_size, PATH):
    data = torch.vstack([batch for batch, labels in loader])
    labels = torch.cat([labels for batch, labels in loader])
    # 100 because there are 1000 instances of each class in the test dataset
    indices = torch.randperm(100)[:(num // 100)]
    random_data = {k: torch.stack([data[i] for i in range(data.shape[0])
                                   if labels[i] == k])[indices, ...]
                   for k in range(100)}
    for k, v in random_data.items():
        v.requires_grad = True
    if PATH is not None:
        with open(PATH + "/images.pickle", "wb") as f:
            pickle.dump(random_data, f)
    return random_data

def get_data_mnist(num, loader, loader_size, PATH):
    data = torch.vstack([batch for batch, labels in loader])
    labels = torch.cat([labels for batch, labels in loader])
    # 892 because there are 892 instances of the class 5 in the test dataset
    # (rest are more, than 892)
    indices = torch.randperm(892)[:(num // 10)]
    random_data = {k: torch.stack([data[i] for i in range(data.shape[0])
                                   if labels[i] == k])[indices, ...]
                   for k in range(10)}
    for k, v in random_data.items():
        v.requires_grad = True
    with open(PATH + "/images.pickle", "wb") as f:
        pickle.dump(random_data, f)
    return random_data

def get_grads_per_layer(model_output, model, PATH, trace_time = True, skip_first = False):
    grads = []
    counter = 0
    for y_x in model_output:
        start_time = time.time()
        tmp = []
        for y_x_i in y_x:
            model.zero_grad()
            y_x_i.backward(retain_graph = True)
            tmpp = []
            for param in model.parameters():
                tg = param.grad.clone().detach().cpu()
                tmpp.append(tg)
            tmp.append(tmpp)
        grads.append(tmp)
        counter += 1
        if trace_time:
            print("done with " + str(counter))
            print("-- %s --" % (time.time() - start_time))
    if PATH is not None:
        with open(PATH + "/grads_per_layer.pickle", "wb") as f:
            print("saving...")
            pickle.dump(grads, f)
    return grads

def get_grads_for_params(model_output, model, param_names, PATH, save_name,
                         trace_time = True,
                         skip_first = False):
    grads = []
    counter = 0
    for y_x in model_output:
        start_time = time.time()
        tmp = []
        for y_x_i in y_x:
            model.zero_grad()
            y_x_i.backward(retain_graph = True)
            tmpp = []
            for name, param in model.named_parameters():
                if name in param_names:
                    tg = param.grad.clone().detach().cpu()
                    tmpp.append(tg)
            tmp.append(tmpp)
        grads.append(tmp)
        counter += 1
        if trace_time:
            print("done with " + str(counter))
            print("-- %s --" % (time.time() - start_time))
    if PATH is not None:
        with open(PATH + "/grads_per_params" + save_name + ".pickle", "wb") as f:
            print("saving...")
            pickle.dump(grads, f)
    return grads

# this has to be done at cpu due to memory capacity :()
def get_flattened_summed_grads(grads):
    good_grads = torch.stack([torch.cat([grad.cpu().flatten()
                                         for grad in grads[i][0]])
                              for i in range(len(grads))])
    for cl in range(1, 10):
        print("cl " + str(cl))
        good_grads += torch.stack([torch.cat([grad.cpu().flatten()
                                              for grad in grads[i][cl]])
                                   for i in range(len(grads))])
    return good_grads

def get_flattened_summed_grads_100(grads):
    good_grads = torch.stack([torch.cat([grad.cpu().flatten()
                                         for grad in grads[i][0]])
                              for i in range(len(grads))])
    for cl in range(1, 100):
        print("cl " + str(cl))
        good_grads += torch.stack([torch.cat([grad.cpu().flatten()
                                              for grad in grads[i][cl]])
                                   for i in range(len(grads))])
    return good_grads

def get_flattened_summed_grads_densenet(grads):
    good_grads = torch.stack([torch.cat([grad.cpu().flatten()
                                         for grad in grads[i][j][0]])
                              for i in range(len(grads))
                              for j in range(len(grads[i]))])
    for cl in range(1, 10):
        print("cl " + str(cl))
        good_grads += torch.stack([torch.cat([grad.cpu().flatten()
                                              for grad in grads[i][j][cl]])
                                   for i in range(len(grads))
                                   for j in range(len(grads[i]))])
    return good_grads

def get_flattened_summed_grads_densenet_100(grads):
    good_grads = torch.stack([torch.cat([grad.cpu().flatten()
                                         for grad in grads[i][j][0]])
                              for i in range(len(grads))
                              for j in range(len(grads[i]))])
    for cl in range(1, 100):
        print("cl " + str(cl))
        good_grads += torch.stack([torch.cat([grad.cpu().flatten()
                                              for grad in grads[i][j][cl]])
                                   for i in range(len(grads))
                                   for j in range(len(grads[i]))])
    return good_grads

def calculate_inner_products(data, GRAD_DIM, cl_size = 200, num_cl = 10,
                             weights = None, metric = "",
                             to_norm = False, device = "cpu"):
    if to_norm:
        data = torch.stack([(1 / torch.linalg.norm(vec, ord = 2)) * vec for vec in data])
    inner_prods_inside = {}
    inner_prods_outside = {}
    if metric == "":
        for cl in range(num_cl):
            data_cl = data[cl * cl_size : (cl + 1) * cl_size].to(device)
            data_compl = torch.vstack([data[:cl * cl_size],
                                       data[(cl + 1) * cl_size: ]]).to(device)
            inner_prods_inside[cl] = torch.mm(data_cl, data_cl.T).detach().cpu()
            inner_prods_outside[cl] = torch.mm(data_cl, data_compl.T).detach().cpu()
    elif metric == "full":
        if weights == None:
            print("provide the weight tensor")
            return -1
        for cl in range(num_cl):
            data_cl = data[cl * cl_size : (cl + 1) * cl_size].to(device)
            data_compl = torch.vstack([data[:cl * cl_size],
                                       data[(cl + 1) * cl_size: ]]).to(device)
            weights = weights.to(device)
            first_term = torch.mm(data_cl, weights.unsqueeze(1)).squeeze()
            second_term = torch.mm(weights.unsqueeze(0), data_compl.T).squeeze()
            inner_prods_inside[cl] = torch.outer(first_term, first_term).detach().cpu()
            inner_prods_outside[cl] = torch.outer(first_term, second_term).detach().cpu()
    elif metric == "block":
        if weights == None:
            print("provide the weight tensor")
            return -1
        # in this case weights is vecs for block
        for cl in range(num_cl):
            data_cl = data[cl * cl_size : (cl + 1) * cl_size].to(device)
            data_compl = torch.vstack([data[:cl * cl_size],
                                       data[(cl + 1) * cl_size: ]]).to(device)
            first_terms = []
            second_terms = []
            for blockvec in weights:
                blockvec = torch.cat([blockvec.to(device),
                                      torch.zeros((GRAD_DIM
                                                   - blockvec.shape[0],)).to(device)])
                first_terms.append(torch.mm(data_cl, blockvec.unsqueeze(1)).squeeze())
                second_terms.append(torch.mm(blockvec.unsqueeze(0),
                                             data_compl.T).squeeze())
            first_term = torch.sum(torch.stack(first_terms), dim = 0)
            second_term = torch.sum(torch.stack(second_terms), dim = 0)
            inner_prods_inside[cl] = torch.outer(first_term,
                                                 first_term).detach().cpu()
            inner_prods_outside[cl] = torch.outer(first_term,
                                                  second_term).detach().cpu()

    return inner_prods_inside, inner_prods_outside

def get_gaps_per_coordinates(grads, cl_size = 200, num_cl = 10, device = "cpu"):
    print("mango")
    # shape of [grad_dim]
    A = torch.zeros((grads.shape[1],)).to(device)
    B = torch.zeros((grads.shape[1],)).to(device)
    for cl in range(num_cl):
        grads_cl = grads[cl * cl_size : (cl + 1) * cl_size].to(device)
        grads_compl = torch.vstack([grads[:cl * cl_size],
                                   grads[(cl + 1) * cl_size: ]]).to(device)
        cl_sum = torch.sum(grads_cl, dim = 0)
        compl_sum = torch.sum(grads_compl, dim = 0)
        for i in range(199):
            A += torch.mul(grads_cl[i], (torch.sum(grads_cl[i + 1:], dim = 0)))
        B += torch.mul(cl_sum, compl_sum)
    return -1 * (A / (19900 * 10)) + (B / (cl_size * 1800 * 10))

def get_gaps_per_coordinates_v2(grads, cl_size = 200, num_cl = 10, device = "cpu"):
    A_pos = torch.zeros((grads.shape[1],)).to(device)
    A_neg = torch.zeros((grads.shape[1],)).to(device)
    B_pos = torch.zeros((grads.shape[1],)).to(device)
    B_neg = torch.zeros((grads.shape[1],)).to(device)
    for cl in range(num_cl):
        grads_cl = grads[cl * cl_size : (cl + 1) * cl_size].to(device)
        grads_compl = torch.vstack([grads[:cl * cl_size],
                                   grads[(cl + 1) * cl_size: ]]).to(device)

        grads_cl_poses = grads_cl.clone().to(device)
        grads_cl_negs = grads_cl.clone().to(device)
        grads_compl_poses = grads_compl.clone().to(device)
        grads_compl_negs = grads_compl.clone().to(device)

        grads_cl_poses[grads_cl < 0] = 0
        grads_cl_negs[grads_cl > 0] = 0
        grads_compl_poses[grads_compl < 0] = 0
        grads_compl_negs[grads_compl > 0] = 0

        grads_cl_pos = torch.count_nonzero(grads_cl_poses,
                                           dim = 0)
        grads_cl_neg = torch.count_nonzero(grads_cl_negs,
                                           dim = 0)
        grads_compl_pos = torch.count_nonzero(grads_compl_poses,
                                              dim = 0)
        grads_compl_neg = torch.count_nonzero(grads_compl_negs,
                                              dim = 0)
        A_pos += grads_cl_pos * (grads_cl_pos - 1) / 2
        A_neg += grads_cl_neg * (grads_cl_neg - 1) / 2
        B_pos += grads_compl_pos * (grads_compl_pos - 1) / 2
        B_neg += grads_compl_neg * (grads_compl_neg - 1) / 2
    return (A_pos + B_neg) / (A_neg + B_pos)

def get_gaps_per_coordinates_v3(grads, cl_size = 200, device = "cpu"):
    A = torch.zeros((grads.shape[1],)).to(device)
    B = torch.zeros((grads.shape[1],)).to(device)
    for cl in range(10):
        grads_cl = grads[cl * cl_size : (cl + 1) * cl_size].to(device)
        grads_compl = torch.vstack([grads[:cl * cl_size],
                                   grads[(cl + 1) * cl_size: ]]).to(device)
        cl_sum = torch.sum(grads_cl, dim = 0)
        compl_sum = torch.sum(grads_compl, dim = 0)
        for i in range(199):
            A += torch.mul(grads_cl[i], (torch.sum(grads_cl[i + 1:], dim = 0)))
        B += torch.mul(cl_sum, compl_sum)
    return (-1 * (A / (19900 * 10)) + (B / (cl_size * 1800 * 10))) \
           / (torch.max(grads, dim = 0).values)

def get_gap_for_each_input(grads, cl_size = 200, device = "cpu"):
    C = torch.zeros(2000)
    i = 0
    for cl in range(10):
        grads_cl = grads[cl * cl_size : (cl + 1) * cl_size].to(device)
        grads_compl = torch.vstack([grads[:cl * cl_size],
                                   grads[(cl + 1) * cl_size: ]]).to(device)
        in_sum = torch.sum(grads_cl, dim = 0)
        out_sum = torch.sum(grads_compl, dim = 0)
        for j, grad in enumerate(grads_cl):
            grad_in = torch.dot(grad, in_sum) / cl_size

            grad_out = torch.dot(grad, out_sum) / 1800
            C[i] = grad_in - grad_out
            i += 1

    return C

def get_gap_for_each_input_v2(inps, outps, cl_size = 200, device = "cpu"):
    C = torch.zeros(2000)
    for cl in range(10):
        for i in range(200):
            ins = torch.sum(inps[cl][i]) / 200
            outs = torch.sum(outps[cl][i]) / 1800
            C[cl * 200 + i] = ins - outs

    return C

def get_gap_for_each_input_v2_100(inps, outps, cl_size = 20, device = "cpu"):
    C = torch.zeros(2000)
    for cl in range(100):
        for i in range(cl_size):
            ins = torch.sum(inps[cl][i]) / 20
            outs = torch.sum(outps[cl][i]) / 1800
            C[cl * 20 + i] = outs - ins

    return C


def calculate_gap(inner_prods_inside, inner_prods_outside, cl_size = 200, num_cl = 10):
    equal = sum((torch.sum(v) - torch.trace(v)) / 2 + torch.trace(v)
                for k,v in inner_prods_inside.items()) / (((cl_size * (cl_size - 1)) / 2) * num_cl)
    notequal = sum(torch.sum(v) for k,v in inner_prods_outside.items()) / (cl_size * (2000 - cl_size) * num_cl)
    return equal - notequal, equal, notequal

def sparsify_v1(grads, device = "cpu", to_norm_output = True, threshold = 0):
    gaps = get_gaps_per_coordinates(grads, device = device)
    sparsified_grads = grads.clone().to(device)
    sparsified_grads[:, (gaps < threshold)] = 0
    if to_norm_output:
        sparsified_grads = torch.stack([(1 / torch.linalg.norm(vec, ord = 2)) * vec
                                        for vec in sparsified_grads])
    return sparsified_grads

def sparsify_v2(grads, device = "cpu", to_norm_output = True, threshold = 0):
    gaps = get_gaps_per_coordinates(grads, device = device)
    sparsified_grads = grads.clone().to(device)
    for i in range(sparsified_grads.shape[0]):
        sparsified_grads[i][sparsified_grads[i] < gaps] = 0
    if to_norm_output:
        sparsified_grads = sparsified_grads.cpu()
        sparsified_grads = torch.stack([(1 / torch.linalg.norm(vec, ord = 2)) * vec
                                        for vec in sparsified_grads])
    return sparsified_grads

def sparsify_v3(grads, device = "cpu", cl_size = 200, num_cl = 10,
                to_norm_output = True, threshold = 1,
                pct = None):
    gaps = get_gaps_per_coordinates_v2(grads, cl_size = cl_size, num_cl =
                                       num_cl, device = device)
    sparsified_grads = grads.clone().to(device)
    sparsified_grads[:, (gaps < threshold)] = 0
    if to_norm_output:
        sparsified_grads = torch.stack([(1 / torch.linalg.norm(vec, ord = 2)) * vec
                                        for vec in sparsified_grads])
    return sparsified_grads

def sparsify_v4(grads, device = "cpu", to_norm_output = True, threshold = 1, pct = None):
    gaps = get_gaps_per_coordinates_v3(grads, device = device)
    sparsified_grads = grads.clone().to(device)
    sparsified_grads[:, (gaps < threshold)] = 0
    if to_norm_output:
        sparsified_grads = torch.stack([(1 / torch.linalg.norm(vec, ord = 2)) * vec
                                        for vec in sparsified_grads])
    return sparsified_grads
