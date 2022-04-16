def build_graph_matrix(file_path, memory_config): ## or availability config
    self.graph = nx.read_gml(topologyFile)
    adjacency_matrix = nx.adjacency_matrix(G)
    ### build feature matrix
    feature_matrix = memory_config 
    




"""
GCN Format: 
In order to use your own data, you have to provide
an N by N adjacency matrix (N is the number of nodes),
an N by D feature matrix (D is the number of features per node), and
an N by E binary label matrix (E is the number of classes).
Have a look at the load_data() function in utils.py for an example.
~~~~~~~~~~~~~~~~~~~~~~~
The input to the inductive model contains:
x, the feature vectors of the labeled training instances,
allx, the feature vectors of both labeled and unlabeled training instances (a superset of x),
graph, a dict in the format {index: [index_of_neighbor_nodes]}.
Let n be the number of both labeled and unlabeled training instances. These n instances should be indexed from 0 to n - 1 in graph with the same order as in allx.
~~~~~~~~~~~~~~~~~~~~~~~
In addition to x, y, allx, and graph as described above, the preprocessed datasets also include:
tx, the feature vectors of the test instances,
test.index, the indices of test instances in graph, for the inductive setting,
ally, the labels for instances in allx.
The indices of test instances in graph for the transductive setting are from #x to #x + #tx - 1, with the same order as in tx.
You can use cPickle.load(open(filename)) to load the numpy/scipy objects x, tx, allx, and graph. test.index is stored as a text file.
"""
def save_graph(grap_info, prefix='ppi', root_dir='data'):
    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]
    class_map = train_data[4]

    print('num of nodes: {}'.format(len(G.node)))
    print('num of features: {}'.format(len(features)))
    print('num of ids: {}'.format(len(id_map)))

    print('data file is read successfully')

    test_index = []
    all_x_index = []

    # ======================
    vertices = G.node

    for key, value in vertices.items():
        if value['test']:
            test_index.append(id_map[key])
        # else:
        #     all_x_index.append(id_map[key])

    for i in range(size):
        if i not in test_index:
            all_x_index.append(i)

    x_index = np.array(np.random.choice(
        len(all_x_index), num_of_x, replace=False))
    x_index = np.array(all_x_index)[x_index]
    x = sp.csr_matrix(features[x_index])
    
    tx = sp.csr_matrix(features[test_index])

    allx = sp.csr_matrix(features[all_x_index])

    H = nx.relabel_nodes(G, id_map)
    graph = nx.to_dict_of_lists(H)

    # root_dir = 'tmp'
    save_object(allx, "{}/ind.{}.allx".format(root_dir, prefix))
    save_object(graph, "{}/ind.{}.graph".format(root_dir, prefix))
    save_object(tx, "{}/ind.{}.tx".format(root_dir, prefix))
    save_object(x, "{}/ind.{}.x".format(root_dir, prefix))

    np.savetxt("{}/ind.{}.test.index".format(root_dir, prefix),
               test_index, delimiter='\n', fmt='%s')
    print('Done')