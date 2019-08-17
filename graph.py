import numpy as np
import networkx as nx
import random
import math
import scipy.io as sio
import scipy

'''
DISCLAIMER:
adapted from the source code of node2vec 
https://github.com/aditya-grover/node2vec
'''

class Graph():
	def __init__(self, file, directed, weighted, p, q, flag=True):
		self.directed = directed
		self.weighted = weighted
		self.p = p
		self.q = q

		self.adj = sio.loadmat(file)['network']
		self.G = self._load_graph()
		self.node_num = self.adj.shape[0]

		if flag:
			labels = sio.loadmat(file)['group']
			if isinstance(labels, scipy.sparse.csc.csc_matrix):
				self.labels = np.array(labels.toarray(), dtype=np.int32)
			else:
				self.labels = np.array(labels, dtype=np.int32)
		else:
			self.labels = None

		degrees = list(np.reshape(np.array(np.sum(self.adj, 1)), (self.adj.shape[0],)))
		self.table = UnigramTable(degrees)
		self.preprocess_transition_probs()

	def _load_graph(self):
		'''
		sparse adj to networkx graph
		'''
		if self.weighted:
		    # node id starts from 0
		    G = nx.from_scipy_sparse_matrix(self.adj, create_using=nx.DiGraph())
		else:
		    G = nx.from_scipy_sparse_matrix(self.adj, create_using=nx.DiGraph())
		    for edge in G.edges():
		        G[edge[0]][edge[1]]['weight'] = 1

		if not self.directed:
		    G = G.to_undirected()

		return G

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
						alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break

		return walk

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		print('Walk iteration:')
		for walk_iter in range(num_walks):
			print(str(walk_iter+1)+'/'+str(num_walks))
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

		return np.array(walks)

	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.directed

		alias_nodes = {}
		for node in G.nodes():
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			alias_nodes[node] = alias_setup(normalized_probs)

		alias_edges = {}
		triads = {}

		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return


class Graph_LINE():
	def __init__(self, file, directed, weighted, base='LINE_2', batch_size=1024, K=1, order=1):
		self.directed = directed
		self.weighted = weighted
		self.base = base
		self.batch_size = batch_size
		self.K = K  # negative sampling
		self.order = order

		self.adj = sio.loadmat(file)['network']
		self.node_num = self.adj.shape[0]

		degrees = list(np.reshape(np.array(np.sum(self.adj, 1)), (self.adj.shape[0],)))
		self.table = UnigramTable(degrees)

		if self.base=='LINE_1':
			self.edge_list = self._adj_to_edgelist(self.adj, order=self.order)
			self.edge_num = self.edge_list.shape[0]
			self.edge_batch = self._batch_generator()
		elif self.base=='LINE_2':
			self.edge_list = self._adj_to_edgelist(self.adj, order=self.order)
			self.edge_num = self.edge_list.shape[0]
			self.edge_batch = self._batch_generator()

	def _batch_generator(self):
		edge_weights = list(self.edge_list[:, 2])
		edge_table = UnigramTable(edge_weights, power=1)
		while True:
			indices = edge_table.sample(self.batch_size)
			pos_pairs = self.edge_list[np.array(indices, np.int32), 0:2]
			neg_pairs = []
			for i in range(self.batch_size):
				for j in range(self.K):
					n_neg = self.table.sample(1)[0]
					while n_neg==pos_pairs[i, 0] or n_neg==pos_pairs[i, 1]:
						n_neg = self.table.sample(1)[0]
					neg_pairs.append([pos_pairs[i, 0], n_neg])
			neg_pairs = np.array(neg_pairs, np.int32)

			yield pos_pairs, neg_pairs

	def _adj_to_edgelist(self, network, order=1):

	    def adj_to_symmetry(network):
	        N = network.shape[0]
	        rows, cols = np.nonzero(network)

	        for i in range(rows.shape[0]):
	            network[cols[i], rows[i]] = network[rows[i], cols[i]]
	        return network

	    if not self.directed:
	    	network = adj_to_symmetry(network)

	    if order>1:
	        tmp = network
	        for i in range(order):
	            network = tmp * network

	    N = network.shape[0]
	    rows, cols = np.nonzero(network)

	    edgelist = []
	    for i in range(rows.shape[0]):
	        edgelist.append([rows[i], cols[i], network[rows[i], cols[i]]])      
	    return np.array(edgelist, dtype=np.int32)


class UnigramTable:
    """
    Using weight list to initialize the drawing 
    """
    def __init__(self, vocab, power=0.75):
        vocab_size = len(vocab)
        norm = sum([math.pow(t, power) for t in vocab]) # Normalizing constant

        table_size = int(1e8) # Length of the unigram table
        table = np.zeros(table_size, dtype=np.uint32)

        print('Filling unigram table')
        p = 0 # Cumulative probability
        i = 0
        for t in range(vocab_size):
            p += float(math.pow(vocab[t], power))/norm
            while i < table_size and float(i) / table_size < p:
                table[i] = t
                i += 1
        self.table = table
        print('Finish filling unigram table')

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
	    q[kk] = K*prob
	    if q[kk] < 1.0:
	        smaller.append(kk)
	    else:
	        larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large
	    q[large] = q[large] + q[small] - 1.0
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
	    return kk
	else:
	    return J[kk]