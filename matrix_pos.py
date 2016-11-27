import numpy as np 




def projection(A, W, R, Iter):
	# learning rate
	upsilon = 0.000002
	# Q start = dot(A, N), normalized
	Q = np.dot(A, R)
	shapeQ = np.shape(Q)
	Q_normalizer = np.reshape(np.repeat(np.sum(Q, 1), shapeQ[1]), shapeQ)
	Q = np.divide(Q, Q_normalizer, out=np.zeros_like(Q), where=Q_normalizer!=0)
	Q_remember = Q
	# only change the cells were sum of row = 0
	# and thus were there was no allignment
	ones = np.ones(shapeQ)
	# only update the rows were the sum = 0 
	# which means that there has not been assigned a distrubution yet 
	# because not alligned 
	update_matrix = np.subtract(ones, np.reshape(np.repeat(np.sum(Q, 1), shapeQ[1]), shapeQ)).astype(int) 
	# construct uniform distribution matrix, scaled by learning rate
	U = np.multiply(np.divide(np.ones(np.shape(Q)), np.shape(Q[0])), upsilon)
	# kappa matrix
	kappa = np.add(np.sum(W, 1), upsilon)
	# start updating
	for i in range(Iter):
		Q = np.divide(np.add(np.dot(W, Q), U), np.reshape(np.repeat(kappa, shapeQ[1]), shapeQ))

	# only want to keep the cells were there was no initial desitribution
	Q = np.add(np.multiply(update_matrix, Q), Q_remember)

	return(Q)

		












if __name__ == "__main__":

	

	# The matrices that are delivered in part1:

	# allignment matrix: A(ij) = 1 if allignment(Vf(i) with Ve(j)) > 0.9
	# size = N x M (N = number of tagalog vertices, M = number of english vertices)
	A = np.array([[1, 0, 0, 1, 0],
				 [0, 1, 0, 1, 0],
				 [0, 0, 0, 0, 1],
				 [1, 1, 0, 0, 0]])

	# neighbourhood matrix: W(ij) = the cosine simularity between Vf(i) and Vf(j)
	# size = N x N (N = number of tagalog vertices)
	W = np.array([[1,0.4,0.002,0.08],
				 [0.4, 1, 0.5, 0.02],
				 [0.002, 0.5, 1, 0.1],
				 [0.08, 0.02, 0.1, 1]])

	# the distribution matrix of the source (engels) R(ij) = the probability of Ve(i) when it's tag is y(j)
	# size = M x T (M = number of english vertices, T = number of possible tags (12 if universal))
	R = [[0,0,0.1,0.2,0,0.5,0.3,0,0,0,0,0],
		 [0,0,0,0,0.8,0,0,0,0.2,0,0,0],
		 [0,0,0,0,0,0,0,0,0,0,0,0],
		 [0.9, 0.01, 0.01, 0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0],
		 [0,0,0,0,0,0,0,0,0,0,0,0],]

	N_iterations = 10	

	Q = projection(A, W, R, N_iterations)

	print(np.sum(Q,1))

