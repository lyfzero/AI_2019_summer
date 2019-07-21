class LogisticRegression():
    def __init__(self, scorefunc=sigmoid, optimizor=BGD, fit_intercept=True, penalty="L2", gamma=0):
        '''
        参数：
        scorefuc：激活函数
        optimizor：梯度优化函数
        fit_intercept：是否在X前面加一列1，将b内置到theta
        penalty:正则化，可选"L1","L2"
        gamma:正则化项的系数
        '''
        self.optimizor = optimizor
        self.scorefunc = scorefunc
        self.theta = None
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.gamma = gamma
        
    def fit(self, X, y, epoch=500, lr=0.01, tol=1e-7):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        theta, loss_data= self.optimizor(X, y, self.scorefunc, epoch, lr, tol, self.penalty, self.gamma)
        self.theta = theta
        return loss_data
        
    def predict(self, X):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
            
        prob = self.scorefunc(np.dot(X, self.theta))
        predictions = get_predictions(prob)
        return predictions
    

def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def get_predictions(prob):
    prob[prob>0.5] = 1
    prob[prob<0.5] = 0
    return prob.astype(int)

def cross_entropy_loss(prob, y):
    return np.mean(-y * np.log(prob) - (1 - y) * np.log(1 - prob))


def get_gradient(predictions, X, y):
    return np.dot(X.T, predictions - y) / X.shape[0]
 
def BGD(X, y, scorefunc, epoch=500, lr=0.01, tol=1e-7, penalty="L2", gamma=0):
    '''
    用梯度下降拟合线性回归
    返回参数和代价
    '''
    m, n = X.shape
    loss_data = []
    theta = np.random.randn(n)
    for i in range(epoch):
        y_prob = scorefunc(np.dot(X, theta))
        loss = cross_entropy_loss(y_prob, y)
        order = 2 if penalty == "L2" else 1
        loss += 0.5 * gamma * np.linalg.norm(theta, ord=order) ** 2 / m
        loss_data.append(loss)
        if loss < tol:
            return theta, loss_data
        gradient_theta = get_gradient(y_prob, X, y)
        theta -= lr * gradient_theta
        if i%100==99:
            print('epoch %d: loss:%f' % (i, loss))
    return theta, loss_data


def SGD(X, y, scorefunc, epoch=500, lr=0.01, tol=1e-7, penalty="L2", gamma=0, batch_num=5):
    m,n = X.shape
    loss_data = []
    theta = np.random.randn(n)   
    indexs = np.arange(m)
    for i in range(epoch):
        np.random.shuffle(indexs)
        indices = np.array_split(indexs, batch_num) 
        for index in indices:
            X_index = X[index]
            y_index = y[index]
            y_prob = scorefunc(np.dot(X_index, theta))
            loss = cross_entropy_loss(y_prob, y_index)
            order = 2 if penalty == "L2" else 1
            loss += 0.5 * gamma * np.linalg.norm(theta, ord=order) ** 2 / m
            loss_data.append(loss)
            if loss < tol:
                return theta, loss_data
            gradient_theta = get_gradient(y_prob, X_index, y_index)
            theta -= lr * gradient_theta
        if i%100==99:
                print('epoch %d: loss:%f' % (i, loss))
    return theta, loss_data

def momentum(X, y, scorefunc,epoch=500, lr=0.01, tol=1e-7, penalty="L2", gamma=0, beta=0.9):
    m, n = X.shape
    loss_data = []
    theta = np.random.randn(n)
    v = np.zeros(n)
    for i in range(epoch):
        y_prob = scorefunc(np.dot(X, theta))
        loss = cross_entropy_loss(y_prob, y)
        order = 2 if penalty == "L2" else 1
        loss += 0.5 * gamma * np.linalg.norm(theta, ord=order) ** 2 / m
        loss_data.append(loss)
        if loss < tol:
            return theta, loss_data
        gradient_theta = get_gradient(y_prob, X, y)        
        v = v * beta + gradient_theta
        theta -= lr * v
        if i%100==99:
            print('epoch %d: loss:%f' % (i, loss))
    return theta, loss_data

def nesterov(X, y, scorefunc, epoch=500, lr=0.01,tol=1e-7, penalty="L2", gamma=0, beta=0.9):
    m,n = X.shape
    loss_data = []
    theta = np.random.randn(n)
    v = np.zeros(n)
    for i in range(epoch):
        y_prob = scorefunc(np.dot(X, theta - lr * beta * v))
        loss = cross_entropy_loss(y_prob, y)
        order = 2 if penalty == "L2" else 1
        loss += 0.5 * gamma * np.linalg.norm(theta, ord=order) ** 2 / m
        loss_data.append(loss)
        if loss < tol:
            return theta, loss_data
        gradient_theta = get_gradient(y_prob, X, y) 
        v = v * beta + gradient_theta
        theta -= v * lr
        if i%100==99:
            print('epoch %d: loss:%f' % (i, loss))
    return theta, loss_data

def adaGrad(X, y, scorefunc, epoch=500, lr=0.01, tol=1e-7, penalty="L2", gamma=0):
    m, n = X.shape
    loss_data = []
    theta = np.random.randn(n)
    G = np.zeros(n)
    eps = 1e-7
    for i in range(epoch):
        y_prob = scorefunc(np.dot(X, theta))
        loss = cross_entropy_loss(y_prob, y)
        order = 2 if penalty == "L2" else 1
        loss += 0.5 * gamma * np.linalg.norm(theta, ord=order) ** 2 / m
        loss_data.append(loss)
        if loss < tol:
            return theta, loss_data
        gradient_theta = get_gradient(y_prob, X, y)
        G += gradient_theta ** 2
        theta -= lr/(np.sqrt(G + eps)) * gradient_theta
        if i%100==99:
            print('epoch %d: loss:%f' % (i, loss))
    return theta, loss_data

def RMSprop():
    pass

def Adam():
    pass