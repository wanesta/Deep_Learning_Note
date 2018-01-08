# 逻辑斯蒂回归和最大熵模型

逻辑斯蒂回归是统计学的最经典分类方法

## 二项逻辑斯蒂回归模型

二项逻辑斯蒂回归模型是如下的条件概率分布： $$ P(Y=1|x)= \frac{\exp(w_x+b)}{1+\exp(w_x+b)}$$ $$ P(Y=0|x)=\frac{1}{1+\exp(w*x+b)}$$

- 注意：P(Y=1|x)模型也经常写成$$ h_{\Theta}(x)=\frac{1}{1+\exp(-\Theta^{T} * x)} $$ 事件的几率（odds）是指该事件发生的概率与该事件不发生的概率的比值。 如果事件发生的概率是p，那么该事件的几率是 $ \frac{p}{1-p} $ 该事件的对数几率（log odds）或logit函数是： $ logit(p)=\log(\frac{p}{1-p})$
- 逻辑回归的对数几率是：$$ \log(\frac{P(Y=1|x)}{1-P(Y=1|x)}) = w_{x} $$ 意义：在逻辑斯蒂回归模型中，输出Y=1的对数几率是输入x的线性函数。或者说，输出Y=1的对数几率是由属于x的线性函数表示的模型，即逻辑斯蒂回归模型。（这里需要再理解下） 感知机只通过决策函数（w⋅x）的符号来判断属于哪一类。逻辑斯蒂回归需要再进一步，它要找到分类概率P(Y=1)与输入向量x的直接关系，再通过比较概率值来判断类别。 令决策函数（w⋅x）输出值等于概率值比值取对数，即： $$ \log\frac{p}{1-p} = w * x \Rightarrow p=\frac{\exp(w_x+b)}{1+\exp(w_x+b)} $$ 逻辑斯蒂回归模型的定义式P(Y=1|x)中可以将线性函数w⋅x转换为概率，这时，线性函数的值越接近正无穷，概率值就越接近1；线性函数的值越接近负无穷，概率值就接近0.

## 模型参数估计(极大似然估计)

应用极大似然法进行参数估计，从而获得逻辑斯蒂回归模型。 设：P(Y=1|x)=π(x),P(Y=0|x)=1−π(x) 似然函数为
$$ \prod_{i=1}^{N}=[\pi (x_{i})]^{y_{i}}[1 - x_{i}]^{1-y_{i}} $$
上式连乘符号内的两项中，每个样本都只会取到两项中的某一项。若该样本的实际标签
$ y_{i}=1 $ 取样本计算为1的概率值为$ \pi(x_{i}) $ 若该样本的实际标签 $ y^{i} = 0 $ ，取样本计算的为0的概率值 $ 1-\pi(x^{i}) $ 。
对数似然函数为:
$$ L(\omega ) = \sum_{i=1}^{N}[y_{i}\log\pi(x_{i})+(1-y_{i})\log(1-\pi(x_{i}))]  $$
$$ =\sum_{i=1}^{N}[y_{i}\log\pi(x_{i}) + \log(1 - \pi(x_{i})) - y_{i}\log(1 - \pi(x_{i}))] $$
$$ = \sum_{i=1}^{N}[y_{i}\log\pi(x_{i})+\log(1-\pi(x_{i}))-y_{i}\log(1-\pi(x_{i}))] $$
$$ = \sum_{i=1}^{N}[y_{i}\log\frac{\pi(x_{i})}{1-\pi(x_{i})}+\log(1-\pi(x_{i}))] $$
$$ = \sum_{i=1}^{N}[y_{i}(w \cdot x_{i}) + \log(\frac{1}{1+\exp(w \cdot x_{i})})] $$
$$ =\sum_{i=1}^{N}[y_{i}(w \cdot x_{i})-\log(1+\exp(w \cdot x_{i})))] $$
对上式中的L(w)求极大值，得到w的估计值。
问题转化成以对数似然函数为目标函数的无约束最优化问题，通常采用梯度下降法以及拟牛顿法求解w。 假设w的极大估计值是
$ \hat{w} $，那么学到的逻辑斯蒂回归模型为：
$$ P(y=1|x)=\frac{\exp(\hat{w} \cdot x)}{1+\exp(\hat{w} \cdot x)} $$
$$ P(Y=0|x)=\frac{1}{1+\exp(\hat{w}\cdot x)} $$

### 交叉熵损失函数的求导

逻辑回归的另一种理解是以交叉熵作为损失函数的目标最优化。交叉熵损失函数可以从上文最大似然推导出来。

- 交叉熵损失函数为:
$$ y^{(i)}\log(h_{\Theta}(x^{(i)}))+(1-y^{i})\log(1-h_{\Theta}(x^{i})) $$
- 则可以得到目标函数为：
$$ J(\Theta)=-\frac{1}{m}\sum_{i=1}^{N}y^{(i)}\log(h_{\Theta}(x^{(i)}))+(1-y^{i})\log(1-h_{\Theta}(x^{i}))$$ $$ =-\frac{1}{m}\sum_{i=1}^{N}y^{(i)}\Theta^{T}x^{i}-\log(1+e^{\Theta^{T}x^{i}}) $$
计算J(θ)对第j个参数分量$\Theta_{j}$求偏导:
$$ \frac{\partial J(\Theta)}{\partial \Theta_{j}}=\frac{\partial }{\partial \Theta_{j}}(\frac{1}{m}\sum_{i=1}^{m}[\log(1+e^{\Theta^{T}x^{i}})-y^{i}\Theta^{T}x^{i}]) $$
$$ =\frac{1}{m}\sum_{i=1}^{m}[\frac{\partial}{\partial \Theta_{j}}\log(1+e^{\Theta^{T}x^{i}})-\frac{\partial}{\partial \Theta_{j}}(y^{i}\Theta^{T}x^{i})] $$
$$ =\frac{1}{m}\sum_{i=1}^{m}(\frac{x_{j}^{i}e^{\Theta ^{T}x^{i}}}{1+e^{\Theta^{T}x^{i}}}-y^{i}x_{j}^{i}) $$
$$ =\frac{1}{m}\sum_{i=1}^{m}(h_{\Theta}(x^{i})-y^{i})x_{j}^{i} $$

Logistic回归与多重线性回归实际上有很多相同之处，最大的区别就在于它们的因变量不同，其他的基本都差不多。 正是因为如此，这两种回归可以归于同一个家族，即广义线性模型（generalizedlinear model）。 这一家族中的模型形式基本上都差不多，不同的就是因变量不同。

- 如果是连续的，就是多重线性回归；
- 如果是二项分布，就是Logistic回归；
- 如果是Poisson分布，就是Poisson回归；
- 如果是负二项分布，就是负二项回归。

Logistic回归的因变量可以是二分类的，也可以是多分类的，但是二分类的更为常用，也更加容易解释。所以实际中最常用的就是二分类的Logistic回归。

Regression问题的常规步骤为：

- 1.寻找h函数（即hypothesis）；
- 2.构造J函数（损失函数）；
- 3.想办法使得J函数最小并求得回归参数（θ）

# 构造预测函数

$$ g(x)=\frac{1}{1-e^{-z}} $$

Logistic回归虽然名字里带"回归"，但是它实际上是一种分类方法，主要用于两分类问题（即输出只有两种，分别代表两个类别），所以利用了Logistic函数（或称为Sigmoid函数），函数形式为：

$$ g(x)=\frac{1}{1-e^{-z}} $$

对于线性边界的情况，边界形式如下：

$$ \Theta_{0}+\Theta_{1}x_{1}+\Theta_{2}x_{2},...,\Theta_{i}x_{i}=\sum_{i=1}^{n}\Theta_{i}x_{i}=\Theta^{T}x $$

## 构造预测函数为：

$$ h_{\Theta}(x)=g(\Theta^{T}x)=\frac{1}{1+e^{-\Theta^{T}x}} $$

函数$ h_{\Theta}(x) $的值有特殊的含义，它表示结果取1的概率，因此对于输入x分类结果为类别1和类别0的概率分别为：

$$ P(y=1 | x;\Theta)=h_{\Theta}(x) $$

$$ P(y=0 | x;\Theta)=1-h_{\Theta}(x) $$

## 极大似然估计

$$ P(y|x;\Theta)=(h_{\Theta}(x))^{y} * (1-h_{\Theta}(x))^{1-y} $$

因为样本数据(m个)独立，所以它们的联合分布可以表示为各边际分布的乘积,取似然函数为:

$$ L(\Theta)=\prod^{m}_{i=1}P(y^{i}|x^{i};\Theta) $$

$$ L(\Theta )=\prod_{i=1}^{m}(h_{\Theta }(x_{i }))^{y^{i } } $$

$$ L(\Theta)=\prod^{m}_{i=1}(h_{\Theta}(x^{(i)}))^{y^{(i)}} * (1-h_{\Theta}(x^{(i)}))^{1-y^{(i)}} $$

取对数似然函数

$$ l(\Theta)=\log(L(\Theta))=\sum_{i=1}^{m}\log((h_{\Theta}(x^{(i)}))^{y^{(i)}})+\log((1-h_{\Theta}(x^{(i)}))^{1-y^{(i)}}) $$

即

$$ l(\Theta)=\log(L(\Theta))=\sum_{i=1}^{m}(y^{i}\log(h_{\Theta}(x^{(i)})+(1-y^{(i)})\log(1-h_{\Theta}(x^{(i)}))) $$

求导:
$$ \frac{\partial l(\Theta)}{\partial \Theta} = \sum_{i=1}^{m}(\frac{y^{i}}{h_{\Theta}(x^{i})} \frac{\partial h_{\Theta}(x_{i})}{\partial \Theta}+\frac{1-y^{i}}{1-h_{\Theta}(x_{i})}\frac{\partial h_{\Theta}(x^{i})}{\partial \Theta}(-1)) $$
$$ =\sum_{i=1}^{m}( \frac{\partial h_{\Theta}(x^{i})}{\partial \Theta} (\frac{y^{i}}{h_{\Theta}(x^{i})}-\frac{1-y^{i}}{1-h_{\Theta}(x^{i})} )) $$ $$ =\sum_{i=1}^{m}(\frac{\partial h_{\Theta}(x^{i})}{\partial \Theta} (\frac{y^{i}(1-h_{\Theta}(x^{i}))+ (y^{i}-1)h_{\Theta}x^(i)}{h_{\Theta}(x^{i})(1-h_{\Theta}(x^{i}))}) $$ $$ =\sum_{i=1}^{m}(\frac{\partial h_{\Theta}(x^{i})}{\partial \Theta}(\frac{y^{i}-h_{\Theta}(x^{i})}{h_{\Theta}(x^{i})(1-h_{\Theta}(x^{i}))})) $$ 其中： $$ \frac{\partial h_{\Theta}(x^{i})}{\partial\Theta} = h_{\Theta}(x^{i})(1-h_{\Theta}(x^{i}))x^{j} $$ 化简: $$ \bigtriangledown_{\Theta}l(\Theta)=\frac{\partial l(\Theta)}{\partial(\Theta)}=\sum_{i=1}^{m}(y^{i}-h_{\Theta}(x^{i}))x^{j} $$

### Factorization Machine & Logistic Regression

两个模型联系起来要从一种角度来看LR:它是对log odd ratio 做线性回归$ \overrightarrow{w}\cdot\overrightarrow{x}=\log\frac{Pr(y = 1 |x ,w)}{Pr(y = -1 | x, w)} $ ,所以LR模型可以说是这个有概率意义对sigmoid形式： $$ Pr(y = 1 | x , w)=\frac{1}{1 + e^{-wx}} $$ 线性模型本身不能区描述特征之间对交互因素，这就需要在生成训练样本时候通过特征工程里面加上特征组合的方式来引入特征之间的交互因素。这个从理论上可以看作是加上多项式的内核，但是这样会把模型参数从n变到$ \frac{n^{2}}{2} $所以一般是人工来完成预先指定引入那一部分特征的组合。专门实现了DSL来指定特征转换处理和组合配置，在ETL中来做这些工作。

在广告系统中的点击预估模型，特征组合是用户特征和context特征的组合来获取。例如： user本身有 feature n 纬 | ad 有 feature m 纬 | context 有 feature l 纬 那么组合的feature_max 则 m _n_ l 但是这 m _n 或 m_ n * l 个特征中绝大多数的会是0，即没有出现对应取值的样本。 从这个角度来看，LR是组合特征的角度区描述单特征之间的交互组合，而FM实际上是从模型的角度来做的。即FM中特征交互的是模型参数的一部分，比如： 只考虑二阶的特征组合： $$ f(x) = wx + \frac{1}{2}x^{T}V^{T}Vx - diag(x^{T}x)^{T}dia(V^{T}V) $$ $$ Pr(y=1|x)=\frac{1}{1+e^{-f(x)}} $$ V 矩阵描述的是每个特征的K纬的latent factor ，二阶的特征间的交互通过 $ \frac{1}{2} x^{T}V^{T}V_{x} $来完成。与LR中国年的人工指定特征组合的方案相比，这里回引入同一类特征的组合。比如：user-user ad-ad。理论上这类组合会在模型learning的过程中被学习出其对应 $ Pr(y = -1|x) $ 与 $ Pr(y = -1|x) $ 非常接近，而最终v^{T}v趋近于0。不过实际应用中，因为数据往往是倾斜的，或者样本集的分布不足以描述实际的分布，往往并不能保证同类特征的组合结果参数趋近于0。 解决同类特征不去做组合这个问题的一个方案是指定哪类特征与哪类特征的embedding矩阵去做交互，比如指定类似的LR的user与ad特征的交互。 另外，当有多个不同范畴当特征交互相交互当时候可以引入不同当embedding。比如user-ad，context-ad，这里边ad与两类特征交互，可以分别安排两个C矩阵来处理。这可以理解为ad对user时对k纬对latent factor与ad对context对k纬对latent factor在每一个维度上意义不同。这个

- LR的损失函数 使用logloss而不使用平房损失：

  - 线性回归 $$ J(\Theta)=\frac{1}{N}\sum_{i=1}^{N}(d_{i}-w^{T}x_{i})^{2} $$ $J(\Theta)$为凸有解析解，
  - 逻辑回归 $${J}'(\Theta)=\frac{1}{N}\sum_{i=1}^{N}(d_{i}-\frac{1}{1+e^{-w^{T}x_{i}}})^{2}$$ ${J}'(\Theta)$非凸，无解析解，不易优化，易局部最小。 而使用logloss可得凸函数能找到最优解。
  - logloss的解释： 将$P(Y=1|x)=\pi(x)$,$P(Y=0|x)=\pi(x)$中$\pi(x)$看成概率分布$ \pi(x)=\frac{1}{1+e^{-wx}} $，其中有未知参数w如何看成概率？ 如果看成概率最大似然法取使样本出现概率最大的w 设$ P(Y=1|x)=\pi(x)$,$P(Y=0|x)=\pi(x) $ 似然函数:
$$ \prod_{i=1}^{N}=[\pi(x_{i})]^{y_{i}}[1-x_{i}]^{1-y_{i}} $$
    对数似然:
$$ L(w)= \sum_{i=1}^{N}[y_{i}\log\pi(x_{i})+(1-y_{i})\log(1-\pi(x_{i}))] $$
