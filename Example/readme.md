# 1.原理

发现写关于非负矩阵的博文还是蛮多的，还是以自己的角度总结一下自己的最近看的若干东西以及对非负矩阵分解有用的一些资料链接。

NMF，全称为non-negative matrix factorization，中文呢为“非负矩阵分解”。非负矩阵，就是矩阵中的每个元素都是非负的。将非负矩阵V分解为两个非负矩阵W和H的乘，叫做非负矩阵分解。

# 2.应用概述

对比了一圈，NMF可以应用的领域很广，源于其对事物的局部特性有很好的解释。在众多应用中，NMF能被用于发现数据库中的图像特征，便于快速自动识别应用；能够发现文档的语义相关度，用于信息自动索引和提取；能够在DNA阵列分析中识别基因等等。我们将对此作一些大致的描述。但是最有效的就是图像处理领域，是图像处理的数据降维和特征提取的一种有效方法。

##2.1 图像分析

NMF最成功的一类应用是在图像的分析和处理领域。图像本身包含大量的数据，计算机一般将图像的信息按照矩阵的形式进行存放，针对图像的识别、分析和处理也是在矩阵的基础上进行的。这些特点使得NMF方法能很好地与图像分析处理相结合。人们已经利用NMF算法，对卫星发回的图像进行处理，以自动辨别太空中的垃圾碎片；使用NMF算法对天文望远镜拍摄到的图像进行分析，有助于天文学家识别星体；美国还尝试在机场安装由NMF算法驱动的识别系统，根据事先输入计算机的恐怖分子的特征图像库来自动识别进出机场的可疑恐怖分子。

学术界中：（1）NMF首次被Lee教授用于处理人脸识别。（2）LNMF被宋教授后面提出用于提取人脸子空间，将人脸图像在特征空间上进行投影，得到投影系数作为人脸识别的特征向量，用来进行人脸识别。一定程度上提高了识别率。（3）GNMF被杨教授提出，该算法是基于gamma分布的NMF进行构建特征子空间，采用最小距离分类对ORL人脸库部分图像进行识别。

对于人脸识别，其中以LNMF最为有效突出，比普通的NMF高效且精度高。

##2.2 文本聚类/数据挖掘

文本在人类日常接触的信息中占有很大分量，为了更快更精确地从大量的文本数据中取得所需要的信息，针对文本信息处理的研究一直没有停止过。文本数据不光信息量大，而且一般是无结构的。此外，典型的文本数据通常以矩阵的形式被计算机处理，此时的数据矩阵具有高维稀疏的特征，因此，对大规模文本信息进行处理分析的另一个障碍便是如何削减原始数据的维数。NMF算法正是解决这方面难题的一种新手段。NMF在挖掘用户所需数据和进行文本聚类研究中都有着成功的应用例子。由于NMF算法在处理文本数据方面的高效性，著名的商业数据库软件Oracle在其第10版中专门利用NMF算法来进行文本特征的提取和分类。为什么NMF对于文本信息提取得很好呢？原因在于智能文本处理的核心问题是以一种能捕获语义或相关信息的方式来表示文本，但是传统的常用分析方法仅仅是对词进行统计，而不考虑其他的信息。而NMF不同，它往往能达到表示信息的局部之间相关关系的效果，从而获得更好的处理结果。

##2.3 语音处理
语音的自动识别一直是计算机科学家努力的方向，也是未来智能应用实现的基础技术。语音同样包含大量的数据信息，识别语音的过程也是对这些信息处理的过程。NMF算法在这方面也为我们提供了一种新方法，在已有的应用中，NMF算法成功实现了有效的语音特征提取，并且由于NMF算法的快速性，对实现机器的实时语音识别有着促进意义。也有使用NMF方法进行音乐分析的应用。复调音乐的识别是个很困难的问题，三菱研究所和MIT(麻省理工学院)的科学家合作，利用NMF从演奏中的复调音乐中识别出各个调子，并将它们分别记录下来。实验结果表明，这种采用NMF算法的方法不光简单，而且无须基于知识库。

##2.4 机器人控制
如何快速准确地让机器人识别周围的物体对于机器人研究具有重要的意义，因为这是机器人能迅速作出相应反应和动作的基础。机器人通过传感器获得周围环境的图像信息，这些图像信息也是以矩阵的形式存储的。已经有研究人员采用NMF算法实现了机器人对周围对象的快速识别，根据现有的研究资料显示，识别的准确率达到了80%以上。

##2.5 生物医学工程和化学工程
生物医学和化学研究中，也常常需要借助计算机来分析处理试验的数据，往往一些烦杂的数据会耗费研究人员的过多精力。NMF算法也为这些数据的处理提供了一种新的高效快速的途径。科学家将NMF方法用于处理核医学中的电子发射过程的动态连续图像，有效地从这些动态图像中提取所需要的特征。NMF还可以应用到遗传学和药物发现中。因为NMF的分解不出现负值，因此采用NMF分析基因DNA的分子序列可使分析结果更加可靠。同样，用NMF来选择药物成分还可以获得最有效的且负作用最小的新药物。

#3.实践

##3.1. NMF-based 推荐算法
在例如Netflix或MovieLens这样的推荐系统中，有用户和电影两个集合。给出每个用户对部分电影的打分，希望预测该用户对其他没看过电影的打分值，这样可以根据打分值为其做出推荐。用户和电影的关系，可以用一个矩阵来表示，每一列表示用户，每一行表示电影，每个元素的值表示用户对已经看过的电影的打分。下面来简单介绍一下基于NMF的推荐算法。

在python当中有一个包叫做sklearn，专门用来做机器学习，各种大神的实现算法都在里面。本文使用

`from sklearn.decomposition import NMF`

**数据**
电影的名称，使用10个电影作为例子：
```
item = [
    '希特勒回来了', '死侍', '房间', '龙虾', '大空头',
    '极盗者', '裁缝', '八恶人', '实习生', '间谍之桥',
]
```
用户名称，使用15个用户作为例子：
```
user = ['五柳君', '帕格尼六', '木村静香', 'WTF', 'airyyouth',
        '橙子c', '秋月白', 'clavin_kong', 'olit', 'You_某人',
        '凛冬将至', 'Rusty', '噢！你看！', 'Aron', 'ErDong Chen']
```
用户评分矩阵：
```
RATE_MATRIX = np.array(
    [[5, 5, 3, 0, 5, 5, 4, 3, 2, 1, 4, 1, 3, 4, 5],
     [5, 0, 4, 0, 4, 4, 3, 2, 1, 2, 4, 4, 3, 4, 0],
     [0, 3, 0, 5, 4, 5, 0, 4, 4, 5, 3, 0, 0, 0, 0],
     [5, 4, 3, 3, 5, 5, 0, 1, 1, 3, 4, 5, 0, 2, 4],
     [5, 4, 3, 3, 5, 5, 3, 3, 3, 4, 5, 0, 5, 2, 4],
     [5, 4, 2, 2, 0, 5, 3, 3, 3, 4, 4, 4, 5, 2, 5],
     [5, 4, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0],
     [5, 4, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
     [5, 4, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
     [5, 4, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]
)
```
用户和电影的NMF分解矩阵，其中nmf_model为NMF的类，user_dis为W矩阵，item_dis为H矩阵，R设置为2：
```
nmf_model = NMF(n_components=2) # 设有2个主题
item_dis = nmf_model.fit_transform(RATE_MATRIX)
user_dis = nmf_model.components_
```
先来看看我们的矩阵最后是什么样子：
```
print('用户的主题分布：')
print(user_dis)
print('电影的主题分布：')
print(item_dis)
```
![这里写图片描述](http://img.blog.csdn.net/20160421091929447)
虽然把矩阵都显示出来了，但是仍然看着不太好观察，于是我们可以把电影主题分布矩阵和用户分布矩阵画出来：
```
plt1 = plt
plt1.plot(item_dis[:, 0], item_dis[:, 1], 'ro')
plt1.draw()#直接画出矩阵，只打了点，下面对图plt1进行一些设置

plt1.xlim((-1, 3))
plt1.ylim((-1, 3))
plt1.title(u'the distribution of items (NMF)')#设置图的标题

count = 1
zipitem = zip(item, item_dis)#把电影标题和电影的坐标联系在一起

for item in zipitem:
    item_name = item[0]
    data = item[1]
    plt1.text(data[0], data[1], item_name,
              fontproperties=fontP, 
              horizontalalignment='center',
              verticalalignment='top')
```
![这里写图片描述](http://img.blog.csdn.net/20160421092940688)
做到这里，我们从上面的图可以看出电影主题划分出来了，使用KNN或者其他距离度量算法可以把电影分为两大类，也就是根据之前的NMF矩阵分解时候设定的n_components=2有关。后面对这个n_components的值进行解释。

好我们再来看看用户的主题划分：

```
user_dis = user_dis.T #把转置用户分布矩阵
plt1 = plt
plt1.plot(user_dis[:, 0], user_dis[:, 1], 'ro')
plt1.xlim((-1, 3))
plt1.ylim((-1, 3))
plt1.title(u'the distribution of user (NMF)')#设置图的标题

zipuser = zip(user, user_dis)#把电影标题和电影的坐标联系在一起
for user in zipuser:
    user_name = user[0]
    data = user[1]
    plt1.text(data[0], data[1], user_name,
              fontproperties=fontP, 
              horizontalalignment='center',
              verticalalignment='top')

plt1.show()#直接画出矩阵，只打了点，下面对图plt1进行一些设置
```
![这里写图片描述](http://img.blog.csdn.net/20160421094257086)
从上图可以看出来，用户'五柳君', '帕格尼六', '木村静香', 'WTF'具有类似的距离度量相似度，其余11个用户具有类似的距离度量相似度。

**推荐**
对于NMF的推荐很简单
1.求出用户没有评分的电影，因为在numpy的矩阵里面保留小数位8位，判断是否为零使用1e-8（后续可以方便调节参数），当然你没有那么严谨的话可以用 = 0。
2.求过滤评分的新矩阵，使用NMF分解的用户特征矩阵和电影特征矩阵点乘。
3.求出要求得用户没有评分的电影列表并根据大小排列，就是最后要推荐给用户的电影id了。
```
rec_mat = np.dot(item_dis, user_dis)
filter_matrix = RATE_MATRIX < 1e-8
print('重建矩阵，并过滤掉已经评分的物品：')
rec_filter_mat = (filter_matrix * rec_mat).T
print(rec_filter_mat)

rec_user = '凛冬将至'  # 需要进行推荐的用户
rec_userid = user.index(rec_user)  # 推荐用户ID
rec_list = rec_filter_mat[rec_userid, :]  # 推荐用户的电影列表

print('推荐用户的电影：')
print(np.nonzero(rec_list))
```
![这里写图片描述](http://img.blog.csdn.net/20160421114051430)
通过上面结果可以看出来，推荐给用户'凛冬将至'的电影可以有'极盗者', '裁缝', '八恶人', '实习生'。

**误差**
下面看一下分解后的误差
```
a = NMF(n_components=2)  # 设有2个主题
W = a.fit_transform(RATE_MATRIX)
H = a.components_
print(a.reconstruction_err_)

b = NMF(n_components=3)  # 设有3个主题
W = b.fit_transform(RATE_MATRIX)
H = b.components_
print(b.reconstruction_err_)

c = NMF(n_components=4)  # 设有4个主题
W = c.fit_transform(RATE_MATRIX)
H = c.components_
print(c.reconstruction_err_)

d = NMF(n_components=5)  # 设有5个主题
W = d.fit_transform(RATE_MATRIX)
H = d.components_
print(d.reconstruction_err_)
```
上面的误差分别是13.823891101850649， 10.478754611794432， 8.223787135382624， 6.120880939704367
在矩阵分解当中忍受误差是有必要的，但是对于误差的多少呢，笔者认为通过NMF计算出来的误差不用太着迷，更要的是看你对于主题的设置分为多少个。很明显的是主题越多，越接近原始的矩阵误差越少，所以先确定好业务的需求，然后定义应该聚类的主题个数。

#总结

以上虽使用NMF实现了推荐算法，但是根据Netfix的CTO所说，NMF他们很少用来做推荐，用得更多的是SVD。对于矩阵分解的推荐算法常用的有SVD、ALS、NMF。对于那种更好和对于文本推荐系统来说很重要的一点是搞清楚各种方法的内在含义啦。这里推荐看一下《SVD和SVD++的区别》、《ALS推荐算法》、《聚类和协同过滤的区别》三篇文章（后面补上）。

好啦，简单来说一下SVD、ALS、NMF三种算法在实际工程应用中的区别。

 - 对于一些明确的数据使用SVD（例如用户对item 的评分） 
 - 对于隐含的数据使用ALS（例如 purchase history购买历史，watching habits浏览兴趣 and browsing activity活跃记录等）   
 -  NMF用于聚类，对聚类的结果进行特征提取。在上面的实践当中就是使用了聚类的方式对不同的用户和物品进行特征提取，刚好特征可以看成是推荐间的相似度，所以可以用来作为推荐算法。但是并不推荐这样做，因为对比起SVD来说，NMF的精确率和召回率并不显著。

引用
[1] http://www.csie.ntu.edu.tw/~cjlin/nmf/  Chih-Jen Lin写的NMF算法和关于NMF的论文，07年发的论文，极大地提升了NMF的计算过程。
[2] https://github.com/chenzomi12/NMF-example 本文代码