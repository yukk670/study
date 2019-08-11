# 数据分析

- 用适当的统计分析方法对收集到的大量数据进行分析，提取对于业务有用的信息形成结论并加以详细研究和概括的过程。

- 数据分析相关的python常用库：

  - Numpy：基础数值运算

  -  Scipy：科学计算

  - Matplotlib：数据可视化

  -  Pandas：处理序列高级函数

---

## Numpy—import numpy

**定义：**

- 基于C语言和Python接口的数值算法库
- 开源免费
- 弥补了Python语言在数值计算方面的短板
- 作为常用科学计算工具的底层支持

---

**性能：**

- 简化代码编写、提高开发效率
- 通过优化底层实现，提高运行速度

---

**优化测试**

~~~python
from __future__ import unicode_literals
import datetime as dt
import numpy as np

n = 100000
start = dt.datetime.now()
A, B = [], []
for i in range(n):
    A.append(i ** 2)
    B.append(i ** 3)
C = []
for a, b in zip(A, B):
    C.append(a + b)
print("正常的时间:",(dt.datetime.now() - start).microseconds)

start = dt.datetime.now()
A, B = np.arange(n) ** 2, np.arange(n) ** 3
C = A + B
print("Numpy优化后的时间:",(dt.datetime.now() - start).microseconds)
~~~

------

### **numpy内部基本数据类型**

| 类型名     | 类型表示符                         |
| ---------- | ---------------------------------- |
| 布尔型     | bool_                              |
| 有符号整型 | int8  int16  int32  int64          |
| 无符号整型 | uint8  uint16  uint32  uint64      |
| 浮点型     | float16  float32  float64          |
| 复数型     | complex64  complex128              |
| 字串型     | str_ 每个字符用32位Unicode编码表示 |

#### **自定义复合类型**

```python
"""
demo04_cls.py  测试自定义复合类型
"""
import numpy as np

data = [('zs', [95, 97, 98], 15),
        ('ls', [85, 87, 88], 16),
        ('ww', [75, 77, 78], 17)]

# 使用dtype定义每个元素的数据类型
# 2个Uncode字符， 3个int32整数， 1个int32
a = np.array(data, dtype='U2, 3int32, int32')
print(a)
print('ls scores:', a[1]['f1'])

# 第二种设置dtype的方式
b = np.array(data,
             dtype=[('name', 'str_', 2),
                    ('scores', 'int32', 3),
                    ('age', 'int32', 1)])
print(b)
print(b[2]['name'], b[2]['age'])

# 第三种设置dtype的方式
c = np.array(
    data, dtype={
        'names': ['name', 'scores', 'age'],
        'formats': ['U2', '3int32', 'int32']})
print(c)
print(c[0]['name'])

# 第四种设置dtype的方式
d = np.array(
    data, dtype={'name': ('U3', 0),
                 'scores': ('3i4', 16),
                 'age': ('int32', 28)})
print(d)

# 测试日期类型数组
f = np.array(['2011', '2012-01-01',
              '2013-01-01 01:01:01',
              '2011-01-01'])
print(f, f.dtype)
# 把f中的元素当做numpy.datetime数据类型，
# 精确到Day
g = f.astype('M8[D]')
print(g, g.dtype)
h = g.astype('int32')
print(h, h.dtype)
print(g[1] - g[0])
```

**类型字符码**

| 类型           | 字符码          |
| -------------- | --------------- |
| bool_          | ?               |
| int8/16/32/64  | i1/i2/i4/i8     |
| uint8/16/32/64 | u1/u2/u4/u8     |
| float16/32/64  | f2/f4/f8        |
| complex64/128  | c8/c16          |
| str_           | U               |
| datetime64     | M8[D/M/Y/h/m/s] |

```
3i4   U8  u8   M8[h]
```

---

### ndarray类数组对象

**结构：**

- 实际数据：数组的内容
- 元数据    ：对数组的描述

---

**优点：**

1. 提高了内存空间的使用效率【规定同类型的数组元素方便存储】
2. 减少对实际数据的访问频率（大部分仅仅访问元数据），提高性能

------

**示意图：**

![img](file:///C:\Users\Python\AppData\Local\Temp\msohtmlclip1\01\clip_image002.jpg)

---

#### ndarray对象的创建

```python
import numpy
a = numpy.array([1, 2, 3, 2, 1])
# 构建从0开始， 到10结束， 步长为2的数组
numpy.arange(0, 10, 2)
# 创建10个元素全为0的数组  
# dtype用于设置元素的数据类型, 默认为float
numpy.zeros(10, dtype='bool')
# 创建10个元素全为1的数组
numpy.ones(10)
# 创建一个与a的结构相同的数组，元素全为0
print(np.zeros_like(a))
```

---

**方法**

+ 将ndarray对象的元素转换为目标类型：
  + astype()
    + 参数：目标类型

------

**属性**

- dtype：元素类型
- shape：纬度
- ndim：维数
- itemsize：元素字节数
- nbytes：数组的总字节数
- real：返回复数数组中所有元素的实部
- imag：返回复数数组中所有元素的虚部
- T：返回数组的转置视图
- flat：多维数组的扁平迭代器

~~~python
import numpy
data = numpy.array([[1 + 1j, 2 + 4j, 3 + 7j],
                    [4 + 2j, 5 + 5j, 6 + 8j],
                    [7 + 3j, 8 + 6j, 9 + 9j]])
print(data.dtype)		#complex128
print(data.ndim)		#2
print(data.itemsize)	#16
print(data.nbytes)		#144(16*9)
print(data.real)	
				[[ 1.  2.  3.]
                 [ 4.  5.  6.]
                 [ 7.  8.  9.]]
print(data.imag)
				[[ 1.  4.  7.]
                 [ 2.  5.  8.]
                 [ 3.  6.  9.]]
print(data.T)			#实虚交换
				[[ 1.+1.j  4.+2.j  7.+3.j]
                 [ 2.+4.j  5.+5.j  8.+6.j]
                 [ 3.+7.j  6.+8.j  9.+9.j]]
for item in data.flat:
    print(item,end=" ")
(1+1j) (2+4j) (3+7j) (4+2j) (5+5j) (6+8j) (7+3j) (8+6j) (9+9j)
~~~

---

**使用方式：**

~~~python
import numpy
a = [[1,2,3],[4.0,5,6]]
b = numpy.array(a)#声明一个ndarray对象
print("b的数值:",b)
print("b内的元素类型:",b.dtype,"第一个元素类型:",type(b[0][0]))
print("b的维度():",b.shape)
c = numpy.arange(1, 10, 2)
print("arange得到的对象:",c, c.shape, c.dtype, sep='\n')
d = numpy.array([np.arange(1, 4), np.arange(4, 7)])
print(d, d.shape, d.dtype, sep='\n')
e = d.astype(float)
print("将对象内所有元素转换为浮点类型:",e, e.shape, e.dtype, sep='\n')
~~~

---

#### ndarray变维

##### 视图变维

> ​			元数据独立，实际数据共享
>

- .reshape()
  - 参数：目标维度
  - 返回：目标维度的ndarray
- .ravel()
  - 返回：一维ndarray

~~~python
import numpy as np
a = np.arange(1,9)
print(a)
b = a.reshape(2,4)
print(b)
~~~

------

##### 复制变维

> ​			元数据独立，实际数据独立
>

- .flatten()
- .copy()



------

##### 就地变维

> ​			修改元数据的维度信息，不产生新的数组
>

- .shape
- .resize()

~~~python
.shape = (4,2)
print(b)
b.resize(2,2,2)
print(b)
~~~

------

#### ndarray切片操作

##### 一维切片

~~~python
import numpy
a = numpy.arange(1,10)
print(a[:3])		#1,2,
print(a[3:6])		#3,4,5
print(a[6:])		#6,7,8,9
print(a[::-1])		#9,8,7,6,5,4,3,2,1
print(a[:-4:-1])
print(a[-4:-7:-1])
print(a[-7::-1])
print(a[:])
print(a[::3])
print(a[1::3])
~~~

------

##### 多维切片

​		以，为分隔符，分别对  页/行/列 每一维度进行切片

#### ndarray掩码操作

~~~python
import numpy
a = numpy.arange(1,10)
#根据条件匹配对应数据
condition = (a%2==0)
print(a[condition])		#2,4,6,8
#对数组进行排序
condition = [8,1,2,7,3,4,6,5,0]
~~~

---

#### ndarray组合拆分

---

##### 垂直

- 垂直方向组合操作：numpy.vstack()
  - 参数：(ndarray1,ndarray2)
- 垂直方向拆分操作：numpy.vsplit()
  - 参数1：目标ndarray
  - 参数2：拆分成几个

~~~python
import numpy	#a，b两个ndarray对象
                [[1 2 3]
                 [4 5 6]](2, 3)
                [[ 7  8  9]
                 [10 11 12]](2, 3)
numpy.vstack((a,b))
                [[ 1  2  3]
                 [ 4  5  6]
                 [ 7  8  9]
                 [10 11 12]] (4, 3)
a = numpy.dsplit(c,2)[0]
                [[[1]
                  [2]
                  [3]]

                 [[4]
                  [5]
                  [6]]] (2, 3, 1)
~~~

------

##### 水平

+ 水平方向组合操作：numpy.hstack()
  - 参数：(ndarray1,ndarray2)
+ 水平方向拆分操作：numpy.hsplit()
  - 参数1：目标ndarray
  - 参数2：拆分成几个

~~~python
import numpy	#a，b两个ndarray对象
                [[1 2 3]
                 [4 5 6]](2, 3)
                [[ 7  8  9]
                 [10 11 12]](2, 3)
numpy.hstack((a,b))
               [[ 1  2  3  7  8  9]
 				[ 4  5  6 10 11 12]] (2, 6)
a,b = numpy.hsplit(c,2)#返回结果与最初a,b一致
~~~

------

##### 深度

+ 深度方向组合操作：numpy.dstack()
  - 参数：(ndarray1,ndarray2)
+ 深度方向拆分操作：numpy.dsplit()
  - 参数1：目标ndarray:
  - 参数2：拆分成几个

~~~python
import numpy	#a，b两个ndarray对象
                [[1 2 3]
                 [4 5 6]](2, 3)
                [[ 7  8  9]
                 [10 11 12]](2, 3)
numpy.dstack((a,b))
               [[[ 1  7]
                 [ 2  8]
                 [ 3  9]]

                [[ 4 10]
                 [ 5 11]
                 [ 6 12]]] (2, 3, 2)
a,b = numpy.dsplit(c,2)#返回结果与最初a,b一致
~~~

------

##### 组合拆分函数

+ 以axis作为轴向进行组合：numpy.concatenate()
  + 参数1：(ndarray1,ndarray2)
  + 参数2：axis=数字
+ 以axis作为轴向进行拆分：numpy.split()
  + 参数1：目标ndarray:
  + 参数2：拆分成几个
  + 参数3：axis=数字
+ axis说明：
  + ndarray1,ndarray2都是二维数组
    + 0：垂直方向
    + 1：水平方向
  + ndarray1,ndarray2都是三维数组
    + 2：深度方向

------

##### 一维组合

+ numpy.row_stack()
  + 参数1：(ndarray1,ndarray2)
+ numpy.column_stack()
  + 参数1：(ndarray1,ndarray2)

~~~python
import numpy	#a，b两个ndarray对象
                [[1 2 3]
                 [4 5 6]](2, 3)
                [[ 7  8  9]
                 [10 11 12]](2, 3)
numpy.row_stack((a,b))
                [[ 1  2  3]
                 [ 4  5  6]
                 [ 7  8  9]
                 [10 11 12]] (4, 3)
numpy.column_stack((a,b))
                [[ 1  2  3  7  8  9]
                 [ 4  5  6 10 11 12]] (2, 6)
~~~

---

#### 数组的轴向汇总

~~~python
#轴向汇总函数
ary = numpy.random.uniform(0,10,(3,5))
ary = ary.astype(int)
def func(data):
    return data.mean()
print(numpy.apply_along_axis(func,0,ary))#axis轴向0,1
~~~

------

### 加载文件

~~~python
date,opening_price = numpy.loadtxt("..//xxx/aapl.csv",
                 delimiter=",",			#分隔符
                 unpack=True,			#是否拆包
                 usecols=(1,3),       	#需要读取哪些列
                 dtype="u10,f8",     #设置每列的数据类型
                 converters={1:func})   #数据转换器
~~~

---

**示例**

```python
import numpy
import matplotlib.pyplot as mp
import matplotlib.dates as md
from datetime import datetime

def dmy2ymd(dmy):
    dmy = str(dmy, encoding="utf-8")
    d = datetime.strptime(dmy, "%d-%m-%Y").date()
    return d.strftime("%Y-%m-%d")

dates, closing_prices = numpy.loadtxt("aapl.csv", delimiter=",", usecols=(1, 6),
                                      unpack=True, dtype="M8[D],f8", converters={1: dmy2ymd})
# 设置主刻度定位器为每周一
ax = mp.gca()
ax.xaxis.set_major_locator(
    md.WeekdayLocator(byweekday=md.MO))
ax.xaxis.set_major_formatter(
    md.DateFormatter("%Y/%m/%d"))
# 把M8[D]转为matplotlib识别的date类型
dates = dates.astype(md.datetime.datetime)
mp.plot(dates, closing_prices, linestyle="-", linewidth=3,
        color="red", alpha=0.3,label="closing_price")
mp.xlabel("Date")
mp.ylabel("Price")
mp.tick_params(labelsize=10)
mp.grid(linestyle=":")
mp.legend()
mp.show()
```

------

#### K线图

> 实体（开盘、收盘）、影线（最高价、最低价）

```python		
#之前的部分-参考加载文件
rise = closing_prices >= opening_prices
#填充色
color = numpy.array(['white' if x else "green" for x in rise])
# 边缘色
edgecolor = numpy.array(['red' if x else "green" for x in rise])

# 绘制K线图-影线
mp.vlines(dates,lowest_prices,highest_prices,color = edgecolor,
          linewidth=4)
# 绘制K线图-实体
mp.bar(dates,closing_prices-opening_prices,0.8,opening_prices,
       color=color,edgecolor=edgecolor,zorder = 2)
```

---

### 符号数组

> 把样本数组变为对应的符号数组，正数变为1，负数变为-1,0依然为0

+ a = np.sign(array)

#### OBV能量潮

> 成交量可以反映市场对某支股票的人气，成交量是一支股票上涨的能量。股票上涨往往需要较大的成交量，而下跌则不然
>
> 若相比上一天收盘价上涨，则为正成交量（红色）。若相比上一天收盘价下跌，则为负成交量（绿色）

~~~python
diff_prices = numpy.diff(closing_prices)
sign_price = numpy.sign(diff_prices)
#
up_obvs = volumns[1:][sign_price>=0]
up_dates = dates[1:][sign_price >= 0]
mp.bar(up_dates,up_obvs,0.8,color="red",label="up_obvs")

low_obvs = volumns[1:][sign_price<0]
low_dates = dates[1:][sign_price < 0]
mp.bar(low_dates,low_obvs,0.8,color="green",label="low_obvs")
~~~

---

#### 数组处理函数

+ 对数组进行二次处理
  + ary=numpy.piecewise(原数组，条件序列，取值序列)

~~~python
a = numpy.array([50,60,80,45,30])
b = numpy.piecewise(a,[a<60,a==60,a>60],[-1,0,1])
print(b)
~~~

------

## Matplotlib--import matplotlib

> 是python的一个绘图库。使用它可以很方便的绘制出版质量级别的图形。

---

### 基本绘图

~~~python
import matplotlib.pyplot as mp
#绘制折线，设置线型、线宽、颜色等
mp.plot()
mp.show()
~~~
> 绘制折线，设置线型、线宽、颜色等
~~~python
mp.plot(xarray,yarray,	#所有点x和y的坐标
        linestyle=“”,	#线型：	‘：’	“-”	“--”
        linewidth=3,	#线宽：3倍线宽
        color=“”，	   #颜色：英文单词颜色或者颜色首字
       	alpha = 0.3)	#透明度
~~~

------

+ 绘制垂直线：mp.vlines(val,ymin,ymax)

+ 绘制水平线：mp.hlines(val,xmin,xmax)

> 设置坐标轴范围

~~~python
mp.xlim(x_min,x_max)#设置x轴的可视范围
mp.ylim(y_min,y_max)#设置y轴的可视范围
~~~

> 设置坐标轴刻度

+ 设置坐标轴位置、颜色：mp.xticks(x_val_list，x_text_list)
  + 参数1：x轴刻度值序列
  + 参数2：x轴刻度值的文本序列（可选）
+ 设置坐标轴位置、颜色：mp.yticks(y_val_list，y_text_list)
  - 参数1：y轴刻度值序列
  - 参数2：y轴刻度值的文本序列（可选）

~~~python
import matplotlib.pyplot as mp
mp.plot(range(12),range(12))
#r"$\frac{分子}{分母}$"   
mp.xticks([i for i in range(12)],[r"$\frac{%d}{4}$"%i for i in range(12)])
mp.show()
~~~

> 设置坐标轴 

~~~python
ax = mp.gca()				#获取当前坐标轴
axis = ax.spines['left']	#left/right/top/bottom
axis.set_color("none")		#修改坐标轴的颜色
#移动坐标轴的位置：	
#data基于数据坐标系进行定位  0把坐标轴移动到0的位置
axis.set_position(("data",0))
~~~

---

> 图例（比如：图形的曲线说明）

~~~python
#label：定义当前曲线的标签名	该标签名会在图例中显示
mp.plot(label='sin(x)')
mp.legend(loc="upper left")	#显示图例loc指定显示位置
#help(mp.legend)可以查看帮助文档
~~~

------

> 特殊点

~~~python
mp.scatter(xarray,yarray,	#给出点的坐标
           marker="D",		#点型（选项见点型图表）
           s =60,		   #点的大小
           edgecolor = "" ,	#边缘色
           facecolor="",	#填充色
           zorder=3		#绘制图层编号(编号越大，图层越上)
)
~~~

------

> 备注

~~~python
mp.annotate(
	r"$[x,y]$",				#备注的文本内容
    xycoords="",			#目标点的坐标系
    xy=(1,2),				#目标点的坐标
    #定位备注文本位置所使用的坐标系
    textcoords="offset points",		
    xytext=(-10,-10)		#备注文本的坐标
    fontsize=12,			#字体大小
    #箭头属性字典
    arrowprops=dict(
    	arrowstyle:"",		#箭头样式
        connectionstyle=""	#连接线的样式
    )
)
~~~

---

### 颜色映射

![matplotlib_cmap](.\matplotlib_cmap.png)

---

### COLOR颜色图表

![matplotlib_colors](.\matplotlib_colors.png)

---

### 点型图表

![matplotlib_markers](.\matplotlib_markers.png)

------

### 刻度文本的特殊语法-latex语法

![laTex_](.\laTex_.png)

------

![laTex_](.\LaTeX_eg.gif)

------

$$
a^2 + b^2 = c^2 \quad\quad\quad
-\frac{\pi}{2}	\quad\quad\quad
\sqrt[3]{\frac{5}{2}}
$$

------

### 高级绘图

#### 设置窗口常用参数

~~~python
#手动创建一个窗口，窗口的标题titleA（多窗口只要标题名不同，相同则为原窗口）
mp.figure('titleA',facecolor='填充色')
mp.title("",fontsize=12)     #设置图标的标题
mp.xlabel("time",fontsize=12)#设置x轴的标签
mp.ylabel("v",fontsize=12)	 
mp.tick_params(labelsize=8)	 #设置刻度参数labelsize刻度大小
mp.grid(linestyle=":")		 #设置图表网格线
mp.tight_layout()			 #紧凑布局
mp.show()
~~~

------

#### 子图

> 矩阵式布局

+ 绘制窗口中的某个子图：mp.subplot(rows,cols,num)
  + 参数3：子图编号

~~~python
mp.subplot(2,2,1)	#绘制2行2列的第一幅图
~~~

------

> 网格式布局

+ 支持单元格的合并

~~~python
import matplotlib.gridspec as mg
mp.figure(...)
gs = mg.GridSpec(3,3)	#构建3*3的网格布局结构
mp.subplot(gs[0,:2])
~~~

------

> 自由布局

+ 任意布局

~~~python
#0.1,0.2:子图左下角定点坐标(整体为1,0.1代表当前位置)
#0.5：子图宽度	0.3：子图高度
mp.axes([0.1,0.2,0.5,0.3])
mp.text(...)
~~~

------

#### 刻度定位器

~~~python
ax=mp.gca()					#获取当前坐标轴
#设置x轴的主刻度定位器,以无刻度显示
ax.xaxis.set_major_locator(mp.NullLocator())
#设置x轴的次刻度定位器，以每0.1为刻度标准显示
ax.xaxis.set_minor_locator(mp.MultipleLocator(0.1))
#mp.MaxNLocator(nbins=5)将当前刻度最多分为5份，分不出5份就按少量分
#mp.AutoLocator()自动分
~~~

#### 刻度网格线

~~~python
ax = mp.gca()
ax.grid(
	which = "",		#'major' 'minor' 'both'
    axis="",		#'x'	'y'		'both'
    linewidth=1,
    linestyle=":",
    color="",
    alpha=0.5
)
~~~

------

#### 半对数坐标

> y轴将会以指数方式递增。基于半对数坐标系表示上述曲线可以更好的观察底部数据细节

> 应用领域：如音频处理

~~~python
#plot改为semilogy，坐标系将会改为半对数坐标系
mp.semilogy()
~~~

---

#### 散点图

~~~python
mp.scatter(x,y,	#给出点的坐标
           marker="D",		#点型（选项见点型图表）
           s =60,		   #点的大小
           edgecolor = "" ,	#边缘色
           facecolor="",	#填充色
           zorder=3,	#绘制图层编号(编号越大，图层越上)
           c=(x-172)**2+(y-65)**2, #设置过渡性颜色，影响颜色映射的一个数组
           cmap = 'jet' #颜色映射
)
~~~

##### 随机生成复合正态分布的随机数

~~~python
#随机生成500个数,172数学期望，20标准差（大小影响波动范围）
numpy.random.normal(172,20,500)
~~~

---

#### 区域填充

> 以某种颜色填充两条曲线的闭合区域

~~~python
mp.fill_between(
	x,						#x值得区间
    sin_x,cos_x,sinx<cosx	#与x组成的两条曲线和填充条件
    color="",alpha=0.5
)
~~~

---

#### 条形图、饼状图

~~~python
#条形图
mp.bar(x,			#横坐标数组
      height,		#对应每个x的柱子高度
      width,        #对应每个x的柱子宽度
      base_paht,	#基本位置，代表高度从哪开始
      color="",     #框内的颜色
      edgecolor="", #边框颜色
      label="",
      alpha=0.2)
#饼状图
mp.axis("equal")	#等轴比例绘制pie
mp.pie(
	values,			#扇形值列表
    spaces,			#扇形之间间距列表
    labels,			#扇形标签文本列表
    colors,		   #扇形颜色列表
    "%.2f%%",		#百分比的格式
    shadow=True,	#阴影（会显得更高）
    startangle=90, #饼状图的绘制起始角度
    radius=1		#饼状图的半径
)
~~~

---

#### 等高线图、热成像图

> 等高线图属于3D数学模型，需要整理网格点坐标矩阵，还需要得到每个点的高度值

~~~python
n = 1000
# 生成网格点坐标矩阵
x,y = numpy.meshgrid(numpy.linspace(-3,3,n),
                     numpy.linspace(-3,3,n))
# 根据x,y计算当前坐标下的z高度值 exp是e的多少次方
z = (1-x/2 + x ** 5 + y **3) * numpy.exp(-x**2-y**2)
contour = mp.contour(
	x,y,z,		#x,y网格点坐标矩阵，z每个坐标高度值
    8,	   	   #把整体高度分为8份
    colors="black",	#等高线颜色
    linewidth=1	#等高线的线宽
)
#等高线对象绘制标注
mp.clabel(contour,
          inline_spacing=1, #文本与线的间隔
          fmt="%.1f"
          ,fontsize=10)
#等高线填充颜色
mp.contourf(x,y,z,8,camp="jet")
mp.grid(linestyle=":")
mp.show()
~~~

---

~~~python
#以jet映射显示z矩阵的图像值
mp.imshow(z,camp='jet'
         ,origin='lower'  #y坐标轴方向lower较小的数字在下方
)						  #upper较小的数字在上方					
~~~

---

#### 极坐标系

> 当需要处理角度【极角&theta;】相关的函数图像时，可能需要使用极坐标系绘制图像。对应点到原点的距离【极径&rho;】
>
> 如：r=0.6*t	极角&theta;关于极径&rho;的函数

~~~python
mp.gca(projection="polar")
mp.plot(x,y,color="dodgerblue")
~~~

#### 3D图形

> matplotlib支持绘制三维线框图，三维曲面图，三维散点图，需要使用axes3d提供3d坐标系。

~~~python
from mpl_toolkits.mplot3d import axes3d
ax3d = mp.gca(projection = "3d")
#绘制3d线框图
ax3d.plot_wireframe(x,y,z,		#xy为numpy.meshgrid网格点坐标矩阵
                   rstride=30,	 #行跨距
                   cstride=30,	 #列跨距
                   linewidth=1,	color="")	
#绘制3d曲面图
ax3d.plot_surface(x,y,z,rstride=50,cstride=50,cmp="jet")  #这里的行列跨距指每隔多少颜色渐变
#绘制3d散点图
ax3d.scatter(x,y,z,alpha=0.6#确定一组散点坐标
           s =60,		   #点的大小
           c=(x-172)**2+(y-65)**2, #颜色映射依据
           cmap = 'jet' #颜色映射
)		
~~~

------

#### 简单动画

> 动画即是在一段时间内快速连续的重新绘制图像的过程。

~~~python
 import matplotlib.animation as ma
 def update(number):#number为运行多少次update
    pass 
 animat = ma.FuncAnimation(mp.gcf(),	#作用域当前窗体
              update,					#更新函数的函数名
              interval=30)	#每隔30毫秒，执行一次update	
~~~

---

**示例**

~~~python
"""1.随机生成100个气泡，每个气泡拥有position，size，growth，color属性，绘制到窗口中
2.开启动画，在update函数中更新每个气泡的属性并重新绘制
"""
import numpy
import matplotlib.pyplot as mp
import matplotlib.animation as ma
n = 100
balls = numpy.zeros(n,dtype=[
    ("position",float,2),
    ("size",float,1),
    ("growth",float,1),
    ("color",float,4),
])
# 初始化每个泡泡位置
# uniform:从0到1取随机数,填充n行2列的数组
balls["position"] = numpy.random.uniform(0,1,(n,2))
balls["size"] = numpy.random.uniform(50,70,n)
balls["growth"] = numpy.random.uniform(10,20,n)
balls["color"] = numpy.random.uniform(0,1,(n,4))

mp.figure('Bubble',facecolor="lightgray")
mp.title("Bubble",fontsize=18)
mp.xticks([])
mp.yticks([])
sc = mp.scatter(balls["position"][:,0],
           balls["position"][:,1],
           s=balls['size'],
           color=balls["color"])

def update(number):
    balls['size'] += balls["growth"]
    # 让某个泡泡破裂,从头开始执行
    boom_i = number % n
    balls[boom_i]["size"] = 60
    balls[boom_i]["position"] = numpy.random.uniform(0,1,(1,2))
    sc.set_sizes(balls["size"])
    # 设置偏移属性
    sc.set_offsets(balls['position'])

anim = ma.FuncAnimation(mp.gcf(),  # 作用域当前窗体
                 update,  # 更新函数的函数名
               interval=30)  # 每隔30毫秒，执行一次update
mp.show()
~~~

---

##### 生成器函数实现动画绘制

~~~python
def update(data):pass
def generator():yield data#生成器函数
#每10毫秒执行生成器，把结果交给update,更新图像
ma.FuncAnimation(mp.gcf(),update,generator,interval=30)
~~~

---

**模拟心电图：y=sin(2πt) * exp(sin(0.2πt))**

~~~python
import numpy
import matplotlib.pyplot as mp
import matplotlib.animation as ma

mp.figure('Bubble',facecolor="lightgray")
mp.title("Bubble",fontsize=18)
mp.xlim(0,10)
mp.ylim(-3,3)
mp.grid(linestyle=":")
# 返回的数据是一个元祖Line2D(Signal)
plot = mp.plot([],[],color="dodgerblue",label="Signal")[0]

def update(data):
    t,v = data
    x,y = plot.get_data()#x,y为ndarray数组
    x = numpy.append(x,t)
    y = numpy.append(y,v)
    #重新绘制图像
    plot.set_data(x,y)
    # 移动坐标轴
    if x[-1] > 5:
        mp.xlim(x[-1]-5,x[-1]+5)
x = 0
def generator():
    global x
    x+=0.05
    y = numpy.sin(2*numpy.pi*x) * numpy.exp(numpy.sin(0.2*numpy.pi*x))
    yield (x,y)

anim = ma.FuncAnimation(mp.gcf(),
                 update,generator,
                 interval=30)

mp.show()
~~~



---

## 数学公式

### 算数平均值

> 算数平均值表示对真值的无偏估计

~~~python
s = [s1,s2,s3....sn]
m = (s1+s2+s3+...+sn) / n
numpy.mean(array)#求array数组的平均值
~~~

#### 加权平均值

> ​	算数平均值是权重都为1所以w内全是1，分子分母可以转换回去

~~~python
s = [s1,s2,s3....sn]	#样本
w = [w1,w2,w3....wn]	#权重
A = (s1w1+s2w2+...snwn)/(w1+w2+w3...+wn) #权重平均值
numpy.average(closing_prices,			#获取加权平均值
              weights=array)			#权重
~~~

> 成交量加权平均值（VWAP)：以每天的交易量作为权重，计算加权平均值。（VWAP体现了时长对当前交易价格的认可度）

~~~python
mp.hlines(numpy.average(closing_prices,weights = volumns),dates[0],dates[-1]
          ,color="limegreen",label="vwap(closing_price)")
~~~

> 时间加权平均值（TWAP）：以时间距离长短为权重，计算加权平均值。（VMAP体现了时长对当前价格的影响）

~~~python
times = numpy.arange(1,closing_prices.size+1)
mp.hlines(numpy.average(closing_prices,weights = times),dates[0],dates[-1]
          ,color="violet",label="twap(closing_price)")
~~~

---

### 最值

~~~python
numpy.max(array)
numpy.min(array)
numpy.pip(array) 		#求array数组的极差（max-min)
numpy.argmax(array) 	#获取array数组最大值下标
numpy.argmin(array)		
#a与b是同维数组
numpy.maximum(a,b)		#a,b数组同索引位置的数字进行比较留下的大的数字组成的新数组
numpy.minimum(a,b)
~~~

---

### 中位数

> 将多个样本按照大小排序，取中间位置的元素
>
> 当样本集中没有太多异常数据时可以用平均数。如果有一些异常数据（极大/极小）比较适合使用中位数。

~~~python
numpy.median(ary)#对有序数组array，求中位数
#中位数算法
(ary[(size-1)/2] + ray[size/2]) /2
~~~

---

### 标准差

> 标准差用于衡量一组数据的离散程度

~~~python
s = [s1,s2,s3....sn]
m = (s1+s2+s3+...+sn) / n
D = [d1,d2,d3...dn]		#离差（di=si-m）
Q = [q1,q2,q3...qn]     #离差方(qi=di^2)
v = （q1+q2+q3...qn）/n  #总体方差
s=sqrt(v)		#总体标准差
v = （q1+q2+q3...qn）/ （n-1） #样本方差
s = sqrt(v)		#样本标准差

std = numpy.std(array)#总体标准差
std = numpy.std(array,ddof=1)#样本标准差，ddof为修正参数，意味着分母n-1
~~~

---

### 移动平均线

> 绘制5日移动均线：从第五天开始，每天计算最近5天的收盘价的平均值而构成的一条线。

~~~python
a b c d e f g...
s1 = (a+b+c+d+e)/5
s2 = (b+c+d+e+f)/5
...
[s1,s2...]#这组数据构成的一条线称为移动均线
# 计算5日均线
sma5 = numpy.zeros(closing_prices.size - 4)
for i in range(sma5.size):
    sma5[i] = closing_prices[i:i+5].mean()
mp.plot(dates[4:], sma5, linestyle="--", linewidth=3,
            color="orangered", label="SMA-5")
~~~

---

### 卷积

> 1.每一次的运算结果会不会受到之前的结果影响，影响的话就使用卷积。那么其卷积核数组所代表的是影响的比重！
>
> 2.卷积常用于数据平滑处理、降噪等操作。可以更好的看出数据的走势。需要考虑清楚的是卷积核的选取。

~~~python
原数组：[1 2 3 4 5]
卷积核数组：[8 7 6]
针对卷积核对原数组执行卷积运行的过程如下(先元素)
                  44 65 86#有效卷积（876都用到了）  
			   23 44 65 86 59#同维卷积（原数组的大小）
第3步：      8  23 44 65 86 59 30  #完全卷积（full）
第1步：0  0  1  2  3  4  5  0  0  
第2步：6  7  8
第4步：   6  7  8
第5步：      6  7  8
第6步：		 6  7  8
第7步：			6  7  8
第8步：			   6  7  8
第9步：				  6  7  8
--第三步-1：0*6+0*7+0*8
--第三步-2：0*6+1*7+2*8
~~~

#### 卷积完成移动平均线

~~~python
#array:原数组	core:卷积核数组
#type:卷积核数组:
#	"valid"-有效卷积	"same"：同维卷积		"full":完全卷积
r = np.convolve(array,core,type)
~~~

---

#### 加权卷积

~~~python
# 从y=e^x,取得5个函数值作为卷集核
weights = numpy.exp(numpy.linspace(-1,0,5))[::-1]
weights /= weights.sum()#保障卷积核之和为1
ema5 = numpy.convolve(closing_prices,weights,"valid")
    mp.plot(dates[4:], ema5, linestyle="--", 	           linewidth=3,color="orangered", label="EMA-5")
~~~

---

### 布林带

> 布林带由三条线组成：
>
> 中轨：移动平均线
>
> 上轨：中轨+2*5日收盘价标准差
>
> 下轨：中轨-2*5日收盘价标准差

~~~python
#绘制布林带
stds = numpy.zeros(ema5.size)
for i in range(stds.size):
    stds[i] = closing_prices[i:i+5].std()

#计算上下轨
uppers = ema5 + 2 * stds
lowers = ema5 - 2 * stds
mp.plot(dates[4:], uppers, linestyle="-", linewidth=2,
            color="red", label="Uppers")
mp.plot(dates[4:], lowers, linestyle="-", linewidth=2,
            color="red", label="Lowers")
mp.fill_between(dates[4:],uppers,lowers,uppers >lowers,color = "red",alpha=0.3)
~~~

---

## 函数

### 线性预测

$$
如（表示如下）：ax+by+cz=d\quad bx+cy+dz=e\quad cx+dy+ez=f\\
\left[ \begin{array}{c}
a & b & c\\
b & c & d\\
c & d & e\\
\end{array}\right]
\times
\left[ \begin{array}{c}
x\\
y\\
z\\
\end{array}\right]
=
\left[ \begin{array}{c}
d\\
e\\
f\\
\end{array}\right]\\
\quad A\quad\quad\quad\quad\quad\quad\quad\quad B
$$

~~~python
# 整理三(N)元一次方程组,基于线性模型,实现线性预测
N = 3
pred_vals = numpy.zeros(closing_prices.size - N * 2 + 1)  # 预测第31天
for i in range(pred_vals.size):
     A = numpy.zeros((N, N))
     for j in range(N):
         A[j,] = closing_prices[i + j:i + j + N]
         B = closing_prices[i + N:i + N * 2]
         #线性预测计算xyz
         x = numpy.linalg.lstsq(A, B)[0]
         # B点乘x [3]*x+[4]*y+[5]*z
         pred_vals[i] = B.dot(x)  
mp.plot(dates[2 * N:], pred_vals[:-1], "o-",
            color="red", label="Predict Prices")
print(pred_vals[-1])
~~~

---

#### 线性拟合

> 可以需求与一组数据走向趋势规律相适应的线性表达式

$$
根据y=kx+b\\
\left[ \begin{array}{c}
x1 \quad 1\\
x2 \quad 1\\
x3 \quad 1\\
xn \quad 1\\
\end{array}\right]
\times
\left[ \begin{array}{c}
k\\
b\\
\end{array}\right]
=
\left[ \begin{array}{c}
y1\\
y2\\
y3\\
y4\\
\end{array}\right]\\
\quad A\quad\quad\quad\quad\quad\quad\quad B
$$

> 使用numpy.linalg.lstsq(A,B)求得的x与b，可能无法让所有的方程成立，但是拟合的直线方程的误差是最小的。

~~~python
#利用线性拟合画出股价的趋势线【（最高价+最低价+收盘价）/3】
# 计算所有趋势点
B = (highest_prices+lowest_prices+closing_prices) / 3
# 线性拟合
days = dates.astype("M8[D]").astype(int)
A = numpy.column_stack((days,numpy.ones(days.size)))
k,b = numpy.linalg.lstsq(A,B)[0]
mp.plot(dates, B, linestyle="--", linewidth=3,
            color="dodgerblue", alpha=0.3, 		                   label="trend_line")
#绘制趋势线
B = days*k + b
mp.plot(dates, B, linestyle="--", linewidth=2,
            color="orangered", label="TrendLine")
~~~

---

### 协方差

> 通过两组统计数据计算而得的协方差可以评估者两组统计数据的相关性。协方差值为正，则为正相关；若值为负，则为负相关。绝对值越大相关性越强。

~~~html
<离差
    dev_A = [a1,a2,a3...an] - ave_A --ave平均值
    dev_B = [b1,b2,b3...bn] - ave_B>
<协方差
     cov_ab = numpy.mean(dev_A * dev_B)--b和a相似度
     cov_ba = numpy.mean(dev_B * dev_A)--a和b的相似度
~~~

~~~python
# 计算两组数据的协方差
ave_bhp = np.mean(bhp_closing_prices)
ave_vale = np.mean(vale_closing_prices)
# 离差
dev_bhp = bhp_closing_prices - ave_bhp
dev_vale = vale_closing_prices - ave_vale
# 协方差
cov_ab = np.mean(dev_bhp * dev_vale)
print(cov_ab)
~~~

#### 相关系数

> 相关系数是一个[-1,1]之间的数。若相关系数越接近于1则表示两组样本越正相关；若相关系数越接近于-1，则表示两组样本越负相关；若相关系数越接近0，说明两组样本没啥大关系。
>
> 协方差除以两组样本标准差之积

~~~python
k = cov_ab / ( numpy.std(bhp_closing_prices) *
                numpy.std(vale_closing_prices) ) 
~~~

---

#### 相关矩阵

> 获取相关矩阵，该矩阵中包含相关系数

~~~python
k = numpy.corrcoef(bhp_closing_prices,
                vale_closing_prices)
~~~

$$
相关矩阵结构\\
\left[ \begin{array}{c}
1 & 0.86 \\
0.86 & 1 \\
\end{array}\right]
=>
\left[ \begin{array}{c}
a与a的相关系数 & a与b的相关系数 \\
b与a的相关系数 & b与b的相关系数 \\
\end{array}\right]
$$

~~~python
#相关矩阵的分子矩阵(协方差矩阵)
cov_ab = numpy.cov(a,b)
~~~

---

### 多项式拟合

> 多项式拟合的目的是为了找到一组P<sub>0</sub>-P<sub>n</sub>，使得拟合方程尽可能的与实际样本数据相符合。

$$
多项式的一般形式：\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\\
f(x) = p_0x^n + p_1x^{n-1}+p_2x^{n-2}+...p_n\\
拟合得到的多项式函数与真实结果的误差方如下：\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\\
loss = ( y_1 - f(x_1) )^2 + ( y_2 - f(x_2) )^2+...( y_n - f(x_n) )^2
$$

> 那么多项式拟合的过程即为求取一组p_0-p_n,使得loss的值最小。

+ 多项式拟合p：numpy.polyfit(x,y,最高次幂)
  + x,y为一组样本数据
  + 最高次幂：想得到的多项式函数的最高次幂
  + 返回值：拟合得到的多项式的系数数组
+ 把x带入多项式函数p，求得每个x对应的函数值
  + numpy.polyval(p,x)
+ 多项式求导Q：numpy.polyder(p)
  + 返回值：导函数的系数数组Q
+ 已知多项式系数p，求多项式函数的根（y=0时x的值）
  + numpy.roots(p)
+ 求P1,P2的差函数
  + Q = np.polysub(P1,P2)

**示例：求多项式y=4x<sup>3</sup>+3x<sup>2</sup>-1000x+1的驻点坐标**

~~~python
x = numpy.linspace(-20,20,1000)
y = 4*x**3 + 3*x**2 - 1000*x + 1
# 求驻点坐标
P = numpy.array([4,3,-1000,1])
Q = numpy.polyder(P)
xs = numpy.roots(Q)
ys = numpy.polyval(P,xs)

mp.plot(x,y,color="dodgerblue")
mp.scatter(xs,ys,s=60,marker = "o",c="red",zorder=3)
mp.show()
~~~

**计算2支股票的差价**

~~~python
# 计算两支股票的差价
diff_prices = bhp_closing_prices - vale_closing_prices
mp.plot(dates, diff_prices, c='dodgerblue',
        linewidth=1, linestyle='--',
        label='Diff Princes', alpha=0.4)
mp.scatter(dates, diff_prices, s=50,c='steelblue')
# 针对差价函数，执行多项式拟合
days = dates.astype('M8[D]').astype('int32')
P = np.polyfit(days, diff_prices, 4)
polyline = np.polyval(P, days)
mp.plot(dates, polyline, c='orangered',
        linewidth=2, linestyle='-',label='PolyLine')
~~~

---

### 函数矢量化

> 只能处理标量参数的函数经过函数矢量化后就可以接受数组参数参数，对数组中每个元素执行相同处理。

~~~python
import math
def f(a,b):
    return math.sqrt(a**2+b**2)
print(f(3,4))# 处理标量
a = numpy.array([3,4,5])
b = numpy.array(4)
# 把f函数矢量化(a,b要求维度相同)
f_vec = numpy.vectorize(f)
print(f_vec(a,b))
print(numpy.vectorize(f)(a,b))
#2代表2个参数，1代表函数有一个返回值
f_func = numpy.frompyfunc(f,2,1)
print(f_func(a,b))
~~~

***示例***

~~~python
# 每天开盘价的0.99买入,收盘价就卖出
def progit(opening_prices, highest_prices, lowest_prices, closing_prices):
    buying_price = opening_prices * 0.99
    # 返回收益率的百分比
    if lowest_prices <= buying_price <= highest_prices:
        return (closing_prices - buying_price) *100 /buying_price
    return numpy.nan  #返回无效值,和任何值计算都是nan
#绘制收益线和均值线
result = numpy.vectorize(progit)(opening_prices, highest_prices, lowest_prices, closing_prices)
nan = ~numpy.isnan(result)
dates,profits = dates[nan],result[nan]

mp.plot(dates,profits,color="dodgerblue",linewidth=2,
        linestyle="-",label="profits")

mp.hlines(profits.mean(),dates[0],dates[-1],color="green",linewidth=2)
~~~

---

### 数组的裁剪与压缩

~~~python
ndarray.clip(min=下限,max=上限)#数组裁剪
ndarray.compress(条件)#数组的压缩

a = numpy.arange(1,10)
#数组裁剪:最小值不小于下限,最大值
print(a.clip(min=3,max=6))
print(a.compress(a>=6))
~~~

---

### 加法通用函数

~~~python
numpy.add(a,a)#求量数组之和
numpy.add.reduce(a)#求a数组元素的累加和
numpy.add.accumulate(a)#a数组累加和的过程
numpy.add.outer([10,20,30],a)#a数组内值和10,20,30依次相加
~~~

---

### 乘除法函数

~~~python
ndarray.prod()#累乘
ndarray.cumprod()#累乘过程
#a数组/b数组
numpy.true_divide(a,b)		numpy.divide(a,b)
#地板除法
numpy.floor_divide(a,b)		numpy.ceil(a/b)
numpy.round(a/b)	#四舍五入
numpy.trunc(a/b)	#截断取整
~~~

---

### 位运算通用函数

> 位异或【不同为1相同为0】:可以方便的判断两个数据是否同号

+ c = a^b
+ c = numpy.bitwise_xor(a,b)

~~~python
a = numpy.arange(1,10)
b = numpy.arange(11,20)
#判断a，b符号是否相等
print(a^b < 0)
print(numpy.bitwise_xor(a,b) < 0)
~~~

> ​	位与：计算某个数字是否是2的幂

~~~python
e = a & b
e = np.bitwise_and(a,b)
#判断a是否是2的幂，使用a&(a-1)
for i in range(1000):
    if i & (i-1) == 0:
        print(i,end=" ")
a = numpy.arange(1,1000)
print("\n",a[a&(a-1) == 0])
~~~

> 其他位运算通用函数

~~~python
numpy.bitwise_or(a,b)  	#或运算
numpy.bitwise_not(a,b) 	#非
numpy.left_shift(a,1)	#左移
numpy.right_shift(a,1)	#右移
~~~

---

## 数据平滑

> 数据的平滑处理通常包含有降噪、拟合等操作。降噪的功能在于去除额外的影响因素。拟合的目的在于数据的数学模型化，可以通过更多的数学工具识别曲线特征。

**绘制2只股票的收益率曲线**

> 收益率=（后一天收盘价-前一个天收盘价）/前一天收盘价

~~~python
# diff[1,2,4,8,2] >>>[1,2,4,-6]后一个-前一个的值,得出的数组比原来的少一个
bhp_returns = numpy.diff(bhp_closing_prices)/bhp_closing_prices[:-1]
vale_returns = numpy.diff(vale_closing_prices)/vale_closing_prices[:-1]
dates = dates[:-1]

mp.plot(dates,bhp_returns,c="dodgerblue",label="bhp_returns",
        linewidth = 1,alpha=0.5)
mp.plot(dates,vale_returns,c="orangered",label="vale_returns",
        linewidth = 1,alpha=0.5)
#卷积降噪
N = 8
core = numpy.array([1 / N for i in range(N)])
print(core)
bhp_returns = numpy.convolve(bhp_returns,core,"valid")
mp.plot(dates[N-1:],bhp_returns,c="blue",label="c_bhp_returns",
        linewidth = 2)
vale_returns = numpy.convolve(vale_returns,core,"valid")
mp.plot(dates[N-1:],vale_returns,c="red",label="c_vale_returns",
        linewidth = 2)
~~~

---

## 矩阵

> 是matrix类型的对象，该类继承自numpy.ndarray.作为子类，矩阵又结合了其自身的特点，做了必要的扩充。比如：矩阵乘法/矩阵求逆等运算

### 矩阵对象的创建

+ m = numpy.matrix(ary,copy = True)
  + ary：任何可以被理解为矩阵的数据结构
  + copy：是否复制数据（True：复制一份副本数据）
+ numpy.mat(ary)
+ numpy.mat(“矩阵拼块规则字符串”)

~~~python
ary = numpy.arange(1,10).reshape(3,3)
m = numpy.matrix(ary,copy = True)#默认为True
m[0,0] = 999
print(m,ary,sep="\n")
# 相当于matrix中copy=False
m2 = numpy.mat(ary)
m2[0,0] = 888
print(ary)

m3 = numpy.mat('1 2 3;4 5 6')
print(m3)
~~~

---

### 矩阵乘法

> ​	元素挨个与另外个数组的列进行相乘

~~~python
m3 = np.mat('1 2 3;4 5 6')
print(m3 * 10)
print(m3 * m3.T)
print(m3.T * m3)
~~~

---

### 矩阵的逆矩阵

> 若两个矩阵A、B满足：AB=BA=E(E为单位矩阵)。则称A与B互为逆矩阵。
>
> 单位矩阵E:主对角线为1，其他元素都为0.

~~~python
mi = m.I
mi = numpy.linalg.inv(m)
~~~

> 若把方阵推广到非方阵，则称为矩阵的***广义矩阵***

---

### 矩阵的应用

~~~python
prices = numpy.mat("3 3.2;3.5 3.6")
totals = numpy.mat("118.4;135.2")
#拟合解,只获取一个最匹配的解,而不是多个解
x = numpy.linalg.lstsq(prices,totals)[0]
print(prices.I)

x = prices.I * totals
print(x)
~~~

**解方程组**
$$
\begin{cases}
x-2y+z=0\\
2y-8z-8=0\\
-4x+5y+9z+9=0\\
\end{cases}
$$

+ x = numpy.linalg.lstsq(A,B)#误差最小解
+ x = numpy.linalg.solve(A,B)#精确解，没有则报错

~~~python
A = numpy.mat("1 -2 1;0 2 -8;-4 5 9")
B = numpy.mat("0;8;-9")
x = numpy.linalg.lstsq(A,B)[0]
print("误差最小值:",x)
x = numpy.linalg.solve(A,B)
print("精确解,没有则报错:",x)
~~~

**求斐波那契数列**

~~~python
m = numpy.mat("1 1;1 0")
for i in range(1,30):
    print((m**i)[0,1])
~~~

---

## 特征值与特征向量

> 对于n阶方阵A，如果存在数a和非零n维列向量x,使得Ax=ax,则称a是矩阵A的一个特征值，x是矩阵A属于特征值的特性向量。

+ 提取方阵A的特征值与特征向量
  + eigvals,eigvecs = numpy.linalg.eig(A)
    + eigvals:一组特征值
    + eigvecs:特征值对应的特征向量
+ 通过特征值与特征向量，逆向推导原方阵
  + S = numpy.mat(eigvecs) * numpy.mat(numpy.diag(eigvals)) *numpy.mat(eigvecs).I

~~~python
A = numpy.mat("1 5 8;2 5 7;8 2 4")
eigvals, eigvecs = numpy.linalg.eig(A)
print(eigvals, eigvecs, sep="\n")
print(type(eigvals), type(eigvecs), sep="\n")
S = numpy.mat(eigvecs) * numpy.mat(numpy.diag(eigvals)) \
    *numpy.mat(eigvecs).I#diag是让数组内原数组对角线显示
print(S)
#抹掉最后一个特征值
eigvals[2:] = 0
S = numpy.mat(eigvecs) * numpy.mat(numpy.diag(eigvals)) \
    *numpy.mat(eigvecs).I
print(S)
~~~

**示例：读取图片的亮度矩阵，提取特征值与特征向量，保留部分特征值，重新生成新的亮度矩阵绘制图片。**

~~~python
import numpy
# 导入处理图片的模块
import scipy.misc as sm
import matplotlib.pyplot as mp

image = sm.imread("lily.jpg",True)
print(type(image),image.shape)

#提取image矩阵的特征值与特征向量
eigvals,eigvecs = numpy.linalg.eig(image)
# 抹掉一部分特征值 重新生成新的图片
print(eigvals.shape)
eigvals[50:] = 0
S = numpy.mat(eigvecs) * numpy.mat(numpy.diag(eigvals)) * numpy.mat(eigvecs).I
S = S.real

mp.figure("Image Features",facecolor="lightgray")
mp.subplot(2,2,1)
mp.xticks([])
mp.yticks([])
mp.imshow(image,cmap="gray")

mp.subplot(2,2,2)
mp.xticks([])
mp.yticks([])
mp.imshow(S,cmap="gray")

mp.tight_layout()
mp.show()
~~~

---

## 奇异值分解

> 有一个矩阵M，可以分解为3个矩阵U S V,使得UxSxV等于M。U和V都是正交矩阵（乘以自身的转置矩阵结果为单位矩阵）。那么S矩阵主对角线上的元素称为矩阵M的奇异值，其他元素都为0.

+ 奇异值分解：U,s,V = numpy.linalg.svd(M)
  + U和V都是正交矩阵
  + s奇异值数组
+ 逆向推导原矩阵： S = U * numpy.diag(s) * V

**示例：读取图片的亮度矩阵，提取奇异值，保留部分奇异值，重新生成图片。**

~~~python
U,s,V = numpy.linalg.svd(image)
s[50:] = 0#保留100基本接近原图
S = numpy.mat(U) * numpy.mat(numpy.diag(s)) * numpy.mat(V)

mp.subplot(2,2,3)
mp.xticks([])
mp.yticks([])
mp.imshow(S,cmap="gray")
~~~

---

## 傅里叶

### 傅里叶定理

> 法国科学家傅里叶说过：任何一个周期函数都是n个不同振幅/不同频率/不同相位的正弦函数叠加而成。

$$
y = 4\pi \times sin(x) \\
y = \frac{4}{3}\pi \times sin(3x)\\
y = \frac{4}{5}\pi \times sin(5x)\\
...\\
y = \frac{4}{2n-1}\pi \times sin((2n-1)x)\\
$$

~~~python
import numpy
import matplotlib.pyplot as mp

x = numpy.linspace(-2*numpy.pi,2*numpy.pi,1000)
y = numpy.zeros(x.size)
y1 = 4*numpy.pi * numpy.sin(x)
y2 = 4/3 * numpy.pi * numpy.sin(3*x)
y3 = 4/5 * numpy.pi * numpy.sin(5*x)

#叠加1000条曲线
y = numpy.zeros(x.size)
for i in range(1,1001):
    y += 4/(2*i-1)*numpy.pi * numpy.sin((2*i-1)*x)

mp.grid(linestyle=":")
mp.plot(x,y,linewidth = 3)
mp.plot(x,y1)
mp.plot(x,y2)
mp.plot(x,y3)
mp.show()
~~~

---

### 快速傅里叶变换（FFT）

> 定义：把非常复杂的周期曲线拆解成一组光滑正弦曲线的过程。
>
> 目的：将时间域（时域）的信号转变为频域（频率域）上的信号，随着域的不同，对同一个事物的了解角度也随之改变。因此在时域中某些不好处理的地方，放在频域中就可以较为简单的处理。这样可以大量减少处理的数据量。
>
> import numpy.fft as nf

傅里叶定理：
$$
y = A_1sin(\omega_1x+\phi_1)+
A_2sin(\omega_2x+\phi_2)+..+C
$$

+ nf.fftfreq(par1，par2)
  + 参数1：采样数量
  + 参数2：采样周期（x轴相邻两点的距离）
    + 返回值：fft分解所得曲线的频率序列即x
+ nf.fft(原目标函数值序列)
  + 返回值：复数序列
    + 长度即是拆解出正弦函数的个数
    + 每个元素的模，代表正弦曲线的振幅，即y
    + 每个元素的辅角，代表每个正弦曲线的相位角
+ 逆向傅里叶变换
  + 原函数值序列 = nf.ifft(复数序列)

~~~python
import numpy
import numpy.fft as nf
import matplotlib.pyplot as mp

x = numpy.linspace(-2*numpy.pi,2*numpy.pi,1000)
y = numpy.zeros(x.size)

#叠加1000条曲线
y = numpy.zeros(x.size)
for i in range(1,1001):
    y += 4/(2*i-1)*numpy.pi * numpy.sin((2*i-1)*x)

#对y做傅里叶变换,绘制屏率图像
ffts = nf.fft(y)
#获取傅里叶交换的频率序列
freqs = nf.fftfreq(x.size,x[1] - x[0])
pows = numpy.abs(ffts) #获得元素的模

#通过复数数组,经过ifft操作,得到原函数
y2 = nf.ifft(ffts)

mp.figure("FFT",facecolor="lightgray")
mp.subplot(121)
mp.grid(linestyle=":")
mp.plot(x,y,linewidth=2)

mp.figure("FFT",facecolor="lightgray")
mp.subplot(121)
mp.grid(linestyle=":")
mp.plot(x,y2,linewidth=7,alpha=0.5)

mp.subplot(122)
mp.grid(linestyle=":")
mp.plot(freqs[freqs>0],pows[freqs>0],linewidth = 3,c = "orangered")

mp.show()
~~~

**基于傅里叶变换的频域滤波**

> 含噪信号是高能信号与低能噪声叠加的信号，可以通过傅里叶变换的频域滤波实现简单降噪。
>
> 通过FFT使含噪信号转换为含噪频谱，手动取出低能噪声，留下高能频谱后，在通过IFFT生成高能信号。

~~~python
import numpy
import matplotlib.pyplot as mp
import numpy.fft as nf
import scipy.io.wavfile as wf

# 1-采样率/采用值(声音的位移值)
sample_rate,noised_sigs = wf.read("noised.wav")
times = numpy.arange(len(noised_sigs)) /sample_rate

mp.figure("Filter",facecolor="lightgray")
mp.subplot(221)
mp.title("Time Domain",fontsize=16)
mp.ylabel("Signal",fontsize=12)
mp.tick_params(labelsize=8)
mp.grid(linestyle=":")
# 绘制音频的时域:时间/位移图像
mp.plot(times[:178],noised_sigs[:178],c="dodgerblue",label="Noised Sigs")

#2-基于傅里叶变换,获取音频

freqs = nf.fftfreq(times.size,1/sample_rate)
noised_ffts = nf.fft(noised_sigs)
noised_pows = numpy.abs(noised_ffts)

mp.subplot(222)
mp.title("Frequency Domain",fontsize=16)
mp.ylabel("Power",fontsize=12)
mp.tick_params(labelsize=8)
mp.grid(linestyle=":")
# 绘制音频的时域:时间/位移图像
mp.semilogy(freqs[freqs>0],noised_pows[freqs>0],c="red",label="Noised")

#3-将低频噪声去除后绘制音频领域:屏率/能量图像
fund_freq = freqs[noised_pows.argmax()]
# 找到所有噪声的下标
noised_inds = numpy.where(freqs != fund_freq)
filter_ffts = noised_ffts.copy()
filter_ffts[noised_inds] = 0
filter_pows = numpy.abs(filter_ffts)

mp.subplot(224)
mp.title("Frequency Domain", fontsize=16)
mp.ylabel("Power", fontsize=12)
mp.tick_params(labelsize=8)
mp.grid(linestyle=":")
# 绘制音频的时域:时间/位移图像
mp.plot(freqs[freqs > 0], noised_pows[freqs > 0], c="red", label="Filter")

# 4-基于逆向弗里叶变换,生成时域的音频信号,绘制时域:时间/位移图像.
filter_sigs = nf.ifft(filter_ffts)

mp.subplot(223)
mp.title("Time Domain", fontsize=16)
mp.ylabel("Signal", fontsize=12)
mp.tick_params(labelsize=8)
mp.grid(linestyle=":")
# 绘制音频的时域:时间/位移图像
mp.plot(times[:178], filter_sigs[:178], c="red", label="Filter Signs")

#重新生成音频文件
wf.write("filter.wav",sample_rate,filter_sigs.astype(numpy.int16))

mp.legend()
mp.show()
~~~

---

## 随机数模块

> 生成服从特定统计规律的随机数序列

### 二项分布binomial

> 重复n次独立试验的伯努利试验。每次试验只有两种可能的结果，而且两种结果发生与否相互独立相互对立。事件发生与否的概率在每一次独立实验中都保持不变。

+ 产生size个随机数：numpy.random.binomial(n,p,size)
  + n次尝试中的成功的次数
  + 其中每次尝试成功的概率为p
  + size尝试次数

~~~python
# 投篮命中率0.3,投10次,进5个的概率
r = numpy.random.binomial(10,0.3,10000)
p = (r == 5).sum() / r.size
print(p)
~~~

---

### 超几何分布hypergeometric

+ numpy.random.hypergeometric(ngood,nbad,nsample,size)
  + 总样本由ngood,nbad好坏样本组成
  + nsample：多少个样本中，好样本的数量
  + 返回值：size个随机数，每个随机数为在总样本中随机抽取

~~~python
#7个好苹果，3个坏苹果，抽出3个苹果，求抽出2个好苹果的概率
r = numpy.random.hypergeometric(7,3,3,1000)
result = (r ==2).sum() / r.size
print(result)
~~~

---

### 正态分布normal

+ numpy.random.normal(size)
  + 返回值：产生符合标准正态发布的size个随机数
    + 标准正态分布：期望=0，标准差=1
+ numpy.random.normal(loc=1,scale=10,size)
  + loc：期望值
  + scale：标准差
  + 返回值同上

---

## 其他功能

+ **排序**：numpy.msort(array)
+ **联合间接排序**（若主序列排序值相同，则用参考序列排序）：
  + numpy.lexsort((参考序列，主序列))

~~~python
#为商品排序。价格->销量
p = numpy.array(["A","B","C","D","E"])
prices = [8888,4500,3999,2999,3999]
totals = [100,200,99,110,80]
indices = numpy.lexsort((totals,prices))
print(p[indices])
~~~

+ **为复数数组排序**：numpy.sort_complex(复数数组)
  + 按照实部排序，对于实部相同的复数则按照虚部排列
  + 返回值：结果数组
+ **插入排序**：向有序数组中插入元素，使得数组依然有序
  + indices = numpy.searchsorted(有序序列，待插入序列)
    + 返回值：位置序列
  + numpy.insert(原数组，位置序列，待插入序列)

~~~python
a = numpy.array([1,2,4,6,8])
b = numpy.array([3,5])
indices = numpy.searchsorted(a,b)
print(indices)
result = numpy.insert(a,indices,b)
print(result)
~~~

+ **插值**

  > scipy提供了常见的插值算法，可以通过样本数据的一定的规律生成插值器函数。若我们给出函数的自变量，将会得到函数值。
  >
  > 插值器可以通过一组离散的样本数据得到一个连续的函数值，从而可以计算样本中不包含的自变量所对应的函数值。
  + si.interp1d(x,y,kind)
    + x：离散点的水平坐标
    + y：离散点的垂直坐标
    + kind：插值算法，默认：linear线型插入

~~~python
import numpy
import matplotlib.pyplot as mp
import scipy.interpolate as si

min_x,max_x = -50,50
dis_x = numpy.linspace(min_x,max_x,15)
dis_y = numpy.sinc(dis_x)

mp.figure("interpolate",facecolor="lightgray")
mp.title("Interpolate",fontsize=16)
mp.grid(linestyle=":")
mp.scatter(dis_x,dis_y,marker="D",c="red",label="Points")

#通过这组散点,构建线性插值器函数
linear = si.interp1d(dis_x,dis_y)
lin_x = numpy.linspace(min_x,max_x,1000)
lin_y = linear(lin_x)
mp.plot(lin_x,lin_y,c="dodgerblue",label="Linear Interpolate")

#通过这组散点,构建三次样条插值器函数
cubic = si.interp1d(dis_x,dis_y,kind="cubic")
lin_x = numpy.linspace(min_x,max_x,1000)
lin_y = cubic(lin_x)
mp.plot(lin_x,lin_y,c="red",label="Cubic Interpolate")

mp.legend()
mp.show()
~~~

+ **积分**

~~~python
import scipy.interpolate as si
def f(x):return 2 * x **2 + 3 * x + 4
#area：定积分的结果
area = si.quad(f,-5,5)[0]#f：函数名	-5:积分下限	 5:积分上限
~~~

+ **简单图像处理**
  + scipy.ndimage提供了一些简单的图像处理。如：高斯模糊，图像旋转，边缘识别等功能

~~~python
import numpy
import matplotlib.pyplot as mp
import scipy.misc as sm
import scipy.ndimage as sn

image = sm.imread("lily.jpg",True)
image2 = sn.median_filter(image,20)# 高斯模糊
image3 = sn.rotate(image,45)#角度旋转
image4 = sn.prewitt(image)#边缘识别

mp.figure("Ndimage")
mp.subplot(221)
mp.imshow(image,cmap="gray")
mp.axis("off")
mp.subplot(222)
mp.imshow(image2,cmap="gray")
mp.axis("off")
mp.subplot(223)
mp.imshow(image3,cmap="gray")
mp.axis("off")
mp.subplot(224)
mp.imshow(image4,cmap="gray")
mp.axis("off")

mp.show()
~~~

---

# Pandas

> **名称**：Python Data Analysis Library 或 pandas       

> **介绍**

+ 是基于NumPy 的一种工具，是为了解决数据分析任务而创建的。Pandas 纳入 了大量库和一些标准的数据模型，提供了高效地操作大型数据集所需的工具。                

> **定义**

+ 提供了大量能使我们快速便捷地处理数据的函数和方法。

+ 是python里面分析结构化数据的工具集，基础是numpy，图像库是matplotlib

> **数据类型**

+ **结构化数据**：是数据的数据库(即行数据,存储在数据库里,可以用二维表结构来逻辑表达实现 的数据) 

+ **非结构化数据**：包括所有格式的办公文档、文本、图片、HTML、各类报表、图像和音频/视 频信息等等 
+ **半结构化数据**： 所谓半结构化数据，就是介于完全结构化数据（如关系型数据库、面向对象数据库中的数据） 和完全无结构的数据（如声音、图像文件等）之间的数据，XML、json就属于半结构化数据。 它一般是自描述的，数据的结构和内容混在一起，没有明显的区分。 

## 数据结构

> 当你有几千几万个数据点时，每一个存放数据点的位置之间的排列关系就是 数据结构。

- [ ] 数据结构是计算机==存储、组织数据==的方式。

+ 数据结构是指相互之间存在一种或多种特定关系的数据元素的集合。
+ 通常情况下，精心选择的 数据结构可以带来更高的运行或者存储效率。
+ 数据结构往往同高效的检索算法和索引技术有 关。

###  Series 

> Series可以理解为一个一维的数组，只是index可以自己改动。类似于定长的有序字典，有Index和 value。

~~~python
import pandas as pd
# 创建Series
s1=pd.Series(data=[3,-5,7,4],index=list('ABCD'))
s1=pd.Series(data=numpy.array([3,-5,7,4]),index=numpy.array(["A","B","C","D"]))
print(s1)
~~~



---

# 机器学习

> 是一门能够让编程计算机从数据中**学习**的计算机科学
>
> 一个计算程序在完成任务T之后，获得经验E，表现效果为P。如果任务T的性能表现（衡量效果P的标准）随着E的增加而增加。那么这样的计算机程序就被称为机器学习程序。

**为什么需要机器学习**

> 自动化升级与维护
>
> 解决哪些算法过于复杂，甚至根本没有已知算法的问题。
>
> 在机器学习的过程中协助人类对未知事物的洞察。

**机器学习的问题**

+ 建模问题

  > 在数据对象中通过统计或推理等方法，寻找一个接收特定输入x,并给出预期输出y的功能函数f。y=f(x)

+ 评估问题

  > 针对已知的输入，函数给出的输出（预测值）与实际输出（目标值）之间存在一定的误差，因此需要构建一个评估体系，根据误差的大小判定函数的优势。

+ 优化问题

  > 学习的核心在于改善模型性能，通过数据对算法的反复锤炼，不断提升函数预测的准确性，直到获得能够满足业务要求的最优解，这个过程就是机器学习过程。

**机器学习的种类**

+ 有监督学习

  > 用已知输出评估模型的性能。

+ 无监督学习

  > 在没有已知输出的情况下，仅仅根据输入信息的相关性，进行类别的划分。

+ 半监督学习

  > 先通过无监督学习划分类别，再根据人工标记，通过有监督学习预测输出。

+ 强化学习

  > 通过对不同决策结果的奖励和惩罚，使机器学习系统在经过足够长时间的训练之后，越来越倾向于期望的结果。

+ 批量学习

  > 将学习的过程和应用的过程截然分开，用全部的训练数据训练模型，然后再应用场景中实现预测，当预测结果不够理想时，重新回到学习过程，如此循环

+ 增量学习

  > 将学习的过程和应用的过程统一起来，在应用的同时以增量的方式，不断学习新内容，边训练边预测。

+ 基于实例的学习（概率论）

  > 根据以往的经验，寻找与待预测输入最接近的样本，以其输出作为预测结果

+ 基于模型的学习（线性关系）

  > 根据以往的经验，建立用于联系输出和输入的某种数学模型，将带预测的输入带入该模型，预测其结果。

**机器学习的一般过程**

+ 数据处理
  + 数据收集（数据检索/数据挖掘/爬虫）
  + 数据清洗
+ 机器学习
  + 选择模型（算法）
  + 训练模型（算法）
  + 评估/优化模型（工具、框架、算法知识）
  + 测试模型
+ 业务运维
  + 应用模型
  + 维护模型

**机器学习的典型应用**

> 股价预测
>
> 推荐引擎
>
> 自然语言识别
>
> 语音识别
>
> 图像识别
>
> 人脸识别

**机器学习的基本问题**

+ 回归问题

  > 根据已知的输入和输出寻找某种性能最佳的模型，将未知输出的输入代入模型，得到连续的输出。

+ 分类问题

  > 根据已知的输入和输出寻找这种性能最佳的模型，将未知输出的输入代入模型，得到离散的输出。

+ 聚类问题

  > 根据已知输入的相似程度，将输入数据划分为不同的群落

+ 降维问题

  > 在性能损失竟可能小的前提下，降低数据的复杂度。

### 目录

==**回归算法**==

> 评估：r2.score()

线性回归

> 输入输出符合一定数学的线性规律，通过基于梯度下降的最小二乘法，找到最小损失函数值，来找到线性回归函数

岭回归

> 岭回归在模型迭代过程中增加了正则项，用来限制模型参数对异常样本的匹配程度，进而提高模型面对大多数正常样本的拟合精度。

多项式回归

> 过于简单的线性回归模型，欠拟合.可以把一元多项式函数看做多元线性方程

决策树回归

> 相似的输入必产生相似的输出

集合算法-正向激励

> 首先为样本矩阵中的样本随机分配初始权重，由此构建一颗带有权重的决策树，在由该决策树提供预测输出时，通过加权平均或加权投票的方式产生预测值。

集合算法-自助聚合

> 每次从总样本矩阵中以有放回抽的方式，随机抽取部分样本构建决策树，这样形成多颗包含不同训练本的决策树，以削弱某些强势样本对模型预测结果的影响。提高模型的泛华特性。

集合算法-随机森林

> 在自助聚合的基础上，每次构建决策树模型时，不仅随机选择部分样本，而且还随机选择部分**特征**（就相当于列）。

支持向量机 svm.SVR()

> 可基于核函数的升维变换，寻求最优分类边界

==**分类算法**==

> 评估：均值报告

简单分类

逻辑分类

> 多元性回归

朴素贝叶斯分类

> 整体样本分布情况知道的时候,可以一种依据统计理论而实现的一种分类方式. 观察一组数据: P(a,b)

决策树分类

支持向量机分类

> 分类较多

==**聚类算法**==

> 评估：轮廓系数

K均值聚类

> 随机选择K个样本作为K个聚类中心, 计算每个样本到各个聚类中心的欧式距离

均值漂移聚类

> 首先假定样本空间中的每个聚类均服从某种已知的概率分布规则, 然后用不同的概率密度函数拟合样本中的统计直方图, 不断移动密度函数的中心位置, 直到获得最佳拟合效果为止

凝聚层次聚类

> 首先假定每个样本都是一个独立的聚类, 如果统计出来的聚类数大于期望的聚类数, 则从每个样本出发, 寻找离自己最近的另外一个样本, 与之聚集, 形成更大的聚类. 同时另总聚类数减少, 不断重复以上过程, 直到统计出来的聚类总数达到期望值为止.

DBSCAN聚类

> 从样本空间中任意选择一个样本, 以事先给定的半径做圆. 凡是被该圆圈中的样本都视为与该样本处于同样的聚类.

## 数据预处理

> 数据预处理的过程：输入数据->模型->输出数据
>
> 通用数据样本矩阵结构：一行一样本，一列一特征

~~~python
import sklearn.preprocessing as sp
~~~

### 均值移除（标准化）

> 由于一个样本的不同特征差异较大，不利于使用现有的机器学习算法进行样本处理。均值移除可以让样本矩阵中的每一列的平均值为0，标准差为1.

~~~python 
例如有一列数据表示年龄：17 20 23
mean = 20
17-20 = -3  	20-20=0 	23-20=3
#使得这组数据的标准差为1
a = -3 b = 0 c = 3
s = std([a,b,c])
[a/s,b/s,c/s]
~~~

+ 均值移除：A = sp.scale(array)
  + array：一行一样本，一列一特征
  + 返回值：对array数组执行均值移除后的结果

~~~python
import sklearn.preprocessing as sp
A = sp.scale([17,20,23])
print(A)
~~~

---

### 范围缩放

> 将样本矩阵中的每一列的最小值和最大值设定为相同的区间，统一各列特征值的范围。一般情况下，会把特征区间缩放至[0,1]

~~~
[17,20,23]
如何使这组数据的最小值等于0：[0,3,6]
如何使这组数据的最大值等于1：[0,1/2,1]
~~~

~~~python
import numpy,math
import sklearn.preprocessing as sp
samples = numpy.array([
    [17.,100.,4000],
    [20.,80.,5000],
    [23.,70.,5500]])
#创建MinMaxScaler缩放器对象(0到1的范围)
mms = sp.MinMaxScaler(feature_range=(0,1))
#调用方法执行范围缩放
r_samples = mms.fit_transform(samples)
print(r_samples)

#基于他们之间的线性关系 实现范围缩放
linear_samples = samples.copy()
for col in linear_samples.T:
    #一个col就是原始样本数组中的一列
    col_min = col.min()
    col_max = col.max()
    A = numpy.array([ [col_min,1],
                      [col_max,1]])
    B = numpy.array([0,1])
    x = numpy.linalg.lstsq(A,B)[0]
    col *= x[0]
    col += x[1]
print(linear_samples)
~~~

---

### 归一化

> 有些情况每个样本的每个特征值具体值并不重要，但是每个样本特征值的占比更加重要。

|      | python | java | php  |
| ---- | ------ | ---- | ---- |
| 2017 | 10     | 20   | 8    |
| 2018 | 5      | 3    | 0    |
| 2019 | ...    | ...  | ...  |

> 归一化即是用每个样本的每个特征值除以该样本的各个特征值绝对值总和。变化后的样本矩阵每个样本的特征值绝对值之和为1.

**示例**

~~~python
#norm	范数
#l1 - l1范数	向量中每个元素绝对值之和
#l2 - l2范数	向量中每个元素平方之和
r = sp.normalize(array,norm="l1")

samples = numpy.array([
    [17.,100.,4000],
    [20.,80.,5000],
    [23.,70.,5500]])
r = sp.normalize(samples,norm="l1")
print(r[0])
~~~

---

### 二值化

> 有些业务并不需要分析矩阵详细完整的数据（例如：图像的边缘识别，只需要分析出边缘即可），可以根据事先给定的阈值，用0和1表示特征值不高于/高于阀值。二值化后，矩阵中每个元素非0即1，达到简化数学模型的目的。

~~~python
bin = sp.Binarizer(threshold=阈值)#获取二值化器对象
#基于二值化器转换原始样本矩阵
result = bin.transform(原始样本矩阵)

samples = numpy.array([
    [17.,100.,4000],
    [20.,80.,5000],
    [23.,70.,5500]])
bin = sp.Binarizer(threshold=80)
r_samples = bin.transform(samples)
print(r_samples)
~~~

---

### 独热编码(One-Hot)

为样本特征的每个值建立一个由一个1和若干个0组成的序列，用该序列对手游的特征值进行编码。

~~~
1 	3 	2
7 	5 	4
1	8	6
7	3	9
为每一个数字进行独热编码：
1-10	3-100	2-1000
7-01	5-010	4-0100
		8-001	6-0010
				9-0001
使用上诉码表，对原始矩阵编码过后的结果为：
101001000
010100100
100010010
011000001
~~~

~~~python
#获取独热编码器对象
ohe = sp.OneHotEncoder(sparse=是否采用紧凑格式)
#对原始样本矩阵进行处理
result = ohe.fit_transform(原始样本矩阵)
~~~

~~~python
#获取独热编码器对象
ohe = sp.OneHotEncoder(sparse=是否采用紧凑格式)
#对原始样本矩阵进行训练，得到编码字典
encode_dict = ohe.fit(原始样本矩阵)
#根据字典对样本矩阵执行独热编码
result = encode_dict.transform(样本矩阵)
~~~

**示例**

~~~python
samples = numpy.array([
    [17.,100.,4000],
    [20.,80.,5000],
    [23.,70.,5500]])

ohe = sp.OneHotEncoder(sparse=True,dtype=int)
r = ohe.fit_transform(samples)
print(r)
~~~

---

### 标签编码

> 根据字符串形式的特征值在特征序列中的位置，为其指定一个数字标签，用于提供给基于数值算法的学习模型。

~~~python
lbe = sp.LabelEncoder()				  #创建标签编码器
result = lbe.fit_transfrom(原始样本矩阵)#执行标签编码
#标签解码，通过result矩阵回推样本矩阵
样本矩阵 = lbe.invers_transform(result)

samples = numpy.array(['audi','ford','audi','toyota',
    'ford','bmw','toyota','ford','audi'])

lbe = sp.LabelEncoder()
r_samples = lbe.fit_transform(samples)
print(lbe.inverse_transform(r_samples))
~~~

---

## 模型选择

### 线性回归

~~~
输入			输出
0.5			  5.0
0.6			  5.5
0.8			  6.0
1.1			  6.8
1.4 		  7.0
...
y = f(x) 	f(x) = kx + b
~~~

+ **预测函数：y=**w<sub>0</sub>+w<sub>1</sub>**x**
  + x:输入	y：输出
  + w<sub>0</sub> w<sub>1</sub>:模型参数

> **模型训练：根据已知的x与y，找到最佳的模型参数w<sub>0</sub> w<sub>1</sub>，使得竟可能精确的描述出输入和输出的关系。**
>
> ****单样本误差：根据预测函数求出输入为x时的预测值：y‘=w<sub>0</sub>+w<sub>1</sub>x。单样本误差则为：1/2(y’-y)<sup>2</sup>
>
> **总样本误差：把所有的单样本误差相加即为总样本误差：**1/2 &Sigma; (y’-y)<sup>2</sup>
>
> **损失函数：**loss = 1/2 &Sigma; (w<sub>0</sub>+w<sub>1</sub>**x** -y)<sup>2</sup>
>
> 所以损失函数就是总样本误差关于模型参数w<sub>0</sub> w<sub>1</sub>的函数，该函数属于三维数学模型，即需要找到一组w<sub>0</sub> w<sub>1</sub>使得loss取最小值。

示例：**实现线性回归模型梯度下降过程**

~~~python
import numpy
import matplotlib.pyplot as mp
import mpl_toolkits.mplot3d as axes3d

train_x = numpy.array([0.5,0.6,0.8,1.1,1.4])
train_y = numpy.array([5.0,5.5,6.0,6.8,7.0])

times = 1000#存储梯度下降的次数
rate = 0.01#记录每次梯度下降模型参数标化率
epoches = [] #记录每次梯度下降的索引
w0,w1,losses = [1],[1],[]
#模拟梯度下降的过程
for i in range(1,times + 1):
    epoches.append(i)
    loss = (( (w0[-1] + w1[-1] * train_x) - train_y ) ** 2).sum() / 2
    losses.append(loss)
    # 把梯度下降的过程,进行输出
    print('{:4}> w0={:.8f},w1={:.8f},loss={:.8f}'.format(epoches[-1],w0[-1],w1[-1],losses[-1]))

    #根据偏倒数公式,求得w0与w1方向上的梯度值
    d0 = ( (w0[-1] + w1[-1] * train_x) - train_y ).sum()
    d1 = ( ( w0[-1] + w1[-1] * train_x  - train_y ) * train_x).sum()
    w0.append(w0[-1]-rate*d0)
    w1.append(w1[-1]-rate*d1)

#经过1000次下降,最终得到的w0与w1使得loss函数值接近最小
pred_y = w0[-1] + w1[-1] * train_x

#绘制训练数据
mp.figure("Linear Regression",facecolor="lightgray")
mp.title("Linear Regression",fontsize=16)
mp.xlabel("x")
mp.ylabel("y")
mp.tick_params(labelsize=10)
mp.grid(linestyle=":")
mp.scatter(train_x,train_y,marker="o",s = 60 ,
           c = "dodgerblue",label = "Train Points")

mp.plot(train_x,pred_y,c="red",linewidth=2,label = "Regression Line")
mp.legend()

#绘制随着每次梯度下降过程,w0 w1 loss函数的变化
mp.figure("Training Progress",facecolor="lightgray")
mp.subplot(311)
mp.title("Training w0",fontsize=14)
mp.ylabel("w0",fontsize=12)
mp.grid(linestyle=":")
mp.tick_params(labelsize=10)
mp.plot(epoches,w0[:-1],c = "dodgerblue",label="w0 Progress")

mp.subplot(312)
mp.title("Training w1",fontsize=14)
mp.ylabel("w1",fontsize=12)
mp.grid(linestyle=":")
mp.tick_params(labelsize=10)
mp.plot(epoches,w1[:-1],c = "dodgerblue",label="w1 Progress")

mp.subplot(313)
mp.title("Training loss",fontsize=14)
mp.ylabel("loss",fontsize=12)
mp.grid(linestyle=":")
mp.tick_params(labelsize=10)
mp.plot(epoches,losses,c = "dodgerblue",label="loss Progress")

#在三维曲面中绘制梯度下降的过程
grid_w0,grid_w1 = numpy.meshgrid(
    numpy.linspace(0,9,500),
    numpy.linspace(0,3.5,500))

grid_loss = numpy.zeros_like(grid_w0)
for x,y in zip(train_x,train_y):
    grid_loss += (grid_w0 + x*grid_w1 - y) ** 2 / 2

mp.figure("Loss Function")
ax3d = mp.gca(projection="3d")
ax3d.set_xlabel("w0",fontsize=14)
ax3d.set_ylabel("w1",fontsize=14)
ax3d.set_zlabel("loss",fontsize=14)
ax3d.plot_surface(grid_w0,grid_w1,grid_loss,
        rstride=10,cstride=10,cmap="jet")
ax3d.plot(w0[:-1],w1[:-1],losses,'o-',c="red",label="BGD")

#以等高线的方式绘制梯度下降的过程
mp.figure("Contour")
mp.title("BGD Contour",fontsize=16)
mp.xlabel("w0",fontsize=12)
mp.ylabel("w1",fontsize=12)
mp.tick_params(labelsize=10)
mp.contourf(grid_w0,grid_w1,grid_loss,10,cmap="jet")
cntr = mp.contour(grid_w0,grid_w1,grid_loss,10,color="black")
mp.clabel(cntr,inline_spacing=0.1,fmt="%.2f",fontsize=8)
mp.plot(w0,w1,"o-",c="red")
mp.tight_layout()

mp.legend()
mp.show()
~~~

**sklearn提供的线性回归相关API**

+ 获取线性回归模型：lm.LinearRegression()
+ 模拟训练：model.fit(输入集，输出集)
  + 输入集：x数据样本矩阵
  + 输出集：列向量
+ 获取预测输出：model.predict(输入样本)

~~~python
import numpy
import matplotlib.pyplot as mp
import sklearn.linear_model as lm

x,y = numpy.loadtxt("single.txt",delimiter=",",usecols=(0,1),unpack=True)

mp.figure("Linear Regression",facecolor="lightgray")
mp.title("Linear Regression",fontsize=18)
mp.xlabel("x",fontsize=16)
mp.ylabel("y",fontsize=16)
mp.tick_params(labelsize=12)
mp.grid(linestyle="-")
mp.scatter(x,y,s=60,c="dodgerblue",label="Points")

#训练模型
model = lm.LinearRegression()
# 把x改为n行1列,这样才可以作为输入交给模型训练
x = x.reshape(-1,1)
y = y.reshape(-1,1)
model.fit(x,y)
pred_y = model.predict(x)
mp.plot(x,pred_y,c="red",label="fPoints")

mp.legend()
mp.show()
~~~

### 岭回归

> 普通的线性回归模型使用基于梯度下降的最小二乘法，在最小化顺势函数的前提下，寻找最优模型参数。在此过程中，包括少数异常样本在内的全部训练数据都会对最终的模型参数造成相等程度的影响，并且异常值对模型所带来的影响无法再训练过程中被识别出来。为此，岭回归在模型迭代过程中增加了正则项，用来限制模型参数对异常样本的匹配程度，进而提高模型面对大多数正常样本的拟合精度。

+ lm.Ridge(正则强度,fit_intercept,max_iter=最大迭代次数)
  + fit_intercept： 是否训练截距（是否经过0,0点）
  + max_iter：最大迭代次数
+ 模拟训练：model.fit(输入集，输出集)
  - 输入集：x数据样本矩阵
  - 输出集：列向量
+ 获取预测输出：model.predict(输入样本)

~~~python
import numpy
import matplotlib.pyplot as mp
import sklearn.linear_model as lm

x,y = numpy.loadtxt("abnormal.txt",delimiter=",",usecols=(0,1),unpack=True)

mp.figure("Linear Regression",facecolor="lightgray")
mp.title("Linear Regression",fontsize=18)
mp.xlabel("x",fontsize=16)
mp.ylabel("y",fontsize=16)
mp.tick_params(labelsize=12)
mp.grid(linestyle="-")
mp.scatter(x,y,s=60,c="dodgerblue",label="Points")

x = x.reshape(-1,1)
y = y.reshape(-1,1)
model = lm.LinearRegression()
model.fit(x,y)
pred_y = model.predict(x)
mp.plot(x,pred_y,c="red",label="fPoints")

model = lm.Ridge(100,fit_intercept=True,max_iter=1000)
model.fit(x,y)
pred_y = model.predict(x)
mp.plot(x,pred_y,c="green",label="rPoints")

mp.legend(loc=2)
mp.show()
~~~

### 多项式回归 

> **一元多项式的一般形式：**

+ y=w<sub>0</sub>+w<sub>1</sub>x+w<sub>2</sub>x<sup>2</sup>+w<sub>3</sub>x<sup>3</sup>+...w<sub>d</sub>x<sup>d</sup>

> 把一元多项式函数看做多元线性方程：

+ y=w<sub>0</sub>+w<sub>1</sub>x<sub>1</sub>+w<sub>2</sub>x<sub>2</sub>+w<sub>3</sub>x<sub>3</sub>+...w<sub>d</sub>x<sub>d</sub>

> 所以一元多项式回归可以看做多元线性回归，可以使用LinearRegression模型对样本数据进行模拟训练。

1. 将一元多项式回归问题转换多元线性回归问题。（只需要给出最高次幂的值）
2. 将1步骤中得到结果中w<sub>1</sub> w<sub>2</sub>...当做样本特征，交给线性回归其训练多元线性模型。最终得到一组w<sub>0</sub> w<sub>1</sub>...使得损失函数接近最小值。

+ 通过管线（pipeline）把2个步骤连在一起执行：                   	              			model = pl.make_pipeline(par1,par2)
  + 参数1：多项式特征扩展器-sp.PolynomialFeatures(4)
  + 参数2：多元线性模型：lm.LinearRegression()

~~~python
import numpy
import matplotlib.pyplot as mp
import sklearn.linear_model as lm
import sklearn.pipeline as pl
import sklearn.preprocessing as sp

x,y = numpy.loadtxt("single.txt",delimiter=",",usecols=(0,1),unpack=True)

mp.figure("Linear Regression",facecolor="lightgray")
mp.title("Linear Regression",fontsize=18)
mp.xlabel("x",fontsize=16)
mp.ylabel("y",fontsize=16)
mp.tick_params(labelsize=12)
mp.grid(linestyle="-")
mp.scatter(x,y,s=60,c="dodgerblue",label="Points")

x = x.reshape(-1,1)
y = y.reshape(-1,1)
model = pl.make_pipeline(sp.PolynomialFeatures(4),
                         lm.LinearRegression())
model.fit(x,y)
#由于x无序,为了绘制多项式模拟曲线,构建1000个点
test_x = numpy.linspace(x.min(),x.max(),1000).reshape(-1,1)
pred_y = model.predict(test_x)
mp.plot(test_x,pred_y,c="red",label="fPoints")


mp.legend(loc=2)
mp.show()
~~~

> 过于简单的模型，无论对于训练数据还是测试数据都无法给出足够高的预测精度，这种现象称为欠拟合
>
> 过于复杂的模型，对于训练数据可以给出足够高的预测精度，但是对于测试数据精度反而很低，这种现象称为过拟合（模型训练过于依赖训练集）
>
> 所以一个性能可以接受的模型应该对训练集与测试集数据都有接近的预测精度。而且精度不能太低。

### 决策树

> 相似的输入必会产生相似的输出。

> 为了提高搜索效率，使用树形数据结构处理样本数据：
>
> 1.从训练样本矩阵中选择第一个特征进行子表划分，使每个子表中该特征的值全部相同。
>
> 2.再在每个子表中选择下一个特征，按照同样的规则继续划分为小的子表，不断重复直到所有的特征用完为止。
>
> 3.得到叶级子表，其中所有样本的特征值完全相同。
>
> 4.对于待预测样本，根据每一个特征值，选择对应的子表，逐一匹配，直到找到与之完全匹配的叶级子表，用该子表中样本的输出，通过平均（回归）或者投票（分类）为待预测样本提供输出。

~~~python
import sklearn.datasets as sd
import sklearn.utils as su
import sklearn.tree as st
import sklearn.metrics as sm
# 读取数据  加载波士顿房屋价格
boston = sd.load_boston()
print("数据的维度:", boston.data.shape)
print("数据的特征名:", boston.feature_names)
print(boston.target.shape)

# 划分测试及与训练集,以random_state随机种子作为参数打乱数据集[为了每次打乱数据,结果一致]
x, y = su.shuffle(boston.data, boston.target, random_state=7)
train_size = int(len(x) * 0.8)
train_x, test_x, train_y, test_y = x[:train_size], x[train_size:], y[:train_size], y[train_size:]

#创建决策树回归器模型,使用训练集训练模型
#测试集测试模型
model = st.DecisionTreeRegressor(max_depth=5)
model.fit(train_x,train_y)
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y,pred_test_y))

['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
 'B' 'LSTAT']
['犯罪率' '住宅用地比例' '商业率' '是否有河' '空气质量' '房间' '房龄' '离市中心距离' '路网密度' '房产税' '师生比[学区房]'
 '黑人比例' '低地位人口比例']
~~~

#### 工程优化

> 决策树在使用时不必用尽所有的特征，叶级子表中允许混杂不同的特征值，以此降低决策树的高度。所以在精度牺牲可接受的情况下，可以降低决策树的高度，提高模型的性能。
>
> 通常情况下，可以优先选择使**信息熵**减少量最大的特征作为划分子表的依据。
>
> 信息熵：信息混杂量

#### 集合算法

> 根据多个不同的模型给出的预测结果，利用平均（回归）或者投票（分类）的方法，得出最终的结果
>
> 基于决策树的集合算法，就是按照某种规则，构建多颗彼此不同的决策树模型，分别给出针对未知样本的预测结果，最后通过平均或投票的方式得到相对综合的结论。

##### 正向激励

> 首先为样本矩阵中的样本随机分配初始权重，由此构建一颗带有权重的决策树，在由该决策树提供预测输出时，通过加权平均或加权投票的方式产生预测值。将训练样本带入模型，预测其输出，针对那些预测值与实际值不同的样本，提高其权重，由此形成第二颗决策树。重复以上过程，构建出不同权重的若干颗决策树。最终使用时，由这些决策树通过平均、投票的方式得到相对综合的输出。

+ t_model = st.DecisionTreeRegressor(...)
+ se.AdaBoostRegressor(t_model ,n_estimators=400,random_state=7)
  + t_model ：正向激励内部的模型
  + n_estimators:构建400颗树
  + random_state：随机种子

~~~python
import sklearn.ensemble as se
t_model = st.DecisionTreeRegressor(max_depth=5)
model = se.AdaBoostRegressor(t_model,n_estimators=400,random_state=7)
model.fit(train_x,train_y)
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y,pred_test_y))
~~~

##### 自助聚合

> 每次从总样本矩阵中以有放回抽的方式，随机抽取部分样本构建决策树，这样形成多颗包含不同训练本的决策树，以削弱某些强势样本对模型预测结果的影响。提高模型的泛华特性。

##### 随机森林

> 在自助聚合的基础上，每次构建决策树模型时，不仅随机选择部分样本，而且还随机选择部分特征。这样的集合算法，不仅规避了强势样本对预测结果的影响，而且也削弱了强势特征的影响，而且也削弱了强势特征的影响，是模型的预测能力更加泛化。

+ 构建随机森林模型：model = se.RandomForestRegressor(max_depth=4,n_estimators=1000,min_samples_slit=2)
  + max_depth：最大深度
  + n_easimators：构建多少颗决策树
  + min_samples_slit：子表中最小样本数，若小于这个数字，则不再继续划分

**分享共享单车的需求，从而判断如何投放**

~~~python
import numpy as np
import sklearn.utils as su
import sklearn.ensemble as se
import sklearn.metrics as sm
import matplotlib.pyplot as mp

data = np.loadtxt('bike_day.csv',
	delimiter=',', unpack=False, 
	dtype='U20')
# 获取输入集与输出集
header = data[0, 2:13]
x = np.array(data[1:, 2:13], dtype=float)
y = np.array(data[1:, -1], dtype=float)
# 打乱数据集
x, y = su.shuffle(x, y, random_state=7)
# 拆分训练集,测试集
train_size = int(len(x)*0.9)
train_x, test_x, train_y, test_y = \
	x[:train_size], x[train_size:], \
	y[:train_size], y[train_size:]
# 随机森林模型训练
model=se.RandomForestRegressor(max_depth=10, 
	n_estimators=1000, min_samples_split=2)
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
# 使用r2得分验证预测结果
print(sm.r2_score(test_y, pred_test_y))
# 输出特征重要性
fi_day = model.feature_importances_
print(fi_day)
print(header)

# 绘制特征重要性柱状图
mp.figure('Bike', facecolor='lightgray')
mp.subplot(211)
mp.title('Day', fontsize=16)
mp.ylabel('Importances', fontsize=12)
mp.tick_params(labelsize=8)
mp.grid(linestyle=':')
pos = np.arange(fi_day.size)
sorted_i = fi_day.argsort()[::-1]
mp.xticks(pos, header[sorted_i])
mp.bar(pos, fi_day[sorted_i], 
	   color='dodgerblue', label='Bike_Day')
mp.legend()


data = np.loadtxt('../ml_data/bike_hour.csv', 
	delimiter=',', unpack=False, 
	dtype='U20')
# 获取输入集与输出集
header = data[0, 2:14]
x = np.array(data[1:, 2:14], dtype=float)
y = np.array(data[1:, -1], dtype=float)
# 打乱数据集
x, y = su.shuffle(x, y, random_state=7)
# 拆分训练集,测试集
train_size = int(len(x)*0.9)
train_x, test_x, train_y, test_y = \
	x[:train_size], x[train_size:], \
	y[:train_size], y[train_size:]
# 随机森林模型训练
model=se.RandomForestRegressor(max_depth=10, 
	n_estimators=1000, min_samples_split=2)
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
# 使用r2得分验证预测结果
print(sm.r2_score(test_y, pred_test_y))
# 输出特征重要性
fi_hour = model.feature_importances_
print(fi_hour)
print(header)

mp.subplot(212)
mp.title('Hour', fontsize=16)
mp.ylabel('Importances', fontsize=12)
mp.tick_params(labelsize=8)
mp.grid(linestyle=':')
pos = np.arange(fi_hour.size)
sorted_i = fi_hour.argsort()[::-1]
mp.xticks(pos, header[sorted_i])
mp.bar(pos, fi_hour[sorted_i], 
	   color='orangered', label='Bike Hour')
mp.legend()
mp.show()
~~~

#### 特征重要性

> 作为决策树模型训练过程中的副产品，根据每个特征划分子表前后信息熵的减少量得到该特征的重要程度，此即为该特征重要性指标。可以通过model.feature_importances_来获取每个特征的特征重要值。

~~~python
import matplotlib.pyplot as mp
import numpy
fi_ab = model.feature_importances_
x = numpy.arange(fi_ab.size)
fi_ab_sort = fi_ab.argsort()[::-1]
mp.xticks(x,boston.feature_names[fi_ab_sort])
mp.bar(x,fi_ab[fi_ab_sort])
~~~

### 支持向量机（SVM）

>**支持向量机原理**

1. **寻求最优分类边界** 

   正确:  对大部分样本可以正确的划分类别.

   泛化:  最大化支持向量间距.

   公平:  各类别与分类边界等距.

   简单:  基于线性模型, 直线/平面.

2. **基于核函数的升维变换**

   通过名为核函数的特征变换, 增加新的特征, 使得低维度空间中的线性不可分问题在高维度空间变得线性可分.

#### 支持向量机的使用
##### 线性核函数: linear  

> 不通过核函数进行维度提升, 仅在原始维度空间中寻求线性分类边界.

```python
import sklearn.svm as svm
model = svm.SVC(kernel='linear')
model.fit(x, y)
pred_test_y = model.predict(test_x)
```

案例: multiple2.txt 使用svm实现分类.

~~~python
x, y = [],[]
data=numpy.loadtxt('../ml_data/multiple2.txt',
                   delimiter=',')
x = data[:, :-1]
y = data[:, -1]
# 拆分训练集与测试集
train_x, test_x, train_y, test_y = \
	ms.train_test_split(x, y, test_size=0.25,
	random_state=5)
# 基于线性核函数的svm绘制分类边界
model=svm.SVC(kernel='linear')
model.fit(train_x, train_y)

# 绘制分类边界线
l, r = x[:, 0].min()-1, x[:, 0].max()+1
b, t = x[:, 1].min()-1, x[:, 1].max()+1
n = 500
grid_x, grid_y = numpy.meshgrid(
	numpy.linspace(l, r, n),
	numpy.linspace(b, t, n))
# 把grid_x与grid_y抻平了组成模型的输入,预测输出
mesh_x = numpy.column_stack(
	(grid_x.ravel(), grid_y.ravel()))
pred_mesh_y = model.predict(mesh_x)
grid_z = pred_mesh_y.reshape(grid_x.shape)

# 看一看测试集的分类报告
pred_test_y = model.predict(test_x)
cr = sm.classification_report(
	  	test_y, pred_test_y)
print(cr)
~~~
##### 多项式核函数: poly

> 通过多项式函数增加原始样本特征的高次方幂.

案例: 基于多项式核函数, 训练multiple2.txt

```python
# 基于多项式核函数的svm绘制分类边界
model=svm.SVC(kernel='poly', degree=3)
model.fit(train_x, train_y)
```

##### 径向基核函数: rbf

> 通过高斯分布函数增加原始样本特征的分布概率作为新的特征.

```python
# C: 正则项
# gamma: 正态分布曲线的标准差
model= svm.SVC(
    kernel='rbf', C=600, gamma=0.01)
```

案例:

```python
# 基于多项式核函数的svm绘制分类边界
model=svm.SVC(kernel='rbf', C=600, gamma=0.01)
model.fit(train_x, train_y)
```

#### 样本类别均衡化

> 通过样本类别权重的均衡化, 使所占比例较小的样本权重较高,而所占比例较大的样本权重较低, 以此平均化不同类别样本对分类模型的贡献, 提高模型预测性能.

> 当每个类别的样本容量相差较大时, 有可能会用到样本类别均衡化.

```python
model = svm.SVC(kernel='linear', 
                class_weight='balanced')
```

案例: imbalance.txt

```python
# 基于线性核函数的svm绘制分类边界
model=svm.SVC(kernel='linear', 
			  class_weight='balanced')
model.fit(train_x, train_y)
```

#### 置信概率

根据样本与分类边界距离的远近, 对其预测类别的可信程度进行量化, 离边界越近的样本, 置信概率越低, 反之则越高.

获取样本的置信概率:

```python
model = svm.SVC(....., probability=True)
pred_test_y = model.predict(test_x)
# 获取每个样本的置信概率
置信概率矩阵 = model.predict_proba(样本矩阵)
```

置信概率矩阵的结构:

|       | 类别1 | 类别2 |
| ----- | ----- | ----- |
| 样本1 | 0.8   | 0.2   |
| 样本2 | 0.3   | 0.7   |
| 样本3 | 0.4   | 0.6   |

案例: 

```python
# 整理测试样本 , 绘制每个样本的置信概率
prob_x = np.array([
	[2, 1.5], 
	[8, 9], 
	[4.8, 5.2], 
	[4, 4], 
	[2.5, 7], 
	[7.6, 2], 
	[5.4, 5.9]])
pred_prob_y = model.predict(prob_x)
probs = model.predict_proba(prob_x)
print(probs)
mp.scatter(prob_x[:,0], prob_x[:,1], s=60,
		marker='D', 
		c=pred_prob_y, label='prob points', 
		cmap='rainbow')
```

#### 网格搜索

> 网格搜索是一种选取最优超参数的解决方案.

> 获取一个最优超参数的方式可以绘制验证曲线, 但是验证曲线只能每次获取一个最优超参数. 如果多个超参数有很多排列组合情况的话, 可以选择使用网格搜索寻求最优超参数的组合.

网格搜索相关API:

```python
import sklearn.model_selection as ms
model = 决策树模型
# 返回分值最高的模型对象 
model = ms.GridSearchCV(
    	model, 超参数列表, cv=折叠数)
# 直接训练模型
model.fit(输入集, 输出集)
# 获取最好的超参数
model.best_params_
model.best_score_
model.best_estimator_
```

案例: 修改置信概率的案例.

```python
# 基于svm绘制分类边界
model=svm.SVC()
# 使用网格搜索,获取最优模型超参数
params = [
  {'kernel':['linear'],'C':[1,10,100,1000]},
  {'kernel':['poly'],'C':[1],'degree':[2,3]},
  {'kernel':['rbf'],'C':[1,10,100,1000], 
    'gamma':[1, 0.1, 0.01, 0.001]}]
model = ms.GridSearchCV(model, params, cv=5)
model.fit(train_x, train_y)

print(model.best_params_)
print(model.best_score_)
print(model.best_estimator_)
```

#### 案例: 事件预测

案例: events.txt  预测某个时间段是否会出现特殊事件.

```python
import numpy as np
import sklearn.preprocessing as sp
import sklearn.model_selection as ms
import sklearn.svm as svm

# 模仿LabelEncoder的接口 设计数字编码器
class DigitEncoder():

    def fit_transform(self, y):
        return y.astype(int)

    def transform(self, y):
    	return y.astype(int)

    def inverse_transform(self, y):
    	return y.astype(str)

data = np.loadtxt('events.txt',
       delimiter=',', dtype='U20',converters={i:lambda x:x.decode() for i in range(6)})
# 转置后删掉1列
data = np.delete(data.T, 1, axis=0)
# 整理训练集
encoders, x = [], []
for row in range(len(data)):
	# 选择编码器, 如果是数字,则用自定义编码器
	if data[row][0].isdigit():
		encoder = DigitEncoder()
	else:
		encoder = sp.LabelEncoder()

	encoders.append(encoder)

	if row < len(data)-1:
		x.append(
    		encoder.fit_transform(data[row]))
	else:
		y = encoder.fit_transform(data[row])

x = np.array(x).T
# 拆分训练集与测试集
train_x, test_x, train_y, test_y = \
    ms.train_test_split(x, y, test_size=0.25,
	random_state=7)
# 选择模型, 开始训练
model = svm.SVC(kernel='rbf',
			    class_weight='balanced')
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
# 输出精确度
print((pred_test_y==test_y).sum()/test_y.size)


# 模拟接收参数: 预测输入  从而预测输出
data = [['Monday','13:30:00','22','31']]
data = np.array(data).T
x = []
for row in range(len(data)):
	encoder = encoders[row]
	x.append(encoder.transform(data[row]))
x = np.array(x).T
pred_y = model.predict(x)
print(encoders[-1].inverse_transform(pred_y))
```

---

#### 案例: 交通流量预测(回归)

> 支持向量机也可以做做多元分类，回归业务

~~~python
import numpy
import sklearn.preprocessing as sp
import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.metrics as sm

class DigitEncoder():

    def fit_transform(self, y):
        return y.astype(int)

    def transform(self, y):
    	return y.astype(int)

    def inverse_transform(self, y):
    	return y.astype(str)

data = numpy.loadtxt('traffic.txt',
					 delimiter=',', dtype='U20', converters={i:lambda x:x.decode() for i in range(5)}).T
# 整理训练集
encoders, x = [], []
for row in range(len(data)):
	# 选择编码器, 如果是数字,则用自定义编码器
	encoder = DigitEncoder() if data[row][0].isdigit() \
 				else sp.LabelEncoder()

	encoders.append(encoder)

	if row < len(data)-1:
		x.append(
    		encoder.fit_transform(data[row]))
	else:
		y = encoder.fit_transform(data[row])


x = numpy.array(x).T
# 拆分训练集与测试集
train_x, test_x, train_y, test_y = \
    ms.train_test_split(x, y, test_size=0.25,
	random_state=7)
# 选择模型, 开始训练
model = svm.SVR(kernel='rbf',C = 10)
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
# 输出精确度
print(sm.r2_score(test_y,pred_test_y))

# 结果预测
data=[['Tuesday','13:35','San Francisco','yes']]
data = numpy.array(data).T

x = [encoders[row].transform(data[row]) for row in range(len(data))]

x = numpy.array(x).T
pred_y = model.predict(x)
print(int(pred_y))
~~~

---

### 简单分类

> 输出结果只有几类，如0或1

#### 人工分类

| 特征1 | 特征2 | 输出 |
| ----- | ----- | ---- |
| 3     | 1     | 0    |
| 2     | 5     | 1    |
| 1     | 8     | 1    |
| 6     | 4     | 0    |
| 5     | 2     | 0    |
| 3     | 5     | 1    |
| 4     | 7     | 1    |
| 6     | 8     | ？   |

~~~python
import numpy as np
import matplotlib.pyplot as mp
x = np.array([
	[3, 1],
	[2, 5],
	[1, 8],
	[6, 4],
	[5, 2],
	[3, 5],
	[4, 7],
	[4, -1]])
y = np.array([0, 1, 1, 0, 0, 1, 1, 0])
# 把样本绘制出来
mp.figure('Simple Classification', facecolor='lightgray')
mp.title('Simple Classification', fontsize=16)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')

# 绘制分类边界线
l, r = x[:, 0].min()-1, x[:, 0].max()+1
b, t = x[:, 1].min()-1, x[:, 1].max()+1
n = 500
grid_x, grid_y = np.meshgrid(
	np.linspace(l, r, n),
	np.linspace(b, t, n))
grid_z = np.piecewise(grid_x, 
	[grid_x >= grid_y, grid_y > grid_x], 
	[0, 1])
mp.pcolormesh(grid_x,grid_y,grid_z,cmap='gray')

mp.scatter(x[:, 0], x[:, 1], s=60, c=y,
	marker='o', label='Points', cmap='jet')

mp.legend()
mp.show()
~~~

#### 逻辑分类 

> 通过输入的样本数据，基于多元线性回归模型求出线性预测方程：

+ y = w<sub>0</sub>+w<sub>1</sub>x<sub>1</sub>+w<sub>2</sub>x<sub>2</sub>

> 基于损失函数最小化做梯度下降后，得到最优的模型参数： w<sub>0</sub> w<sub>1</sub> w<sub>2</sub>

​		基于损失函数最小化做梯度下降后,得到最优的模型参数: w<sub>0</sub>  w<sub>1</sub> w<sub>2</sub> .  通过得到的线性回归方程进行类别预测, 把x1与x2带入方程得到最终"类别".  但是方程返回的结果是连续值, 不可以直接用于分类业务模型,  所以急需一种方式实现: 连续的预测值 -> 离散的预测值 之间的一个转换. [-∞, ∞] -> {0, 1}
$$
逻辑函数 sigmoid: y= \frac{1}{1+e^{-x}}
$$
该逻辑函数当x>0时, y>0.5;  当x<0时, y<0.5; 可以把样本数据经过线性预测模型求得的值带入逻辑函数的x, 这样的话就可以逻辑函数的返回值看做预测输出被划分为1类别的概率.择概率大的类别作为预测结果. 可以根据该逻辑函数划分2个类别.  

这也是线性函数离散化的一种方式.

逻辑函数的相关API:

```python
import sklearn.linear_model as lm
model = lm.LogisticRegression(
    solver='liblinear', C=正则强度)
model.fit(x, y)
pred_test_y = model.predict(test_x)
```

~~~python
# 构建逻辑分类器
model = lm.LogisticRegression(
	solver='liblinear', C=10)
model.fit(x, y)
# 把grid_x与grid_y抻平了组成模型的输入,预测输出
test_x = np.column_stack(
	(grid_x.ravel(), grid_y.ravel()))
pred_test_y = model.predict(test_x)
grid_z = pred_test_y.reshape(grid_x.shape)
~~~

##### 多元分类

通过多个二元分类器解决多元分类问题.

| 特征1 | 特征2 | ==>  | 所属类别 |
| ----- | ----- | ---- | -------- |
| 4     | 7     | ==>  | A        |
| 3.5   | 8     | ==>  | A        |
| 1.2   | 1.9   | ==>  | B        |
| 5.4   | 2.2   | ==>  | C        |

若拿到一组新的样本, 可以基于二元逻辑分类训练出一个模型, 判断属于A类别的概率. 再基于同样的方法训练处两个模型,分别判断属于B类别/ 属于C类别的概率, 最终选择概率最高的作为新样本的分类结果.

~~~python
import numpy as np
import matplotlib.pyplot as mp
import sklearn.linear_model as lm
x = np.array([
	[4, 7],
	[3.5, 8],
	[3.1, 6.2],
	[0.5, 1],
	[1, 2],
	[1.2, 1.9],
	[6, 2],
	[5.7, 1.5],
	[5.4, 2.2]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
# 把样本绘制出来
mp.figure('Logistic Classification', facecolor='lightgray')
mp.title('Logistic Classification', fontsize=16)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')

# 绘制分类边界线
l, r = x[:, 0].min()-1, x[:, 0].max()+1
b, t = x[:, 1].min()-1, x[:, 1].max()+1
n = 500
grid_x, grid_y = np.meshgrid(
	np.linspace(l, r, n),
	np.linspace(b, t, n))

# 构建逻辑分类器
model = lm.LogisticRegression(
	solver='liblinear', C=1000)
model.fit(x, y)
# 把grid_x与grid_y抻平了组成模型的输入,预测输出
test_x = np.column_stack(
	(grid_x.ravel(), grid_y.ravel()))
pred_test_y = model.predict(test_x)
grid_z = pred_test_y.reshape(grid_x.shape)

mp.pcolormesh(grid_x,grid_y,grid_z,cmap='gray')

mp.scatter(x[:, 0], x[:, 1], s=60, c=y,
	marker='o', label='Points', cmap='jet')

mp.legend()
mp.show()
~~~

#### 朴素贝叶斯分类

> 一种依据统计理论而实现的一种分类方式. 观察一组数据:

| 天气情况 | 穿衣风格 | 约女朋友 | ==>  | 心情    |
| -------- | -------- | -------- | ---- | ------- |
| 0(晴天)  | 0(休闲)  | 0(约了)  | ==>  | 0(高兴) |
| 0        | 1(风骚)  | 1(没约)  | ==>  | 0       |
| 1(多云)  | 1        | 0        | ==>  | 0       |
| 0        | 2(破旧)  | 1        | ==>  | 1(郁闷) |
| 2(下雨)  | 2        | 0        | ==>  | 0       |
| ...      | ...      | ...      | ...  | ...     |
| 0        | 1        | 0        | ==>  | ?       |

通过上述训练样本如何预测:010的心情? 可以依照决策树的方式找相似输入预测输出. 但是如果在样本空间中没有完全匹配的相似样本该如何预测?

**贝叶斯公式:**
$$
联合概率（AB同时出现的概率）：P(A,B) = P(A)P(B|A) = P(B)P(A|B) \\
\Downarrow \Downarrow \Downarrow \\
条件概率（B出现时，A出现的概率）：P(A|B) = \frac{P(A)P(B|A)}{P(B)}
$$
例如:

假设一个学校中有60%男生和40%女生. 女生穿裤子的人数和穿裙子的人数相等. 所有男生都穿裤子. 一人在远处随机看到了一个穿裤子的学生, 那么这个学生是女生的概率是多少?

```
P(女) = 0.4
P(裤子|女) = 0.5
P(裤子) = 0.8
P(女|裤子) = P(女)*P(裤子|女)/P(裤子)
          = 0.4 * 0.5 / 0.8 = 0.25
```

根据贝叶斯定理, 如何预测: 晴天并且休闲并且没约并且高兴的概率?

```
P(晴天,休闲,没约,高兴)
P(晴天|休闲,没约,高兴)P(休闲,没约,高兴)
P(晴天|休闲,没约,高兴)P(休闲|没约,高兴)P(没约,高兴)
P(晴天|休闲,没约,高兴)P(休闲|没约,高兴)P(没约|高兴)P(高兴)
(朴素: 条件独立, 特征值之间没有任何关系)
P(晴天|高兴)P(休闲|高兴)P(没约|高兴)P(高兴)
```

朴素贝叶斯相关API:

```python
import sklearn.naive_bayes as nb
# 构建高斯朴素贝叶斯
model = nb.GaussianNB()
model.fit(x, y)
pred_test_y = model.predict(test_x)
```

**示例**

~~~python
import numpy as np
import sklearn.naive_bayes as nb
import matplotlib.pyplot as mp

data = np.loadtxt('multiple1.txt',
	unpack=False, delimiter=',')
print(data.shape, data.dtype)
# 获取输入与输出
x = np.array(data[:, :-1])
y = np.array(data[:, -1])

# 绘制这些点, 点的颜色即是点的类别
mp.figure('Naive Bayes', facecolor='lightgray')
mp.title('Naive Bayes', fontsize=16)
mp.xlabel('X', fontsize=14)
mp.ylabel('Y', fontsize=14)
mp.tick_params(labelsize=10)
# 通过样本数据,训练朴素贝叶斯分类模型
model = nb.GaussianNB()
model.fit(x, y)
# 绘制分类边界线
l, r = x[:, 0].min()-1, x[:, 0].max()+1
b, t = x[:, 1].min()-1, x[:, 1].max()+1
n = 500
grid_x, grid_y = np.meshgrid(
	np.linspace(l, r, n),
	np.linspace(b, t, n))
test_x = np.column_stack(
	(grid_x.ravel(), grid_y.ravel()))
pred_test_y = model.predict(test_x)
grid_z = pred_test_y.reshape(grid_x.shape)
mp.pcolormesh(grid_x,grid_y,grid_z,cmap='gray')

mp.scatter(x[:,0], x[:,1], s=60, c=y,
	cmap='jet', label='Points')
mp.legend()
mp.show()
~~~

---

### 决策树分类

> 决策树分类模型会找到与样本特征匹配的叶子节点然后以投票的方式进行分类. 

**示例**

> car.txt  基于决策树分类,预测小汽车等级.

1. 读取文本数据, 对每列进行标签编码, 基于随机森林分类器进行模型训练, 进行交叉验证.

~~~python
import numpy
import sklearn.preprocessing as sp#标签编码
import sklearn.ensemble as se#随机森林
import sklearn.model_selection as ms#评估

data = numpy.loadtxt("car.txt",delimiter=",",
  dtype="U20",converters={i:lambda x:x.decode() for i in range(7)}).T
encoders,train_x,train_y = [],[],[]

for row in range(len(data)):
    # 创建适用于当前特性的标签编码器
    encoder = sp.LabelEncoder()
    if row < len(data) - 1:
        train_x.append(
            encoder.fit_transform(data[row]))
    else:
        train_y = encoder.fit_transform(data[row])
    encoders.append(encoder)

train_x,train_y = numpy.array(train_x).T,numpy.array(train_y).T

#模型训练
model = se.RandomForestClassifier(max_depth=6,n_estimators=200,random_state=7)
#交叉验证
score = ms.cross_val_score(model,train_x,train_y,cv=4,scoring="f1_weighted")
print(score.mean())

model.fit(train_x,train_y)
~~~

2. 自定义测试集，使用模型进行预测

~~~python
# 自定义测试集  进行预测
data = [
['high','med','5more','4','big','low','unacc'],
['high','high','4','4','med','med','acc'],
['low','low','2','4','small','high','good'],
['low','med','3','4','med','high','vgood']]
# 训练时如何做的标签编码, 测试时需要使用相同的
# 标签编码器进行编码
data = numpy.array(data).T
test_x,test_y = [],[]

for row in range(len(data)):
    # 得到标签编码器
    encoder = encoders[row]
    if row < len(data) - 1:
        test_x.append(
            encoder.transform(data[row]))
    else:
        test_y = encoder.transform(data[ro1y))
print(encoders[-1].inverse_transform(pred_test_y))
~~~

#### 验证曲线

验证曲线:  模型性能(得分) = f(超参数)

```python
train_score,test_score=ms.validation_curve(
	model, 		# 需要验证的模型对象
    输入集, 输出集,  
    'n_estimators',	# 需要进行测试的超参数名称 
    np.arange(50, 550, 50), #超参数选值
    cv=5   # 折叠数)
```

train_scores的结构:

| 超参数取值 | cv_1  | cv_2  | cv_3  | cv_4  | cv_5  |
| ---------- | ----- | ----- | ----- | ----- | ----- |
| 50         | 0.912 | 0.912 | 0.912 | 0.912 | 0.912 |
| 100        | 0.912 | 0.912 | 0.912 | 0.912 | 0.912 |
| ...        | ...   | ...   | ...   | ...   | ...   |

通过验证曲线, 可以选择模型的较优超参数.

~~~python
model = se.RandomForestClassifier(n_estimators=150,random_state=7)
#输出验证曲线 选取最优:n_estimators
train_scores,test_scores = ms.validation_curve(model,train_x,train_y,"max_depth",numpy.arange(1,13,2),cv=5)

me_test = test_scores.mean(axis=1)

import matplotlib.pyplot as mp
mp.plot(numpy.arange(1,13,2),me_test)
mp.show()
~~~

---

#### 学习曲线

学习曲线: 模型性能 = f(训练集大小)

```python
_,train_score,test_score=ms.learning_curve(
	model, 输入集, 输出集, 
    [0.9, 0.8, 0.7], # 训练集大小序列
    cv=5	# 交叉验证折叠数
)
```

案例: 在小汽车评级案例中使用学习曲线选择训练集大小.

~~~python
# 验证学习曲线,获得最优训练集大小
train_sizes = numpy.linspace(0.1,1,10)
_,train_scores,test_scores = ms.learning_curve(model,train_x,train_y,train_sizes=train_sizes,cv=5)
import matplotlib.pyplot as mp
mp.plot(train_sizes,test_scores.mean(axis=1))
mp.show()
~~~

---

### 聚类

分类(class) 与 聚类 (cluster) 不同, 分类属于有监督学习, 聚类属于无监督学习模型.  聚类讲究使用一些算法把样本划分为n个群落. 一般情况下,这种算法都需要计算欧氏距离. 

欧氏距离(欧几里得距离):
$$
P(x_1)-P(x_2): |x_1-x_2| = \sqrt{(x_1-x_2)^2}\\
p(x_1,y_1)-p(x_2,y_2):  \sqrt{(x_1-x_2)^2 + (y_1-y_2)^2} \\
p(x_1,y_1,z_1)-p(x_2,y_2,z_2):  \sqrt{(x_1-x_2)^2 + (y_1-y_2)^2 + (z_1-z_2)^2}
$$

#### K均值算法

第一步: 随机选择K个样本作为K个聚类中心, 计算每个样本到各个聚类中心的欧式距离, 将该样本分配到与之距离最近的聚类中心所在的类别中.

第二步: 根据第一步得到的聚类划分, 分别计算每个聚类所有样本的几何中心, 将几何中心作为新的聚类中心,重复第一步. 直到计算所得几何中心与聚类中心重合或接近重合为止.

**注意:**

1. 聚类数K必须事先已知. 可以借助某些指标,选择最好的K.
2. 聚类中心的初始选择会影响到最终的聚类划分的结果. 初始重新样本尽量选择距离较远的样本.

K均值算法相关API:

```python
import sklearn.cluster as sc
#n_clusters: 聚类数
model = sc.KMeans(n_clusters=3)
model.fit(x)
# 获取聚类中心
centers = model.cluster_centers_
```

案例: multiple3.txt

```python
import numpy
import sklearn.cluster as sc
import matplotlib.pyplot as mp
point = numpy.loadtxt('multiple3.txt',
					 delimiter=',')
# KMeans聚类
model = sc.KMeans(n_clusters=4)
model.fit(point)
pred_y = model.predict(point)

# 划分聚类边界
l, r = point[:, 0].min() - 1, point[:, 0].max() + 1
b, t = point[:, 1].min() - 1, point[:, 1].max() + 1
n = 500
grid_x, grid_y = numpy.meshgrid(
	numpy.linspace(l, r, n),
	numpy.linspace(b, t, n))
mesh_x = numpy.column_stack((grid_x.ravel(),
                             grid_y.ravel()))

pred_mesh_y = model.predict(mesh_x)
grid_z = pred_mesh_y.reshape(grid_x.shape)

mp.pcolormesh(grid_x,grid_y,grid_z,cmap="gray")
mp.scatter(point[:,0], point[:,1], s=60,marker='o', c=pred_y,cmap="jet")
centers = model.cluster_centers_
mp.scatter(centers[:,0],centers[:,1],marker='+',s=130,c="red")
mp.show()
```

#### 图像量化

Kmeans聚类算法可以应用于图像量化领域. 通过KMeans算法可以把一张图像所包含的颜色值进行聚类划分. 得到划分后的聚类中心后, 把靠近某个聚类中的点的亮度值改为聚类中心值, 由此生成新的图片. 达到图像降维的目的. 这个过程称为图像量化.  

图像量化可以很好的保存图像的轮廓, 降低机器识别的难度.

案例:  lily.jpg

```python
import scipy.misc as sm  # 读取图片
import sklearn.cluster as sc

img = sm.imread('lily.jpg', True)
# 图像量化
x = img.reshape(-1, 1)
model = sc.KMeans(n_clusters=4)
model.fit(x)
# 把每个亮度值修改为相应的聚类中心值
centers = model.cluster_centers_.ravel()
print(centers)
y = model.predict(x)
y = y.reshape(img.shape)
# 使用numpy的掩码操作 修改y数组的每个值
result = centers[y]

sm.imshow(result)-
```

#### 均值漂移算法

首先假定样本空间中的每个聚类均服从某种已知的概率分布规则, 然后用不同的概率密度函数拟合样本中的统计直方图, 不断移动密度函数的中心位置, 直到获得最佳拟合效果为止.这些概率密度函数的峰值点就是聚类的中心, 再根据每个样本距离各个中心的距离, 选择最近的聚类中心所属的类别作为该样本的类别.

均值漂移算法的特点:

1. 聚类数不必事先已知, 算法会自动识别出统计直方图的中心数量.
2. 聚类中心不依据于最初假定, 聚类划分的结果相对稳定.
3. 样本空间应该服从某种概率分布规则, 否则算法的准确性将会大打折扣.

均值漂移相关的API:

```python
# x: 输入 n_samples: 样本数量
# quantile: 量化宽度 (直方图一条的宽度)
bw = sc.estimate_bandwidth(
    x, n_samples=len(x), quantile=0.1)
# 构建均值漂移模型
model = sc.MeanShift(bandwidth=bw)
```

案例: multiple3.txt  

```python
import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as mp

x = np.loadtxt('multiple3.txt',
	delimiter=',')

# 均值漂移实现聚类划分

bw = sc.estimate_bandwidth(x,n_samples=len(x),quantile=0.2)
model = sc.MeanShift(bandwidth=bw)

model.fit(x)
centers = model.cluster_centers_
pred_y = model.predict(x)

# 划分聚类边界
l, r = x[:, 0].min()-1, x[:, 0].max()+1
b, t = x[:, 1].min()-1, x[:, 1].max()+1
n = 500
grid_x, grid_y = np.meshgrid(
	np.linspace(l, r, n),
	np.linspace(b, t, n))
mesh_x = np.column_stack((grid_x.ravel(),
	grid_y.ravel()))
pred_mesh_y = model.predict(mesh_x)
grid_z = pred_mesh_y.reshape(grid_x.shape)

mp.figure('MeanShift', facecolor='lightgray')
mp.title('MeanShift', fontsize=16)
mp.xlabel('X',fontsize=14)
mp.ylabel('Y',fontsize=14)
mp.tick_params(labelsize=10)
mp.pcolormesh(grid_x,grid_y,grid_z,cmap='gray')
mp.scatter(x[:,0], x[:,1], c=pred_y, cmap='jet',
		label='points')
# 绘制聚类中心点
mp.scatter(centers[:,0], centers[:,1],
	marker='+', s=230, c='orangered')
mp.legend()
mp.show()
```

#### 凝聚层次算法

首先假定每个样本都是一个独立的聚类, 如果统计出来的聚类数大于期望的聚类数, 则从每个样本出发, 寻找离自己最近的另外一个样本, 与之聚集, 形成更大的聚类. 同时另总聚类数减少, 不断重复以上过程, 直到统计出来的聚类总数达到期望值为止.

凝聚层次算法的特点:

1. 凝聚数量必须事先已知. 可以借助于某些指标, 优选参数.
2. 没有聚类中心的概念, 因此只能在训练集中划分聚类, 但不能对训练集以外的未知样本确定其归属.
3. 在确定被凝聚样本时, 除了以距离作为条件以外, 还可以根据连续性来确定被聚集的样本.

凝聚层次相关API:

```python
# 构建凝聚层次聚类模型
model = sc.AgglomerativeClustering(
        n_clusters=4)
pred_y = model.fit_predict(x)
```

案例: multiple3.txt

```python
import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as mp

x = np.loadtxt('multiple3.txt',
	delimiter=',')

# 凝聚层次实现聚类划分
model = sc.AgglomerativeClustering(n_clusters=4)
pred_y = model.fit_predict(x)

mp.scatter(x[:,0], x[:,1], c=pred_y, cmap='jet')
mp.show()
```

案例: 以连续性为条件进行聚类划分.

```python
import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as mp
import sklearn.neighbors as nb

x = np.loadtxt('multiple3.txt',
	delimiter=',')

# 凝聚层次实现聚类划分 以连续性为条件
# 近邻筛选器(10代表临近角度)
conn = nb.kneighbors_graph(x,10,include_self=False)
model = sc.AgglomerativeClustering(linkage="average",n_clusters=4,connectivity=conn)
pred_y = model.fit_predict(x)

mp.scatter(x[:,0], x[:,1],s=40, c=pred_y, cmap='jet')
mp.show()
```

#### 轮廓系数

轮廓系数用于评估一个聚类模型的性能. 一个好的聚类: 内密外疏. 同一个聚类内部的样本要足够密集, 不同聚类之间的样本要足够稀疏.

轮廓系数的计算规则: 针对样本空间中的一个特定样本, 计算它与所在聚类其它样本的平均距离a, 以及该样本与距离最近的另一个聚类中所有的样本的平均距离b. 那么该样本的轮廓系数为(b-a)/max(a,b).   若将整个样本空间中所有样本的轮廓系数取算数平均值, 就可以把该结果作为聚类划分的指标.

该公式结果处于:[-1, 1].  -1代表分类效果比较差, 1代表分类效果好. 0代表聚类重叠, 没有很好的划分聚类.

```python
import sklearn.metrics as sm
score = sm.silhouette_score(
    输入集, 输出集, 
    sample_size=样本数,
    # 距离算法: euclidean 欧几里得距离
	metric='euclidean' 
)
```

案例:

```python
import sklearn.metrics as sm
# 输出轮廓系数
score = sm.silhouette_score(
    point,pred_y,
    sample_size=len(point),
    metric="euclidean"
)
print(score)
```

#### DBSCAN算法

从样本空间中任意选择一个样本, 以事先给定的半径做圆. 凡是被该圆圈中的样本都视为与该样本处于同样的聚类. 以这些被圈中样本为圆心继续做圆.不断的扩大被圈中样本的规模, 直到没有新的样本加入为止, 由此得到一个聚类. 

在剩余样本中重复以上过程,直到耗尽样本空间中所有的样本为止.

DBSCAN算法的特点:

1. 事先给定的半径会影响最后的聚类效果, 可以根据轮廓系数选择较优的方案.

2. 根据聚类的形成过程, DBSCAN算法支持把样本分为3类:

   **外周样本:** 被其他样本聚集到某个聚类中, 但无法引入新样本的样本.

   **孤立样本:** 聚类中样本数低于所设置的下限, 则不称其为聚类, 反之称为孤立样本.

   **核心样本:** 除了外周样本和孤立样本外的其他样本.

```python
# 构建DBSCAN聚类模型
# eps: 半径
# min_samples: 最小样本数,若低于该值,则为孤立样本
model = sc.DBSCAN(eps=1, min_samples=5)
model.fit(x)
#样本标签分类值，如-1,0,1,2,3之类的分类
pred_y = best_model.labels_
# 获取核心样本的索引
core_indices=model.core_sample_indices_
#孤立样本的类别标签为-1
offset_mask = pred_y == -1
# 外周样本掩码 (不是核心也不是孤立样本)
p_mask = ~(core_mask | offset_mask)
```

案例: perf.txt

```python
import numpy
import sklearn.cluster as sc
import sklearn.metrics as sm
import matplotlib.pyplot as mp

x = numpy.loadtxt('perf.txt',
                  delimiter=',')
# 准备训练模型相关数据
epsilons, scores, models = \
	numpy.linspace(0.3, 1.2, 10), [], []
# 遍历所有的半径, 训练模型, 查看得分
for epsilon in epsilons:
	model=sc.DBSCAN(eps=epsilon,min_samples=5)
	model.fit(x)
	score=sm.silhouette_score(x, model.labels_,
		sample_size=len(x), metric='euclidean')
	scores.append(score)
	models.append(model)
# 转成ndarray数组
scores = numpy.array(scores)
best_i = scores.argmax() # 最优分数的索引
best_eps = epsilons[best_i]
best_sco = scores[best_i]
print(best_eps)
print(best_sco)
# 获取最优模型
best_model = models[best_i]

# 对输入x进行预测得到预测类别
pred_y = best_model.labels_ 

# 获取孤立样本, 外周样本, 核心样本
core_mask = numpy.zeros(len(x), dtype=bool)
# 获取核心样本的索引, 把对应位置的元素改为True
core_mask[best_model.core_sample_indices_]=True
# 孤立样本的类别标签为-1
offset_mask = pred_y == -1
# 外周样本掩码 (不是核心也不是孤立样本)
p_mask = ~(core_mask | offset_mask)


# 绘制核心样本
mp.scatter(x[core_mask][:,0], x[core_mask][:,1],
	s=60, cmap='brg', c=pred_y[core_mask])
# 绘制外周样本
mp.scatter(x[p_mask][:,0], x[p_mask][:,1],
	s=60, cmap='brg', c=pred_y[p_mask],
	alpha=0.5)
# 绘制孤立样本
mp.scatter(x[offset_mask][:,0],
	x[offset_mask][:,1], s=60, c='gray')

mp.show()
```
---

### 评估训练结果误差（metrics）

> 线性回归模型训练完毕后，可以利用测试集评估训练结果的误差。sklearn.metrics模块提供了计算模型误差的几个常用算法。

+ 平均值绝对值误差：sm.mean_absolute_error(y,pred_y)
  + y真实输出
  + pred_y预测输出
  + 返回值：1/m∑丨预测输出-真实输出丨
+ 平均值平方误差：sm.mean_squared_error(y,pred_y)
  + 返回值：sqrt（1/m∑丨预测输出-真实输出丨^2）
+ 中位数绝对值误差：sm.median_absolute_error(y,pred_y)
  + 返回值：median（1/m∑丨预测输出-真实输出丨）
+ R2得分：sm.r2_score(y,pred_y)
  + (0,1]的一个分值，分数越高，误差越小

~~~python
print(sm.mean_absolute_error(y,pred_y))#平均值绝对值误差
print(sm.mean_squared_error(y,pred_y))#平均值平方误差
print(sm.median_absolute_error(y,pred_y))#中位数绝对值误差
print(sm.r2_score(y,pred_y))#R2得分
~~~


#### 数据集的划分

> 对于分类问题训练集和测试集的划分不应该用整个样本空间的特定百分比作为训练数据, 而应该在其每一个类别的样本中抽取特定百分比作为训练数据. 最终提高分类的可信度.

sklearn提供了数据集划分的相关API:

```python
import sklearn.model_selection as ms
# 训练集测试集划分
ms.train_test_split(
    输入集, 输出集, 
	test_size=测试集占比,
	random_state=7
)
返回: train_x, test_x, train_y, test_y
```

**示例**

~~~python
import numpy as np
import sklearn.naive_bayes as nb
import matplotlib.pyplot as mp
import sklearn.model_selection as ms

data = np.loadtxt('multiple1.txt',
	unpack=False, delimiter=',')
print(data.shape, data.dtype)
# 获取输入与输出
x = np.array(data[:, :-1])
y = np.array(data[:, -1])

# 绘制这些点, 点的颜色即是点的类别
mp.figure('Naive Bayes', facecolor='lightgray')
mp.title('Naive Bayes', fontsize=16)
mp.xlabel('X', fontsize=14)
mp.ylabel('Y', fontsize=14)
mp.tick_params(labelsize=10)

# 训练集测试集划分
train_x, test_x, train_y, test_y = \
	ms.train_test_split(
	x, y, test_size=0.25, random_state=7)

# 通过训练样本,训练朴素贝叶斯分类模型
model = nb.GaussianNB()

model.fit(train_x, train_y)
# 对测试样本进行预测, 输出预测精确度
pred_test_y = model.predict(test_x)
# 精确度 = 预测正确的个数/总个数
print((test_y==pred_test_y).sum()/test_y.size)

# 绘制分类边界线
l, r = x[:, 0].min()-1, x[:, 0].max()+1
b, t = x[:, 1].min()-1, x[:, 1].max()+1
n = 500
grid_x, grid_y = np.meshgrid(
	np.linspace(l, r, n),
	np.linspace(b, t, n))
mesh_x = np.column_stack(
	(grid_x.ravel(), grid_y.ravel()))
pred_mesh_y = model.predict(mesh_x)
grid_z = pred_mesh_y.reshape(grid_x.shape)
mp.pcolormesh(grid_x,grid_y,grid_z,cmap='gray')

mp.scatter(test_x[:,0], test_x[:,1], s=60,
	c=test_y, cmap='jet', label='Train Points')
mp.legend()
mp.show()
~~~

##### 交叉验证 

> 由于数据集的划分有不确定性, 若随机划分的样本正好处于某类特殊样本, 则得到的训练模型所预测的结果的可信度会受到质疑. 所以要进行多次**交叉验证**, 把样本空间中的所有样本均分成n份, 使用不同的训练集训练模型, 对不同的测试集进行测试并输出指标得分.

```python
import sklearn.model_selection as ms
# 使用给出的模型,针对输入与输出进行5次交叉验证
# 把每次交叉验证得到的精准度得分以数组的方式返回
score = ms.cross_val_score(
    模型, 输入集, 输出集,
    cv=5,          	   # 交叉验证的次数
    scoring='accuracy' # 指标名称 (精准度)
)
```

**示例**

~~~python
# 到底用不用这个模型呢? 交叉验证看一下分数吧
score = ms.cross_val_score(
	model, train_x, train_y, cv=5, 
	scoring='accuracy')
print('CV Accuracy:', score)
~~~

##### 交叉验证的指标

+ 精准度(accuracy): 分类正确的样本数/总样本数
+ 查准率(precision_weighted): 针对每一类别, 预测正确的样本数 / 预测出来的样本数
+ 召回率(recall_weighted): 针对每一类别, 预测正确的样本数 / 实际存在的样本数
+ f1得分(f1_weighted): 2x查准率x召回率/(查准率+召回率)

~~~python
ac = ms.cross_val_score(
	model, train_x, train_y, cv=5, 
	scoring='accuracy')
print('CV Accuracy:', ac.mean())
pw = ms.cross_val_score(
	model, train_x, train_y, cv=5, 
	scoring='precision_weighted')
print('CV PW:', pw.mean())
rw = ms.cross_val_score(
	model, train_x, train_y, cv=5, 
	scoring='recall_weighted')
print('CV RW:', rw.mean())
fw = ms.cross_val_score(
	model, train_x, train_y, cv=5, 
	scoring='f1_weighted')
print('CV FW:', fw.mean())
~~~

---

##### 混淆矩阵

混淆矩阵中每一行和每一列分别对应样本输出中的每一个类别, 行表示实际类别, 列表示预测类别.

|       | A类别 | B类别 | C类别 |
| ----- | ----- | ----- | ----- |
| A类别 | 5     | 0     | 0     |
| B类别 | 0     | 6     | 0     |
| C类别 | 0     | 0     | 7     |

上述的混淆矩阵, 是一个理想的混淆矩阵, 不理想的如下:

|       | A类别 | B类别 | C类别 |
| ----- | ----- | ----- | ----- |
| A类别 | 3     | 1     | 1     |
| B类别 | 0     | 4     | 2     |
| C类别 | 0     | 0     | 7     |

查准率 = 主对角线上的值 /  该值所在列的和（如B=4/5）

召回率 = 主对角线上的值 / 该值所在行的和 (如A=3/5)

```python
import sklearn.metrics as sm
# 返回混淆矩阵
m = sm.confusion_matrix(实际输出, 预测输出)
```

**示例**

```python
# 输出混淆矩阵
m = sm.confusion_matrix(test_y, pred_test_y)
print(m)

mp.figure('CM')
mp.imshow(m, cmap='gray')
```

##### 分类报告

> sklearn提供了分类报告, 不仅可以得到混淆矩阵, 还可以得到交叉验证的查准率/ 召回率/f1得分的结果, 可以方便的分析出哪些样本是异常样本.

```python
cr=sm.classification_report(实际输出, 预测输出)
print(cr)
```

---

### 模型的保存和加载

> 模型训练是一个耗时的过程，一个优秀的机器学习模型是非常宝贵的。所以当模型训练完毕后，可以把模型保存在磁盘中，在需要的时候可以从磁盘中重新加载模型。不需要在重新训练。

~~~python
import pickle
pickle.dump(model,磁盘文件)		#保存模型
model = pickle.load(磁盘文件)	#加载模型

with open("linear.pkl",'wb') as f:
    pickle.dump(model,f)
with open("linear.pkl",'rb') as f:
    model = pickle.load(f)
~~~

---
## 推荐引擎

> 把用户最需要的内容找到并推荐给用户.

针对不用的业务需求, 一般情况下推荐流程:

1. 根据当前用户信息, 寻找相似用户
2. 根据相似用户的行为, 选择推荐内容.
3. 对推荐内容进行重要性排序, 最终推荐给用户.



针对不同推荐业务场景都需要分析相似样本. 统计相似样本可以基于欧式距离分数.(也可以基于皮氏距离分数)
$$
欧式距离分数= \frac{1}{1+欧式距离}
$$
该欧式距离分数区间处于: (0,1], 越趋近于0, 样本间的欧式距离越远,样本越不相似; 越趋近于1, 则样本间的欧式距离越近, 越相似.

|      | a    | b    | c    | d    | ...  |
| ---- | ---- | ---- | ---- | ---- | ---- |
| a    | 1    | 0.4  | 0.8  | 0.1  | ...  |
| b    | 0.4  | 1    | 0.9  | 0.2  | ...  |
| c    | 0.8  | 0.9  | 1    | 0.6  | ...  |
| d    | 0.1  | 0.2  | 0.6  | 1    | ...  |
| ...  | ...  | ...  | ...  | ...  | ...  |

代码实现:   ratings.json

```python
import json
import numpy
with open("ratings.json","r") as f:
    ratings = json.loads(f.read())

users,scmat = list(ratings.keys()),[]
# 把每个相似度得分,存入scmat矩阵.供以后使用
for user1 in users:
    scrow = []#存储user1与其他人的相似度得分
    for user2 in users:
        movies = set()
        for movie in ratings[user1]:
            #user1看过的user2也看过
            if movie in ratings[user2]:
                movies.add(movie)
        #2人没交集
        if len(movies) == 0:score = 0
        else:
            #2人都有看过的电影
            x,y = [],[]
            for movie in movies:
                x.append(ratings[user1][movie])
                y.append(ratings[user2][movie])
            x,y = numpy.array(x),numpy.array(y)
            score = 1 / (1+numpy.sqrt( ( (x-y)**2 ).sum() ))
        scrow.append(score)
    scmat.append(scrow)

users = numpy.array(users)
scmat = numpy.array(scmat)
for scrow in scmat:
    output = " ".join('{:.2f}'.format(score) for score in scrow)
    print(output)
```

**皮尔逊相关系数**

```
score = numpy.corrcoef(x, y)[0, 1]
```

皮尔逊相关系数 = 协方差 / 标准差之积

相关系数处于[-1, 1]之间.  越靠近1越正相关, 越靠近-1越负相关.

**按照相关系数从高到低排列每个用户的相似度**

```python
for index,user in enumerate(users):
    sorted_indices = scmat[index].argsort()[::-1]
    # 忽略当前user
    sorted_indices = sorted_indices[sorted_indices != index]
    #获取相似用户数组
    similar_users = users[sorted_indices]
    #获取每个相似用户的相似度得分
    similar_scores = scmat[index,sorted_indices]
    print(user,similar_users,similar_scores,sep="\n")
```

**生成推荐清单**

1. 找到所有皮尔逊相关系数正相关的用户.
2. 遍历当前用户的每个相似用户, 拿到相似用户看过,且当前用户没有看过的电影. 
3. 计算每个电影的推荐度.  获取所有用户对当前推荐电影的打分情况, 求均值, 以此来作为该电影的推荐度.
4. 排序, 输出.

```python
	# 生成推荐清单
	# 找到所有正相关用户即相关性分数
	positive_mask=similar_scores > 0
	similar_users=similar_users[positive_mask]
	similar_scores=similar_scores[positive_mask]
	# 存储对当前用户的推荐的电影
	# recomm_movies = {'电影名':[0.5, 0.4, 0.3]}
	recomm_movies = {}
	for i, similar_user in \
				enumerate(similar_users):
		# 拿到相似用户看过, 但user没看过的电影
		for movie, score in \
				ratings[similar_user].items():
			if movie not in ratings[user].keys():
				if movie not in recomm_movies:
					recomm_movies[movie]=[score]
				else:
					recomm_movies[movie].append(score)

	print(user)
	#print(recomm_movies)
	movie_list = sorted(recomm_movies.items(), 
		key=lambda x:np.average(x[1]), 
		reverse=True)
	print(movie_list)
```

---

## 自然语言处理(NLP)

Siri工作流程: 1. 听  2. 懂  3.思考  4. 组织语言  5.回答

1. 语音识别
2. 自然语言处理 - 语义分析
3. 业务逻辑分析 - 结合场景 上下文
4. 自然语言处理 - 分析结果生成自然语言文本
5. 语音合成

#### 自然语言处理

自然语言处理的常用处理过程:

先针对训练文本进行分词处理(词干提取, 原型提取), 统计词频, 通过词频-逆文档频率算法获得该词对整个样本语义的贡献, 根据每个词对语义的贡献力度, 构建有监督分类学习模型. 把测试样本交给模型处理, 得到测试样本的语义类别.

自然语言处理工具包 - nltk,

+ nltk.download()下载对应语言

#### 文本分词

```python
import nltk.tokenize as tk
# 把一段文本拆分句子
sent_list = tk.sent_tokenize(text)
# 把一句话拆分单词
word_list = tk.word_tokenize(sent)
# 通过文字标点分词器 拆分单词 比如It's拆为It ' s
punctTokenizer = tk.WordPunctTokenizer()
word_list = punctTokenizer.tokenize(text)
```

案例:

```python
import nltk.tokenize as tk
doc = """Are you curious about tokenization?
Let's see how it works!
We neek to analyze a couple of sentences with punctuations to see it in action."""

sent_list = tk.sent_tokenize(doc)
for index,sent in enumerate(sent_list,start=1):
    print("%2d" %(index),sent)

word_list = tk.word_tokenize(doc)
for index,word in enumerate(word_list,start=1):
    print("%2d" %(index),word)

tokenizer = tk.WordPunctTokenizer()
word_list = tokenizer.tokenize(doc)
for index,word in enumerate(word_list,start=1):
    print("%2d" %(index),word) 
```

#### 词干提取

```python
import nltk.stem.porter as pt
import nltk.stem.lancaster as lc
import nltk.stem.snowball as sb

# 波特词干提取器  (偏宽松)
stemmer = pt.PorterStemmer()
# 朗卡斯特词干提取器   (偏严格)
stemmer = lc.LancasterStemmer()
# 思诺博词干提取器   (偏中庸)
stemmer = sb.SnowballStemmer('english')
r = stemmer.stem('playing') # 词干提取
```

#### 词性还原

与词干提取作用类似, 次干提取出的词干信息不利于人工二次处理(人读不懂), 词性还原可以把名词复数等形式恢复为单数形式. 更有利于人工二次处理.

```python
import nltk.stem as ns
# 词性还原器
lemmatizer = ns.WordNetLemmatizer()
n_lemm=lemmatizer.lemmatize(word, pos='n')
v_lemm=lemmatizer.lemmatize(word, pos='v')
```

案例:

```python
import nltk.stem as ns

words = ['table', 'probably', 'wolves', 
	'playing', 'is', 'the', 'beaches', 
	'grouded', 'dreamt', 'envision']

lemmatizer = ns.WordNetLemmatizer()
for word in words:
	n_lemm = lemmatizer.lemmatize(word,pos='n')
	v_lemm = lemmatizer.lemmatize(word,pos='v')
	print('%8s %8s %8s' % \
		  (word, n_lemm, v_lemm))
```

#### 词袋模型

文本分词处理后, 若需要分析文本语义, 需要把分词得到的结果构建样本模型, 词袋模型就是由每一个句子为一个样本, 单词在句子中出现的次数为特征值构建的数学模型. 

The brown dog is running. The black dog is in the black room. Running in the room is forbidden.

1. The brown dog is running. 
2. The black dog is in the black room.
3. Running in the room is forbidden.

| the  | brown | dog  | is   | running | black | in   | room | forbidden |
| ---- | ----- | ---- | ---- | ------- | ----- | ---- | ---- | --------- |
| 1    | 1     | 1    | 1    | 1       | 0     | 0    | 0    | 0         |
| 2    | 0     | 1    | 1    | 0       | 2     | 1    | 1    | 0         |
| 1    | 0     | 0    | 1    | 1       | 0     | 1    | 1    | 1         |

获取一篇文档的词袋模型:

```python
import sklearn.feature_extraction.text as ft
import nltk as tk
text = "The brown dog is running. The black dog is in the black room. Running in the room is forbidden."
sentences = tk.sent_tokenize(text)
model = ft.CountVectorizer()
bow = model.fit_transform(sentences)
print(bow.toarray())
words = model.get_feature_names()
print(words)
```

#### 词频(TF)

单词在句子中出现的次数 除以 句子的总词数 称为词频. 即一个单词在句子中出现的频率.  词频相对于单词出现的次数可以更加客观的评估单词对一句话的语义的贡献度. **词频越高,代表当前单词对语义贡献度越大.** 

#### 文档频率(DF)

含有某个单词的文档样本数 / 总文档样本数.

#### 逆文档频率(IDF)

总文档样本数 / 含有某个单词的文档样本数

**单词的逆文档频率越高, 代表当前单词对语义的贡献度越大.**

#### 词频-逆文档频率(TF-IDF)

词频矩阵中的每一个元素乘以相应单词的逆文档频率, 其值越大, 说明该词对样本语义的贡献度越大. 可以根据每个单词的贡献力度, 构建学习模型.

获取TFIDF矩阵相关API:

```python
model = ft.CountVectorizer()
bow = model.fit_transform(sentences)
# 获取IFIDF矩阵
tf = ft.TfidfTransformer()
tfidf = tf.fit_transform(bow)
# 基于tfidf 做模型训练
....
```

案例:

```python
import sklearn.feature_extraction.text as ft
import nltk.tokenize as tk
import numpy as np
doc = 'The brown dog is running. \
	The black dog is in the black room. \
	Running in the room is forbidden.'
# 拆分句子
sents = tk.sent_tokenize(doc)
print(sents)
# 构建词袋模型
model = ft.CountVectorizer()
bow = model.fit_transform(sents)
print(model.get_feature_names())

# 通过词袋矩阵  得到tfidf矩阵
tf = ft.TfidfTransformer()
tfidf = tf.fit_transform(bow)
print(np.round(tfidf.toarray(), 2))
```

#### 文本分类(主题识别)

```python
import sklearn.datasets as sd#包含读取数据的方法
import sklearn.feature_extraction.text as ft
import sklearn.naive_bayes as nb#普贝斯
import numpy

train = sd.load_files("20news",encoding="latin1",shuffle = True,random_state=7)

print(numpy.array(train.data).shape)#每个文件的字符串内容
#返回每个文件的父目录名（分类数字标签）
print(numpy.array(train.target).shape)

#整理输入集 输出集
categories = train.target_names#类别名称
cv_model = ft.CountVectorizer()
bow = cv_model.fit_transform(train.data)
tf = ft.TfidfTransformer()

train_x = tf.fit_transform(bow)
#选择使用朴素贝叶斯进行模型训练
model = nb.MultinomialNB()
model.fit(train_x,train.target)

# 测试集  测试模型是否可用
test_data = [
	'The curveballs of right handed pitchers \
	tend to curve to the left.',
	'Caesar cipher is an ancient form of \
	encryption.',
	'This two-wheeler is really good on slippery \
	roads.']
bow = cv_model.transform(test_data)
tfidy = tf.transform(bow)
pred_y = model.predict(tfidy)

for sent,index in zip(test_data,pred_y):
    print(sent,"->",categories[index])
```

#### nltk内置分类器

nltk提供了朴素贝叶斯分类器方便的处理NLP相关的分类问题. 并且可以自动处理词袋, 完成TFIDF矩阵的整理, 完成模型训练, 最终实现类别预测.  使用方法:

```python
import nltk.classify as cf
import nltk.classify.util as su
'''
train_data与test_data的格式
不再是一行一样本, 一列一特征   格式如下:
[
({'age':10, 'score1':95, 'score2':80},'g'),
({'age':10, 'score1':35, 'score2':40},'b'),
({'age':10, 'score1':95, 'score2':80},'g')
]
'''
model =
  cf.NaiveBayesClassifier.train(train_data)
ac = cu.accuracy(model, test_data)
```

#### 情感分析

分析语料库中movie_reviews文档, 通过正面及负面评价进行自然语言训练, 实现情感分析.

```python
import nltk.corpus as nc
import nltk.classify as cf
import nltk.classify.util as cu
def get_data(data, train_y):
	# 存储数据(格式要求为：[({train_x:数字},tranin_y)]
	result = []
	for fileid in data:
		sample={}
		# 对全文进行分词,得到words列表
		for word in nc.movie_reviews.words(fileid):
			sample[word] = True
		result.append((sample, train_y))
	return result

def split_train_test(perce,x,y):
	x_size,y_size = int(len(x) * perce),int(len(y) * perce)
	return (x[:x_size]+y[:y_size],x[x_size:]+y[y_size:])

# 存储正面数据
# 读取语料库中movie_reviews文件夹中的dir_name文件夹
#把每个文件的文件名返回
data = nc.movie_reviews.fileids("pos")
pdata = get_data(data,"POSITIVE")
# 存储负面数据
ndata = get_data(nc.movie_reviews.fileids("neg"),"NEGTIVE")

#拆分测试集与训练集
train_data,test_data = split_train_test(0.8,pdata,ndata)

#基于朴素贝斯模型，训练测试数据
model = cf.NaiveBayesClassifier.train(train_data)
accuracy = cu.accuracy(model,test_data)
print(accuracy)

# 模拟业务场景
reviews = [
 'It is an amazing movie. ',
 'This is a dull movie, I would never \
  recommend it to anyone. ',
 'The cinematography is pretty great \
  in this movie. ',
 'The direction was terrible and the story \
  was all over the place. ']

for review in reviews:
	sample = {}
	words = review.split() # 野蛮分词
	for word in words:
		sample[word] = True
	# classify类似predict方法, 通过样本预测类别
	pred_y = model.classify(sample)
	print(review, '->', pred_y)
```

---

## 语音识别

语音识别可以实现通过一段音频信息(wav波) 识别出音频的内容. 

通过傅里叶变换, 可以将时间域的声音分解为一系列不同频率的正弦函数的叠加. 通过频率谱线的特殊分布, 建立音频内容与文本之间的对应关系, 以此作为模型训练的基础.

1. 准备多个声音样本作为训练数据. 并且为每个音频都标明其类别.
2. 读取每一个音频文件, 获取音频文件的mfcc矩阵.
3. 以mfcc作为训练样本, 进行训练.
4. 对测试样本进行测试.  (基于隐马模型)
MFCC相关API:

```python
import scipy.io.wavfile as wf
import python_speech_features as sf
#采样率/采用值
sample_rate, sigs = wf.read('../xx.wav')
mfcc = sf.mfcc(sigs, sample_rate)
```

案例:

```python
import scipy.io.wavfile as wf
import python_speech_features as sf
import matplotlib.pyplot as mp

sample_rate, sigs=wf.read("speeches/training/apple/apple01.wav")
mfcc = sf.mfcc(sigs, sample_rate)
print(mfcc.shape)

mp.matshow(mfcc.T, cmap='gist_rainbow')
mp.title('MFCC')
mp.ylabel('Features', fontsize=14)
mp.xlabel('Samples', fontsize=14)
mp.tick_params(labelsize=10)
mp.show()
```

隐马尔科夫模型相关API:

```python
import hmmlearn.hmm as hl
# 构建隐马模型 
# n_components: 用几个高斯函数拟合样本数据
# covariance_type:使用相关矩阵辅对角线进行相关性比较
# n_iter: 最大迭代上限
model = hl.GaussianHMM(
    n_components=4, 
    covariance_type='diag', 
	n_iter=1000)
model.fit(mfccs)
# 通过训练好的隐马模型  验证音频mfcc的得分 
# 匹配度越好, 得分越高
score = model.score(test_mfcc)
```

案例:

```python
import os
import numpy
import scipy.io.wavfile as wf
import python_speech_features as sf
import hmmlearn.hmm as hh

def search_files(directory):
    #对文件路径进行跨平台处理
    directory = os.path.normcase(directory)
    # {'apple':[dir,dir,dir], 'banana':[dir..]}
    result = {}
    #当前目录、子目录、文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                label = root.split(os.path.sep)[-1]
                if label not in result:
                    result[label] = []
                path = os.path.join(root,file)
                result[label].append(path)
    return result
#构建高斯结构的x_y
def GaussianHMM_x_y(files):
    train_x, train_y = [], []
    for label, filenames in files.items():
        mfccs = numpy.array([])
        for filename in filenames:
            sample_rate, sigs = wf.read(filename)
            mfcc = sf.mfcc(sigs, sample_rate)
            if len(mfccs) == 0:
                mfccs = mfcc
            else:
                mfccs = numpy.append(mfccs, mfcc, axis=0)

        train_x.append(mfccs)
        train_y.append(label)
    return (train_x,train_y)

train_samples = search_files("speeches/training")
#整理训练集，把每一个类别中的音频的mfcc放在一起，基于隐码模型开始训练
train_x,train_y = GaussianHMM_x_y(train_samples)

#基于隐码模型进行训练，把所有类别的模型都存起来
# 一个七个类别循环7次
models = {}
for mfccs,label in zip(train_x,train_y):
    model = hh.GaussianHMM(n_components=4,
        covariance_type="diag",n_iter=1000)
    models[label] = model.fit(mfccs)

# 读取测试集中的文件，使用每个模型对文件进行评分
#取分直大的模型对于的label作为预测类别
test_samples = search_files("speeches/testing")
#整理训练集，把每一个类别中的音频的mfcc放在一起，基于隐码模型开始训练
test_x,test_y = GaussianHMM_x_y(test_samples)

# 使用七个模型，对每一个文件进行预测得分
pred_test_y = []
for mfccs in test_x:
    best_score,best_label = None,None
    for label,model in models.items():
        score = model.score(mfccs)
        if not best_score or best_score < score:
            best_score,best_label=score,label
    pred_test_y.append(best_label)

print(test_y,pred_test_y,sep="\n")
```

## 图像识别

#### OpenCV基础

opencv是一个开源的计算机视觉库. 提供了很多图像处理的常用工具.

案例:

```python
import numpy as numpy
import cv2 as cv

# 读取图片并显示
img = cv.imread('forest.jpg')
cv.imshow('Image', img)
# 显示图像中每个颜色通道的图像
print(img.shape)
# 0 保留蓝色通道 其他颜色值为0
blue = numpy.zeros_like(img)
blue[:,:,0] = img[:,:,0]
cv.imshow('blue', blue)

# 1 保留绿色通道 其他颜色值为0
green = numpy.zeros_like(img)
green[:,:,1] = img[:,:,1]
cv.imshow('green', green)

# 2 保留红色通道 其他颜色值为0
red = numpy.zeros_like(img)
red[:,:,2] = img[:,:,2]
cv.imshow('red', red)

# 图像裁剪
h, w = img.shape[:2]
l, t = int(w/4), int(h/4)
r, b = int(w*3/4), int(h*3/4)
cropped = img[t:b, l:r]
cv.imshow('cropped', cropped)

# 图像缩放
scaled = cv.resize(img, (int(w/4), int(h/4)),
	interpolation=cv.INTER_LINEAR)
cv.imshow('scaled', scaled)

scaled2 = cv.resize(scaled, (w, h),
	interpolation=cv.INTER_LINEAR)
cv.imshow('scaled2', scaled2)

# 图像文件的保存
cv.imwrite('blue.jpg', blue)

cv.waitKey()  # 阻塞方法, 按下键盘继续执行
```

#### 边缘检测

物体的边缘检测是物体识别的常用手段. 边缘检测常用亮度梯度方法, 通过识别亮度梯度变化最大的像素点从而检测出物体的边缘.

```python
# Canny边缘检测
# 50: 水平方向上的阈值   240: 垂直方向上的阈值
cv.Canny(img, 50, 240)
```

案例:

```python
import cv2 as cv
img = cv.imread('chair.jpg', 
				cv.IMREAD_GRAYSCALE)
cv.imshow('img', img)
# canny边缘检测
canny = cv.Canny(img, 50, 240)
cv.imshow('canny', canny)

# cv.Sobel() 索贝尔边缘检测
# cv.CV_64F: 做索贝尔偏微分时使用的数据精度
# 1: 水平方向做偏微分   0: 垂直方向不做偏微分
# ksize: 索贝尔卷积过程中使用卷积核的大小 5*5
hsobel=cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
cv.imshow('hsobel', hsobel)

cv.waitKey()
```

#### 亮度提升

opencv提供了直方图均衡化的方式实现亮度提升, 更有利于边缘识别与物体识别模型的训练.

```python
# 彩色图转灰度图
gray = cv.cvtcolor(img, cv.COLOR_BGR2GRAY)
# 直方图均衡化
equalized_gray = cv.equalizeHist(gray)
```

案例:

```python
import cv2 as cv

img = cv.imread('sunrise.jpg')
cv.imshow('img', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)
equalized_gray = cv.equalizeHist(gray)
cv.imshow('equalized_gray', equalized_gray)
# 对彩色图像提亮
# YUV: 亮度, 色度, 饱和度（鲜艳值）
yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
yuv[:,:,0] = cv.equalizeHist(yuv[:,:,0])
equalized_color = cv.cvtColor(yuv, cv.COLOR_YUV2BGR)
cv.imshow('equalized_color', equalized_color)

cv.waitKey()
```

#### 角点检测

对一个图像执行角点检测, 可以检测出平直楞线的交汇点. (求亮度梯度方向改变的像素点的位置)

```python
# Harris角点检测器
# gray: 灰度图像
# 边缘水平方向,垂直方向亮度梯度值改变超过阈值7/5时即为边缘.
# 边缘线方向改变超过阈值0.04弧度值即为一个角点.
corners = cv.cornerHarris(gray, 7, 5, 0.04)
```

案例:

```python
import cv2 as cv

img = cv.imread('box.png')
cv.imshow('img', img)
# 角点监测
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)
corners = cv.cornerHarris(gray, 7, 5, 0.04)
print(img.shape, corners.shape)
# 在图像中绘制角点
mixture = img.copy()
mixture[corners>corners.max()*0.01] = [0,0,255]
cv.imshow('mixture', mixture)

cv.waitKey()
```

#### 特征点检测

STAR特征点检测 / SIFT特征点检测

特征点检测结合了边缘检测与角点检测从而识别出物体的特征点.

STAR特征点检测

```python
import cv2 as cv
# 创建star特征点检测器
star = cv.xfeatures2d.StarDetector_create()
keypoints = star.detect(gray)
# cv提供了方法吧keypoint绘制在图像中
cv.drawKeypoints(img, keypoints, 
                 mixture, flags=
cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
```

SIFT特征点检测

```python
sift = cv.xfeatures2d.SIFT_create()
keypoints = sift.detect(gray)
```

案例:

```python
import cv2 as cv
img = cv.imread('table.jpg')
cv.imshow('img', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)
# 提取特征点
sift = cv.xfeatures2d.SIFT_create()
keypoints = sift.detect(gray)
mixture = img.copy()
cv.drawKeypoints(img, keypoints, 
                 mixture, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('mixture', mixture)
cv.waitKey()
```

#### 特征值矩阵

图像的特征值描述矩阵记录了图像的特征点以及每个特征点的梯度信息. 相似图像的特征值矩阵也相似. 如果有足够多的样本就可以基于隐马模型进行图像内容的识别.

```python
keypoints = sift.detect(gray)
# desc即为图像的特征值矩阵
_, desc = sift.compute(gray, keypoints)
```

#### 物体识别

1. 读取每个图片文件, 加载每个文件的特征值描述矩阵, 整理训练集, 与某个类别名绑定在一起.
2. 基于隐马模型, 对三个类别的特征值描述矩阵训练集进行训练, 得到3个隐马模型, 分别用于识别三个类别.
3. 对测试集分别进行测试, 取得分高的为最终预测类别.

```python
import os
import numpy
import cv2 as cv
import hmmlearn.hmm as hh

def search_files(directory,suffix_name):
    #对文件路径进行跨平台处理
    directory = os.path.normcase(directory)
    # {'apple':[dir,dir,dir], 'banana':[dir..]}
    result = {}
    #当前目录、子目录、文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            #找到对应文件后缀名
            if file.endswith(suffix_name):
                label = root.split(os.path.sep)[-1]
                if label not in result:
                    result[label] = []
                path = os.path.join(root,file)
                result[label].append(path)
    return result

def GaussianHMM_x_y(files):
    train_x, train_y = [], []
    for label, filenames in files.items():
        x_result = numpy.array([])
        for filename in filenames:
            img = cv.imread(filename)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            #对图片进行范围缩放
            h,w = gray.shape[:2]
            f = 200 / min(h,w)
            #fx:x轴伸缩比例 fy:y轴伸缩比例
            gray = cv.resize(gray,None,fx=f,fy=f)

            sift = cv.xfeatures2d.SIFT_create()
            keypoints = sift.detect(gray)
            _,desc = sift.compute(gray,keypoints)

            if len(x_result) == 0:
                x_result = desc
            else:
                x_result = numpy.append(x_result, desc, axis=0)

        train_x.append(x_result)
        train_y.append(label)
    return (train_x,train_y)

train_samples = search_files("objects/training",".jpg")
# 整理训练集, 把每一个类别中的图片的desc摞在一起, 基于隐马模型开始训练.
train_x,train_y = GaussianHMM_x_y(train_samples)

# 基于隐马模型进行训练, 把所有类别的模型都存起来,一共3个类别循环3次
models = {}
for x_result,label in zip(train_x,train_y):
    model = hh.GaussianHMM(n_components=4,
        covariance_type="diag",n_iter=200)
    models[label] = model.fit(x_result)

# 读取测试集中的文件，使用每个模型对文件进行评分
#取分直大的模型对于的label作为预测类别
test_samples = search_files("objects/testing",".jpg")
#整理训练集，把每一个类别中的音频的mfcc放在一起，基于隐码模型开始训练
test_x,test_y = GaussianHMM_x_y(test_samples)

# 使用3个模型, 对每一个文件进行预测得分.
pred_test_y = []
# test_x一共3个样本, 遍历3次, 每次验证1个文件
for x_result in test_x:
    best_score,best_label = None,None
    for label,model in models.items():
        score = model.score(x_result)
        if not best_score or best_score < score:
            best_score,best_label=score,label
    pred_test_y.append(best_label)

print(test_y,pred_test_y,sep="\n")
```

## 人脸识别

人脸识别与图像识别的区别在于人脸识别需要识别出两个人的不同点. 眉间距离, 鼻子位置.眼睛位置等等..

opencv的视频捕捉

opencv提供了访问视频捕捉设备的API(摄像头), 从而获取图像帧.

```python
import cv2 as cv
# 获取视频采集设备   下标为0的摄像头
videoCapture = cv.VideoCapture(0)
# 获取采集到的第一张图片(第一帧)

while True:
	frame = videoCapture.read()[1]
	cv.imshow('frame', frame)
	# 每33毫秒自动解除阻塞,27是退出键的值
	if cv.waitKey(33) == 27:
		break

# 释放视频设备
videoCapture.release()
cv.destroyAllWindows()  # 销毁所有窗口
```

### 人脸定位

哈尔级联人脸定位

```python
fd = cv.CascadeClassifier('../xxx/face.xml')
# frame, 图像
# 1.3: 最小的人脸尺寸
# 5: 最多找5张脸
faces = fd.detectMultiScale(frame, 1.3, 5)
```

案例:

```python
import cv2 as cv

fd = cv.CascadeClassifier('../ml_data/haar/face.xml')
ed = cv.CascadeClassifier('../ml_data/haar/eye.xml')
nd = cv.CascadeClassifier('../ml_data/haar/nose.xml')

# 获取视频采集设备   下标为0的摄像头
videoCapture = cv.VideoCapture(0)
# 获取采集到的第一张图片(第一帧)
while True:
	frame = videoCapture.read()[1]
	# 通过哈尔定位器 进行局部定位 并在图像中标注
	faces = fd.detectMultiScale(frame, 1.3, 2)
	for l, t, w, h in faces:
		# 绘制椭圆
		a, b = int(w/2), int(h/2)
		cv.ellipse(frame, 
			(l+a, t+b), # 椭圆心坐标
			(a, b), # 椭圆半径
			0, 0, 360, (255,0,255), 
			2 # 椭圆的线宽
		)
		# 绘制鼻子和眼睛
		face = frame[t:t+h, l:l+w]
		eyes = ed.detectMultiScale(face, 1.3, 5)
		for l, t, w, h in eyes:	
			a, b = int(w/2), int(h/2)
			cv.ellipse(face, 
				(l+a, t+b), # 椭圆心坐标
				(a, b), # 椭圆半径
				0, 0, 360, (0,255,255), 
				2 # 椭圆的线宽
			)
		
		nodes = nd.detectMultiScale(face, 1.3, 5)
		for l, t, w, h in nodes:	
			a, b = int(w/2), int(h/2)
			cv.ellipse(face, 
				(l+a, t+b), # 椭圆心坐标
				(a, b), # 椭圆半径
				0, 0, 360, (255,255,0), 
				2 # 椭圆的线宽
			)
	cv.imshow('frame', frame)
	# 每33毫秒自动解除阻塞
	if cv.waitKey(33) == 27:
		break

# 释放视频设备
videoCapture.release()
cv.destroyAllWindows()  # 销毁所有窗口
```

### 人脸识别

简单人脸识别:  opencv的LBPH(局部二值模式直方图)

```python
import os
import numpy
import cv2 as cv
import sklearn.preprocessing as sp
fd = cv.CascadeClassifier('haar/face.xml')

def search_files(directory,suffix_name):
	#对文件路径进行跨平台处理
	directory = os.path.normcase(directory)
	# {'apple':[dir,dir,dir], 'banana':[dir..]}
	result = {}
	#当前目录、子目录、文件
	for root, dirs, files in os.walk(directory):
		for file in files:
			#找到对应文件后缀名
			if file.endswith(suffix_name):
				label = root.split(os.path.sep)[-1]
				if label not in result:
					result[label] = []
				path = os.path.join(root,file)
				result[label].append(path)
	return result

def GaussianHMM_x_y(files):
	train_x, train_y = [], []
	for label, filenames in files.items():
		for filename in filenames:
			gray = cv.imread(filename,cv.IMREAD_GRAYSCALE)
			faces = fd.detectMultiScale(gray, 1.1, 2,
										minSize=(100, 100))
			# 把这张脸加入训练集的输入 label作为样本的输出
			for l, t, w, h in faces:
				train_x.append(gray[t:t + h, l:l + w])
				train_y.append(
					codec.transform([label])[0])
	train_y = numpy.array(train_y)
	return (train_x,train_y)

train_faces = search_files('faces/training',".jpg")
# 类别名称标签编码器
codec = sp.LabelEncoder()
codec.fit(list(train_faces.keys()))
train_x, train_y = GaussianHMM_x_y(train_faces)
# 基于局部二值模式直方图做人脸识别模型
model = cv.face.LBPHFaceRecognizer_create()
model.train(train_x, train_y)
#模型训练完毕  模型预测
test_faces = search_files('faces/testing',".jpg")
test_x, test_y = GaussianHMM_x_y(test_faces)

pred_test_y = []
for face in test_x:
	pred_code = model.predict(face)[0]
	pred_test_y.append(pred_code)

print(codec.inverse_transform(test_y),codec.inverse_transform(pred_test_y),sep="\n")
```
---

# Git基础命令

- 初始化仓库：git init

  - 将某个项目目录变维git操作目录，生成git本地仓库。即该项目目录可以使用git管理

- 查看本地仓库状态：git status

  - 初始化仓库后默认工作在master分支，当工作区与仓库区不一致时会有提示。

- 将工作内容记录到暂存区：git add [files…]

  - 例如：将所有文件（不包含隐藏文件）记录到暂存区

    git add *

- 撤回某个文件暂存记录：git rm –cached [files…]

- 将文件同步到本地数据库：git commit -m [message]

  - -m表示添加一些同步信息，表达同步内容

  - 例如：将暂存区所有记录同步到仓库区

    git commit -m “add files”

- 查看commit日志记录：git log       或 git log –pretty=online
  
- 比较工作区文件和仓库文件的差异：

- 放弃工作区文件修改：git checkout – [file]

- 从仓库区恢复文件：git checkout [file]

- 移动或者删除文件：记录到暂存区

  - git mv [file] [path]
  - git rm [files] 

---

## 版本控制

1. 退回到上一个commit节点

   git reset –hard HEAD^		==一个^表示回退一次==

------

## 远程仓库交互

~~~scala
<!---->
tarena@tedu:~$ mkdir gitrepo
tarena@tedu:~$ chown tarena:tarena gitrepo
//将该目录初始化为git共享目录(.git为固定写法)
tarena@tedu:~$ cd gitrepo
tarena@tedu:~/gitrepo$ git init --bare tedu.git
初始化空的 Git 仓库于 /home/tarena/gitrepo/tedu.git/
//将git配置目录与项目目录设置为相同的属主
tarena@tedu:~/gitrepo$ chown -R tarena:tarena tedu.git
//客户端添加远程仓库到主机(origin要添加的远程仓库名自命名)
tarena@tedu:~/gitrepo$ git remote add origin tarena@127.0.0.1:/home/tarena/gitrepo/tedu.git
//删除远程主机
tarena@tedu:~/gitrepo$ git remote rm [origin]
//查看连接的主机
tarena@tedu:~/gitrepo$ git remote
origin
//将本地master分支推送给远程仓库
tarena@tedu:~/gitrepo$ git push -u origin master
~~~

# 爬虫案例

==SpyderHelper==

~~~python
import logging
LOGGER = logging.getLogger("SpyderHelper")
# 代表该类是否要进行测试
def random_user_agent():
    """
    获取随机User-Agent
    :return:随机的一个User-Agent
    """
    import random
    user_agents = [{
        "User-Agent": "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50"},
        {
            "User-Agent": "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50"},
        {
            "User-Agent": "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0"},
        {
            "User-Agent": "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)"},
        {
            "User-Agent": "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)"},
        {
            "User-Agent": "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)"},
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1"},
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1"},
        {
            "User-Agent": "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11"},
        {
            "User-Agent": "Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11"},
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11"},
        {
            "User-Agent": "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Maxthon 2.0)"}]
    return random.choice(user_agents)
def data_to_map(strings):
    """
    浏览器的header字符串转换为map格式
    :param strings: 浏览器的header字符串
    :param map: 要更改的属性值:{}
    :return:已打包好的数据格式
    """
    data = dict()
    for line in strings.strip().split("\n"):
        key, value = re.split(r":",line)
        data[key.strip()] = value.strip()
    logging.warning(data)

def urlencode(map,encoding="utf-8"):
    """
    把提交的网页数据转码并转为bytes数据类型
    :param map:正常网页数据
    :param encoding: 默认以utf-8转为bytes
    :return 网页转码并转为bytes数据类型
    """
    from urllib import parse
    return parse.urlencode(map).encode(encoding)
~~~

------

## 有道翻译

+ 链接中translate_o?的_o去除就不会去验证加密

### request:post请求

~~~python
import SpyderHelper,json
from urllib import request

url = "http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule"
key = input("请输入要翻译的文字:")
data = {'client': 'fanyideskweb','smartresult': 'dict',
        'sign': '0414e194569520f1317134338051e412','salt': '15561524595357',
        'ts': '1556152459535','version': '2.1',
        'action': 'FY_BY_REALTlME','i': key, 'to': 'AUTO',
        'from': 'AUTO','doctype': 'json',
        'keyfrom': 'fanyi.web','bv': '94d71a52069585850d26a662e1bcef22'}
headers = SpyderHelper.random_user_agent()
post = request.Request(url,headers=headers,data=SpyderHelper.urlencode(data))
response = request.urlopen(post)
html = response.read().decode("utf-8")
# html为json格式的字符串
json_html = json.loads(html)
translate_result = json_html['translateResult'][0][0]["tgt"]
print("翻译的结果的是:",translate_result)
~~~

------

### requests:post请求

​		*requests获取响应的方法，会自动 把提交的网页数据转码并转bytes数据类型*

~~~python
import requests,SpyderHelper
"""同request：post"""
response = requests.post(url,headers=headers,data=data)
response.encoding = response.apparent_encoding
json_html = response.json()
"""同request：post"""
~~~

------

