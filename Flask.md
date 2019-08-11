# 基础知识

## WEB SERVER

1. Flask - 轻量级WEB框架 
2. AJAX - WEB中异步技术
3. Django - 重量级WEB框架

## 网页

+ 静态网页：无法与服务器进行交互的网页
+ 动态网页：能够与服务器进行交互的网页

## WEB 与 服务器

> WEB : 网页(HTML,CSS,JS)

> 服务器：能够给用户提供服务的计算机

> 硬件：一台主机
> 软件：一个能够接收用户请求并给出响应的程序

+ 系统软件
  + Linux
  + Windows Server
+ 应用软件:
  + APACHE 
  + TOMCAT
  + IIS - Internet Information Service
  + Nginx
+ 应用软件作用：
  		1.存储WEB上的信息 
    		2.能够处理用户的请求(request)并给出响应(response)
    		3.执行处理程序
    		4.具备一定的安全功能

## 框架

> 框架是一个为了解决开放性问题而存在的程序结构

> Python WEB 框架

+ Flask - 轻量级框架
+ Webpy - 轻量级框架
+ Tornado - 异步框架
+ Django - 重量级框架

### 框架模式

​	1.MTV
​		M:Models,模型层,与数据库打交道 - ORM
​		T:Templates,模板层,前端页面
​		V:Views,视图层，处理与用户打交道的内容
​	2.MVC
​		M:Models,模型层,与数据库打交道 - ORM
​		V:Views,视图层,前端页面
​		C:Controller,控制器,处理请求和响应

			M --- M
			T --- V
			V --- C
# Flask框架



> Flask是一个基于Python并且依赖于Jinja2模板引擎和Werkzeug WSGI 服务的一个微型框架(Micro Framework)
> 		

+ WSGI:Web Server Gateway Interface
+ WEB 服务网关接口
+ 官网: http://flask.pocoo.org

## 安装Flask

+ 查看已安装的Flask版本

~~~python
import flask
flask.__version
~~~

+ 安装Flask	
  sudo pip3 install flask == 1.0.2

## 路由（route）

> 路由是为了匹配用户的请求地址以及对应的视图函数

~~~python
@app.route('/地址')
def funName():
	"""业务实现"""
	return "" # 响应给客户端浏览器的内容
~~~

**示例**

~~~python
1. http://127.0.0.1:5000
@app.route('/')
def index():
    return "这是首页"
			
2. http://127.0.0.1:5000/admin/login
@app.route('/admin/login')
def login():
    return "xxxx"
~~~

### 参数的路由实现

> http://127.0.0.1:5000/show/lvzemaria
> http://127.0.0.1:5000/show/laowei
> http://127.0.0.1:5000/show/wangdanbo

+ 带多个参数的路由

~~~python
from flask import Flask
app = Flask(__name__)
#num1 表示的就是路由中的num1参数值
@app.route("/calculate/<num1>/<num2>")
def show(num1,num2):
    return str(int(num1) + int(num2))

if __name__ == '__main__':
    app.run(debug=True)#开启调试模式后代码改正会显示
~~~

### 指定参数类型的路由

~~~python
@app.route('/show/<int:num1>')
def show(num1):
    num1 允许直接当成整数去处理
~~~

Flask中所支持的类型转换器:

| 类型转换器 | 作用                   |
| ---------- | ---------------------- |
| 缺省       | 字符串,不能包含斜杠(/) |
| int:       | 整数                   |
| float:     | 浮点数                 |
| path:      | 字符串,允许包含斜杠(/) |

### 多url的路由匹配

> 多个访问地址最终匹配到同一个视图处理函数

**示例**

> http://127.0.0.1:5000
> http://127.0.0.1:5000/index
> http://127.0.0.1:5000/数字
> http://127.0.0.1:5000/index/数字

~~~python
from flask import Flask
app = Flask(__name__)
@app.route("/")
@app.route("/index")
@app.route("/<int:num1>")
@app.route("/index/<int:num1>")
def show(**kwargs):
    index = kwargs.get("num1")
    if index != None:
        return "当前的页数为{}页".format(index)
    else:return "当前为主页"

if __name__ == '__main__':
    app.run(debug=True)
~~~

## 模板（Templates）

> 模板就是能够呈现给用户看的html+python的网页结构
> 在模板中，允许包含"变量"以及"标签"表示动态内容

> Flask中的模板时依赖于 Jinja2 的模板引擎

+ Jinja2的文档地址:http://jinja.pocoo.org/

> 模板的配置

+ 默认情况下，Flask会到项目中找到一个 templates 的文件夹，去搜索要用到的模板

> 显示模板

+ 作用：在视图中，将模板文件(xx.html)先渲染成字符串
  			再将字符串响应给客户端
+ 函数

~~~python
from flask import render_template
return render_template('模板名称.html')
~~~

### 语法规范

#### 变量

> 在模板中，变量是一种占位符,告诉模板该位置的值是从哪个数据中读取出来的

**语法**

+ 在视图中对变量的处理
  + 渲染模板时，要指定带到模板上的变量们
  + return render_template('xxx.html',变量1=值1,... ...)

~~~python
@app.route("/")
def show():
    return render_template("01-var.html", name='MM.ZH', age=18)
~~~

+ 在模板中显示变量的值
  + 将变量名对应的值输出在指定的位置处
  + {{变量名}} ------------页面标签

### 标签

> 每个标签表示的是服务器端一段独立的功能

**语法：**

> {% 标签内容 %}
> {% 结束标签 %}  (结束标签要根据需求灵活选择)

#### if标签

~~~python
{% if 条件1 %}
{% elif 条件2 %}
{% else %}
{% endif %}
~~~

~~~python
class Animal(object):
    name = None

    def eat(self):
        return self.name + "正在吃饭"
@app.route("/")
def show():
    # 字符串
    name = "小泽Maria"
    # 数字
    age = 16
    # 元组
    tup = ("大肠刺身", "小肠刺身")
    # 字典
    dic = {
        "MSN": "美少女战士",
        "XMX": "巴拉巴拉小魔仙",
    }

    dog = Animal()
    dog.name = "猫咪"
    return render_template("02-var.html",params = locals())
~~~

**02-var.html**

~~~html
<body>
    <h1>姓名:{{ params.name }}</h1>
    <h1>
        年龄:{{ params.age }}
        (
            {% if params.age <= 18 %}
                未成年{{ params.age }}
            {% elif params.age >= 50 %}
                老年人
            {% else %}
                青年人
            {% endif %}
        )
    </h1>
    <h1>食物:{{ params.tup }}</h1>
    <h1>食物[0]:{{ params.tup[0] }}</h1>
    <h1>食物[1]:{{ params.tup.1 }}</h1>
    <h1>电视:{{ params.dic }}</h1>
    <h1>电视['XMX']:{{ params.dic['XMX'] }}</h1>
    <h1>电视.MSN : {{ params.dic.MSN }}</h1>
    <h1>宠物:{{ params.dog }}</h1>
    <h1>宠物.name:{{ params.dog.name }}</h1>
    <h1>宠物.eat():{{ params.dog.eat() }}</h1>
</body>
~~~

#### for标签

~~~python
{% for 变量 in 可迭代元素 %}
	变量是服务器端内容
{% endfor %}
~~~

+ 内部变量-loop

  **作用：**

  > 无需声明，直接使用
  >
  > 表示本次循环中的一些相关信息

  ~~~python
  loop.index
      表示当前循环的次数，从1开始记录
  loop.index0
      表示当前循环的次数，从0开始记录
  loop.first
      表示是否为第一次循环
      值为True,表示是第一次循环
  loop.last
      表示是否为最后一次循环
      值为True,表示是最后一次循环
  ~~~

~~~python
class User:
    def __init__(self,name,age,gender):
        self.name = name
        self.age = age
        self.gender = gender
        
@app.route("/")
def show():
    return render_template("03-for.html",list = [User("A",18,"男"),User("B",10,"女")])
~~~

~~~html
<table border="1" width="300">
    <tr>
        <th>姓名</th>
        <th>年龄</th>
        <th>性别</th>
    </tr>
    {% for user in list %}
    <tr
        {% if loop.first %}
        class="c1"
        {% elif loop.last %}
        class="c2"
        {% else %}
        class="c3"
        {% endif %}
        >
        <td>{{ user.name }}</td>
        <td>{{ user.age }}</td>
        <td>{{ user.gender }}</td>
    </tr>
    {% endfor %}
</table>
~~~

+ 模板中声明函数

  ~~~python
  使用 {% macro%} ... {% endmacro%} 声明宏
  {% macro 名称(参数列表) %}
  	xxxx xxxxx
  {% endmacro%}
  ~~~

  **在独立的文件中声明宏**

  > 推荐将宏们放在统一的文件中进行管理

  1. 创建 macro.html 模板文件，声明所有的宏

  2. 在使用的网页中,导入宏模板文件(macro.html)

     ~~~python
     {% import 'macro.html' as macros %}
     ~~~

### 静态文件

> 不与服务器做动态交互的文件一律是静态文件

**静态文件的处理**

1. 在项目工程目录中创建一个 static 文件夹

   > 作用:为了存放所有的静态资源

2. 所有的静态文件必须通过 /static/ 路径访问

   >/static : 表示 static后的子路径将到static文件夹中继续搜索

### 模板的反向解析

~~~html
from flask import url_for
url_for（'img/a.jpg'）
<img src="{{url_for('static',filename='img/a.jpg')}}" >
~~~

### 模板的继承

> 如果一个模板中的内容大部分与另外一个模板一样时,可以使用继承的方式简化模板开发

#### 父模块文件

> 需要定义出哪些内容在子模板中是可以被重写的

**语法**

~~~python
{% block 块名 %}
定义在父模板中要正常显示的内容
{% endblock %}
~~~

**示例**

~~~HTML
{% block title %}
这是parent中显示的内容
{% endblock %}
{% block content %}
<h1>这是parent允许被修改的内容</h1>
{% endblock %}
~~~

#### 子模块文件

> 需要指定继承自哪个模板，可重写父模板

~~~python
{% extends '父模板名称' %}
~~~

**示例**

~~~html
{% extends '04-parent.html' %}

{% block content %}
    <h1 style="color:red">
    这是在05-child中被修改的内容
    </h1>
{% endblock %}
~~~

## 修改配置

+ 构建Flask应用时指定配置信息

~~~python
app = Flask(
    __name__,
    template_folder="muban",#指定存放模板的文件夹名称
    static_url_path="/s",	#指定访问静态资源的路径
    static_folder="sta"		#指定保存静态资源的完整路径
)
~~~

+ 启动程序时的运行配置

~~~python
app.run(
    debug=True,
    port=5555,
    host='0.0.0.0'#指定访问到本项目的地址,0.0.0.0 表示局域网内任何机器都可以访问当本项目
)
~~~

## HTTP协议

> HTTP:Hyper Text Transfer Protocol
> 规范了数据是如何打包以及传递的

+ 请求消息

  > 由客户端带给服务器端的消息

  ~~~python
  #请求起始行	GET / HTTP/1.1
  1.请求方式 - GET
  2.请求资源路径 - /
  3.HTTP协议及版本 - HTTP/1.1
  
  #请求消息头
  所有以key:value格式存在的内容都是消息头
  #【如Referer:记录请求源地址】
  每个消息头都是要传递给服务器的信息
      
  #请求主体
  只有post,put请求方式才有请求主体
  ~~~

+ 响应消息

  > 由服务器端带给客户端的消息

  ~~~python
  #响应起始行	HTTP/1.1 200 OK
  1.协议以及版本号-HTTP/1.1
  2.响应状态码-200
  3.原因短句 - OK
  
  #响应消息头
  以 key:value 格式存在的内容
  服务器要传递给浏览器的信息
  
  #响应主体
  服务器端响应回来的数据
  ~~~

### request

> request对象中会封装所有与请求相关的信息
> 如:请求数据,消息头,路径,... ...

**语法**

~~~python
from flask import request
~~~

**常用属性**

| 属性名    | 功能                                     |
| --------- | ---------------------------------------- |
| scheme    | 获取请求协议                             |
| method    | 获取本次请求的方式                       |
| args      | 获取使用get请求所提交的数据              |
| form      | 获取使用post请求所提交的数据             |
| cookies   | 获取cookies相关数据                      |
| files     | 获取上传的所有文件                       |
| path      | 获取请求的资源路径（不包含请求参数）     |
| full_path | 获取请求的资源的完整路径（包含请求参数） |
| url       | 获取完整的请求地址，从协议处开始         |
| headers   | 获取请求头的信息                         |

**示例**

~~~python
#localhost:5000/03-request
#目的:查看request中的成员
@app.route('/03-request')
def request_views():
  headers = request.headers #请求消息头

  referer = request.headers.get('Referer','/')

  return render_template('03-request.html',params=locals())

#目的:在03-request中查询请求源地址 - Referer
@app.route('/04-referer')
def referer_views():
  return "<a href='/03-request'>去往03-request</a>"

#目的:接收get请求提交的数据
@app.route('/05-get')
def get_views():
  name = request.args['name']
  age = request.args['age']
  return "<h1>提交的数据:name:%s,age:%s</h1>" % (name,age)

#localhost:5000/06-post
#目的:接收post请求的数据
#需要在路由中设置method（不写默认只有get方式）
@app.route('/06-post',methods=['POST','GET'])
def post_views():
  #判断请求方式是get还是post
  if request.method == 'GET':
    return render_template('06-post.html')
  else:
    #接收post请求方式所提交的数据
    uname=request.form.get('uname')
    upwd = request.form.get('upwd')
    return "<h1>uanme:%s,upwd:%s</h1>" % (uname,upwd)
~~~

**03-request.html**

~~~html
<h2>
    <a href="/05-get?name=bobo&age=40">去往/05-get</a>
</h2>
<h2>
	<!--
    服务器端:referer=request.headers.get('Referer','/')
    返回:
    1.从哪个地址来的,则返回到哪个地址取
    2.手动输入地址进来的,则返回到首页
	-->
	<a href="{{params.referer}}">返回</a>
</h2>
~~~

**06-post.html**

~~~html
<form action="/06-post" method="post">
  <p>用户名称: <input type="text" name="uname"></p>
  <p>用户密码: <input type="password" name="upwd"></p>
  <p><input type="submit"></p>
</form>
~~~

### 加密

~~~python
# 目的:在03-request中查询请求源地址 - Referer
@app.route('/', methods=["GET", "POST"])
def referer_views():
    if request.method == "GET":
        return render_template("06-post.html")
    else:
        #加密
        pwd = generate_password_hash(request.form["upwd"])
        #解密
        result = check_password_hash(pwd,"123456")

        return '密码为123456' if result else '密码不是123456'
    return ""
~~~

## 文件上传

> 文件一定要放在表单中上传,表单格式如下所示

~~~html
<form action="/" method="post" enctype="multipart/form-data">
     用户头像:<input type="file" name="uimg">
</form>
~~~

> 文件上传首先会传到"缓存区"(服务器),我们需要将文件从缓存区中取出来保存到指定的位置处

~~~python
try:
    uimg = request.files["uimg"]
    import datetime
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%s%f")
    filename = "%s.%s"%(time,uimg.filename.split(".")[-1])
    uimg.save("static/"+filename)
except Exception as ex:
    print(ex)
    return "上传文件失败"
return "上传成功"
~~~

## ORM框架

> 对象关系映射：ORM - Object Relational Mapping
> 简称:ORM , O/RM , O/R Mapping

> ORM的三大特征

1. 数据表(Table)到编程类(class)的映射
2. 数据类型的映射
3. 关系映射
   1. 一对一关系
   2. 一对多关系
   3. 多对多关系

> ORM的优点

1. 封装了数据库中所有操作，提升效率。
2. 可以省略庞大的数据访问层,即便不用SQL编码也能完成对数据库的CRUD操作。

| C      | U      | R        | D      |
| ------ | ------ | -------- | ------ |
| Create | Update | Retrieve | Delete |

### SQLAlchemy框架

> 安装SQLAlchemy

+ pip3 install sqlalchemy
+ pip3 install flask-sqlalchemy

>配置数据库

~~~python
from flask_sqlalchemy import SQLAlchemy
import pymysql

pymysql.install_as_MySQLdb()
app = Flask(__name__)
# 为app指定数据库的连接信息
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://root:123456@localhost:3306/stock"
#设置数据库的信号追踪
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
#设置执行完视图函数后自动提交操作回数据库
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
#设置程序的启动模式为调试模式
app.config['DEBUG']=True

# 创建SQLAlchemy的实例-db,以后在程序中通过db来操作数据库
db = SQLAlchemy(app)
~~~

## 模型

> 根据数据库中的表结构而创建出来的类(模型类,实体类)

**语法**

~~~python
class MODELNAME(db.Model):
    __tablename__="TABLENAME"
	COLUMN_NAME = db.Column(db.TYPE,OPTIONS)
    #OPTIONS：列选项
~~~

| 类型名       | 说明           |
| ------------ | -------------- |
| Integer      | 普通整数       |
| SmallInteger | 小范围整数     |
| BigInteger   | 不限精度整数   |
| Float        | 浮点数         |
| Numeric      | 定点数         |
| String       | 字符串         |
| Text         | 字符串         |
| Boolean      | 布尔值         |
| Date         | 日期类型       |
| Time         | 时间类型       |
| DateTime     | 日期和时间类型 |

**OPTIONS**

| 选项名        | 说明               |      |
| ------------- | ------------------ | ---- |
| autoincrement | Ture自动增长       |      |
| primary_key   | （True）该列为主线 |      |
| unique        | （True）该列为唯一 |      |
| index         | （True）该列加索引 |      |
| nullable      | （True）该列可为空 |      |
| default       | 指定默认值         |      |

如果一个列中要包含多个列选项,使用 ',' 隔开即可，列选项 与 列类型之间使用 ',' 隔开的。

### 表模型

> 将所有的实体类都创建到数据库上，只有类对应的表不存在时才会创建,否则无反应。

+ db.create_all()

> 将所有的数据表全部删除

+ db.drop_all()

~~~python
class Users(db.Model):
  #省略__tablename__,默认映射的表名就是将类变成全小写
  id = db.Column(db.Integer,primary_key=True)
  username = db.Column(
    db.String(80),unique=True,index=True
  )
  age = db.Column(db.Integer,nullable=True)
  email = db.Column(
    db.String(120),unique=True
  )
  #增加字段 isActive,表示用户是否被激活,布尔类型,默认为True
  isActive = db.Column(db.Boolean,default=True)

# db.create_all()
db.drop_all()
~~~

### 数据库迁移

> 将实体类的改动再映射回数据库

> 依赖第三方库

+ flask-script
  + Manager：对项目进行管理，如：启动项目，添加命令等。
  + python3 xxx.py runserver
  + python3 xxx.py runserver --port 5555
  + python3 xxx.py runserver --host 0.0.0.0
  + python3 xxx.py runserver --host 0.0.0.0 --port 5555
+ flask-migrate
  + Migrate：用于管理app和db之间的协调关系
  + MigrateCommand：允许在终端中提供实体类的迁移命令

> 实现数据库的迁移

1. 项目和数据库的初始化操作
   + python3 xxx.py db init
2. 将编辑好的实体类生成中间文件并保存在migrations文件夹中
   + python3 xxx.py db migrate
   + 特点:只要检测到实体类有更改,就会生成中间文件
3. 将中间文件映射回数据库
   + python3 xxx.py db upgrade

~~~python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_script import Manager
from flask_migrate import Migrate,MigrateCommand

app = Flask(__name__)
#设置连库字符串
app.config['SQLALCHEMY_DATABASE_URI']="mysql+pymysql://root:123456@localhost:3306/flask"
#设置数据库的信号追踪
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
#设置执行完视图函数后自动提交操作回数据库
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
#设置程序的启动模式为调试模式
app.config['DEBUG']=True
#创建db
db = SQLAlchemy(app)

#创建Manager对象并指定要管理哪个应用(app)
manager = Manager(app)
#创建Migrate对象,并指定关联的app和db
migrate=Migrate(app,db)
#为manager增加命令,允许做数据库的迁移操作
#为manager绑定一个叫 db 的子命令,该子命令执行操作由MigrateCommand来提供
manager.add_command('db',MigrateCommand)

if __name__ == '__main__':
  manager.run()
~~~

### 数据库操作

#### 增加 - C(Create)

+ 创建实体类对象,并为对象的属性赋值

+ 将实体对象保存回数据库

  + db.session.add(user)

    > 针对非查询操作,必须手动将操作提交回数据库

    db.session.commit()#提交回数据库(可在config配置)

+ 配置自动提交操作或数据库

  + app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN']=True

#### 删除 - D

1. 查询出要删除的实体对象
2. db.session.delete(对象)

#### 查询 - R(Retrieve)

##### 基于db.session进行查询

+ db.session.query()-----------返回一个Query对象,类型为BaseQuery

```python
1.查询 Users实体类中的id,username,age					db.session.query(Users.id,Users.username,Users.age)
2.查询Users实体类中所有的列
db.session.query(Users)
3.查询Users实体类以及Wife实体类中所有列
db.session.query(Users,Wife)
```
+ 执行查询函数
  + Query对象.查询执行函数

| 函数           | 说明                                           |
| -------------- | ---------------------------------------------- |
| all()          | 以列表的方式返回所有数据                       |
| first()        | 以实体对象的方式返回查询数据的第一条，默认None |
| first_or_404() | 效果同上，没数据则响应404                      |
| count()        | 返回查询结果的数量                             |

+ 查询过滤函数

  > db.session.query().查询过滤器函数().查询执行函数()

| 过滤器函数  | 作用               |
| ----------- | ------------------ |
| filter()    | 指定查询条件       |
| filter_by() | 等值查询条件中使用 |
| limit()     | 获取前几行数据     |
| offset      | 指定结果偏移量     |
| orderby()   | 排序               |
| groupby()   | 分组               |

**示例**

~~~python
#1.查询年龄大于17岁的Users的信息
db.session.query(Users).filter(Users.age>17).all()

#2.查询年龄大于17岁并且id大于1的Users的信息
语法1:
    db.session.query(Users)
			.filter(Users.age>17).filter(Users.id > 1)
			.all()
语法2:#使用 , 连接多个条件 - and
db.session.query(Users).filter(Users.age>17,Users.id>1).all()

#3.查询Users年龄大于17或者id>1的信息
from sqlalchemy import _or
xxx.filter(or_(条件1,条件2))

#4.查询Users中id为2的用户的信息,特征:等值查询
db.session.query(Users).filter(Users.id==2).first()

#5.select * from users where email like '%wang%'
db.session.query(Users).filter(Users.email.like("%wang%"))

#6.模糊查询 - in
#查询Users实体中年龄是17,30,45岁的用户的信息
users = db.session.query(Users).filter(
			Users.age.in_([17,30,45])).all()
#7.模糊查询 - between and
#查询年龄在30-45岁之间的用户的信息
users=
db.session.query(Users).filter(
    Users.age.between(30,45) ).all()

#8.获取前5条数据select * from users limit 5
db.session.query(Users).limit(5).all()

#9.获取 users 表中跳过前3条取剩余前5条
#select * from users limit 3,5
db.session.query(Users).offset(3).limit(5).all()

#10.Users实体中大于18岁的人,按年龄降序排序,如果年龄相同按id升序排序
db.session.query(Users).filter(Users.age > 18).order_by("age desc,id").all()
~~~

**filter_by()**

> 只能做单表的等值条件过滤筛选
> 1.不用实体类.属性,而直接用属性名即可
> 2.等值判断使用 = , 而不是 ==

~~~python
#查询Users实体中isActive为True的信息
users=db.session.query(Users).filter_by(
    isActive=True).all()
~~~

+ 聚合查询

~~~python
from sqlalchemy import func
~~~

| 聚合函数 | 说明   |
| -------- | ------ |
| sum()    | 求和   |
| count()  | 计数   |
| max()    | 最大值 |
| min()    | 最小值 |
| avg()    | 平均值 |

~~~python
#查询Users实体中所有人的平均年龄是多少
db.session.query(func.avg(Users.age)).all()
#Users实体中,按isActive分组求每组人数
db.session.query(func.count(Users.age))
	.group_by('isActive').all()
#查询users表中按isActive分组后, 组内人数大于2人的组的信息
db.session.query(Users.isActive,func.count(Users.id))
	.group_by("isActive").having(func.count(Users.id) 	  >2).all()
~~~

##### 基于实体类的查询

> 实体类.query.查询过滤器函数().查询执行函数()

~~~python
#1.查询Users实体中所有的数据
Users.query.all()
#2.查询Users实体中isActive为True的数据
Users.query.filter_by(isActive=True).all()		Users.query.filter(Users.isActive==True).all()
~~~

### 关系映射

#### 多对一

+ “多”实体类中

~~~python
class Teacher(db.Model):
  #增加对Course(一)的外键引用
  course_id=db.Column(
    db.Integer,
    db.ForeignKey('course.id'),
    nullable=True
  )
~~~

+ “一“实体类中

~~~python
class Course(db.Model):
  #增加关联属性和反向引用关系属性
  teachers = db.relationship(
    'Teacher',
    backref='course',
    lazy="dynamic"
  )
~~~

+ 具体使用方式

~~~python
#1.查询"爬虫"课程以及对应的授课老师
  course = Course.query.filter_by(cname='爬虫').first()
  print("课程名称:"+course.cname)
  # print(type(course.teachers))
  # course.teachers : 是针对course课程的对应的所有的授课老师的一个查询对象(并非最终结果)
  teachers = course.teachers.all()
~~~

#### 一对一

> 1.增加外键,引用另一张表主键，并且要实施唯一约束

~~~python
#在任意一个实体类中增加外键和唯一约束
class Wife(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    #增加外键,引用自 users表的主键id(一对一)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey('users.id'),
        unique=True,
        nullable=True
    )
#在另一个实体类中增加关联属性和反向引用关系属性
class Users(db.Model):
    id = db.Column(db.Integer,primary_key=True)
	wife = db.relationship(
        "Wife",
        backref="user",
        uselist = False
    )
~~~

#### 多对多

> 依靠第三章关联表的方式来实现

~~~python
class Student(db.Model):
  #实现与Teacher的关联关系(多对多,中间借助student_teacher关联表进行关联)
  teachers = db.relationship(
    "Teacher",
    secondary="student_teacher",
    lazy="dynamic",
    backref=db.backref(
      "students",lazy="dynamic"
    )
  )
#声明一个实体类表示关联Student和Teacher的第三张表
class StudentTeacher(db.Model):
  __tablename__ = "student_teacher"
  id = db.Column(db.Integer,primary_key=True)
  #外键:teacher_id,引用自teacher.id
  teacher_id = db.Column(
    db.Integer,
    db.ForeignKey('teacher.id')
  )
  #外键:student_id,引用自student.id
  student_id = db.Column(
    db.Integer,
    db.ForeignKey('student.id')
  )
    
#查询方式
# 查询1号老师所教授的学员
teacher=Teacher.query.filter_by(id=1).first()
print("老师姓名:%s" % teacher.tname)

students=teacher.students.all()#students对应Student的backref的值
for s in students:
    print("学员姓名:%s" % s.sname)
~~~

+ 在关联的两个类中的任意一个类中增加

**lazy:**

> 指定如何加载相关记录                                                                                     如：lazy 指的是如何加载 course中对应的Teacher们的信息）

| lazy取值  | 含义                                                         |
| --------- | ------------------------------------------------------------ |
| select    | 首次访问时加载关联数据                                       |
| immediate | 原对象加载后立马加载关联数据（使用连接）                     |
| subquery  | 效果同上（使用子查询）                                       |
| noload    | 永不加载                                                     |
| dynamic   | 不加载记录,但提供加载记录的查询<br/>couser.teachers.all()<br/> |



## 重定向

> 由服务器通知浏览器向一个新的地址发送请求

~~~python
from flask import redirect
return redirect('/重定向地址')
~~~

## Cookie

>cookie 是一种数据存储的手段
>将一段文本保存在客户端(浏览器)的一种手段,并可以长时间保存数据

> 1.各个浏览器中的cookie是不共享的
> 2.不同域名下的cookie也是不共享的

**Flask中使用cookies**

1. 响应对象

   1. 重定向就是响应对象

      resp = redirect() 

   2. 通过 make_response() 构建响应对象，将字符串/模板构建成响应对象

   ~~~python
   from flask import make_response
   resp = make_response(""或render_template())
   ~~~

   

2. 将数据保存进cookies

   + 响应对象.set_cookie(key,value,max_age)
     + key：保存的cookie的名称
     + value：保存的cookie的值
     + max_age：存活时长，取值为数字，以s为单位

3. 获取cookie值

   > 每次向服务器发送请求时,都会将cookies的数据封装到request中带到服务器

   + request.cookies 获取所有的cookies值
   + 通过具体的cookie名称,得到具体的cookie的值

4. 删除cookie的值

   + 响应对象.delete_cookie(“key”)

## Session

>session - 会话 , 当浏览器打开时,跟一个服务器交互的完整过程就是一次会话

>session 的目的:保存会话中所涉及到的一些信息

>session是保存在服务器上的,会为每个浏览器开辟一段空间,就是为了保存当前浏览器和服务器在会话过程中所涉及到的一些信息

**session 在 Flask 中的实现**

1. 配置SECRET_KEY
   + app.config['SECRET_KEY'] = "key值"
2. 使用session

~~~python
from flask import session
#向session中保存数据
session["key"] = vlaue
#从session中获取数据
value=session.get('key')
#从session中删除数据
del session['key']
~~~

1. cookie
   			1.保存在客户端[位置]
      			2.长久保存[时长]
      			3.因为是明文,可以修改,安全性较低[安全]
2. session
   			1.保存在服务器[位置]
      			2.临时存储[时长]
      			3.安全性教高[安全]

**登陆注册**

~~~html
<form action="/login" method="post">
  <p>
    用户名称: <input type="text" name="uname">
  </p>
  <p>
    用户密码: <input type="password" name="upwd">
  </p>
  <p>
    <input type="checkbox" name="isSaved">记住密码
  </p>
  <p>
    <input type="submit">
  </p>
</form>
~~~

~~~python
from flask import Flask, request, session, render_template, redirect, make_response

app = Flask(__name__)
app.config['SECRET_KEY'] = '写啥都行'


@app.route('/')
def hello_world():
  return 'Hello World!'

@app.route('/login',methods=['POST','GET'])
def login_views():
  if request.method == 'GET':
    #获取请求原地址,并保存在session中
    url=request.headers.get('Referer','/')
    print(url)
    session['url'] = url
    #判断session有没有uname
    if 'uname' in session:
      return redirect(url)
    else:
      #session中没有uname的值,继续判断cookie
      if 'uname' in request.cookies:
        #cookies中有uname,从cookie中获取uname的值
        uname = request.cookies['uname']
        #判断uname的有效性(值是否为admin,可以判断DB)
        if uname == 'admin':
          #用户名正确,将uname保存进session
          session['uname'] = uname
          return redirect(url)
        else:
          #用户名不正确
          resp = make_response(
            render_template('login.html')
          )
          #通过resp删除cookies中uname的值
          resp.delete_cookie('uname')
          return resp
      else:
        #没有uname
        return render_template('login.html')

  else:
    #接收登录信息
    uname=request.form['uname']
    upwd = request.form['upwd']
    #验证登录是否成功(值是否为admin,可以改成DB版)
    if uname=='admin' and upwd=='admin':
      #登录成功后的处理
      #1.将登录名称保存进session
      session['uname'] = uname
      #2.从session中将url获取出来,构建响应对象
      url = session['url']
      resp = redirect(url)
      #3.判断是否要记住密码,记住密码则将uname保存进cookie
      if 'isSaved' in request.form:
        resp.set_cookie('uname',uname,60*60*24*365*10)
      return resp
    else:
      #登录失败的处理
      return render_template('login.html')


if __name__ == '__main__':
  app.run(debug=True)

~~~