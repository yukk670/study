# 常用指令

> :one:.启动服务：
>
> ​			python3 manage.py runserver 0.0.0.0:端口号
>
> :two:.将每个应用下的models.py文件生成一个中间文件,并保存在migrations文件夹中：
>
> ​			python3 manage.py makemigrations
>
> :three:.将每个应用下的migrations目录中的中间文件同步回数据库
>
> ​            python3 manage.py migrate
>
> :four:.使用指令创建后台管理员:
> 			python3 manage.py createsuperuser
>
> :five:.查看对应polls模块的0001_initial执行的sql语句
>
> ​           python manage.py sqlmigrate polls 0001
>
> :six:.检查项目中的任何问题，而无需进行迁移或触摸数据库
>
> ​		python manage.py check
>
> :seven:.版本回滚

# Django

​	:o:2005年发布,采用Python语言编写的开源框架
​	:o:早期的时候Django主做新闻和内容管理的
​	:o:Django中自带强大的后台管理功能

+ 官网:link: http://www.djangoproject.com

+ 中文文档(第三方)::link: http://djangobook.py3k.cn/2.0/ 

> Django的框架模式
> 		M -- 模型层
> 		T -- 模板层
> 		V -- 视图层

> Django的安装

~~~python
#查看已安装的版本
import django
django.VERSION

#安装django的最新版本
sudo pip3 install django
#安装django的指定版本
sudo pip3 install django==1.11.8
#离线安装
sudo pip3 install 离线安装包路径
~~~

## 使用方法

### 创建项目

> ​	django-admin startproject 项目名称

### 项目目录结构

#### manage.py

> 包含项目管理的子命令

| runserver | 启动服务   |
| --------- | ---------- |
| startapp  | 创建应用   |
| migrate   | 数据库迁移 |

#### 主文件夹（与项目名称一致的文件夹）

> __init__.py--------项目初始化文件,服务启动时自动运行

> wsgi.py-------WEB服务网关接口的配置文件(部署项目时使用)

> urls.py                                                                                                                  项目的基础路由配置文件(所有的动态路径必须先走该文件进行匹配)

> settings.py----项目的配置文件，启动服务时自动调用

**settings内置配置说明：**

| 配置名         | 说明                                                         |
| -------------- | ------------------------------------------------------------ |
| BASE_DIR       | 表示当前项目的绝对路径                                       |
| DEBUG          | 是否启用调试模式（False）                                    |
| ALLOWED_HOSTS  | 设置允许访问到本项目的地址列表<br> 1.为空：只有localhost和127.0.0.1可访问本项目<br> 2.[‘*’]：任何表示本机的地址都能访问到当前项目<br> 在局域网访问的话，启动方式:<br> python3 manage.py runserver 0.0.0.0:端口号 |
| INSTALLED_APPS | 指定已安装的应用                                             |
| MIDDLEWARE     | 注册中间件                                                   |
| TEMPLATES      | 指定模板的配置信息                                           |
| DATABASES      | 指定数据库的配置信息                                         |
| LANGUAGE_CODE  | 指定语言配置：如中文（zh-Hans）                              |
| TIME_ZONE      | 时区：如中国时区（Asia/Shanghai）                            |

## URL

#### urls.py

+ 默认存在于主文件夹内,主路由配置文件
+ 包含最基本的路由-视图的映射关系
+ 该文件会包含 urlpatterns 的列表用于表示路由-视图映射,通过 url() 表示具体映射

#### url()

~~~python
from django.conf.urls import url
url(regex,views,kwargs=None,name=None)
#regex:字符串类型，匹配的请求路径，允许是正则表达式
#views：指定路径所对应的视图处理函数的名称
#kwargs：向视图中传递的参数
#name：为地址起别名，反向解析时使用
~~~

> 通过别名实现地址的反向解析

+ 在模板中：
  + {% url ‘别名’ %}
  + {% url ‘别名’  '参数值1' '参数值2' %}

~~~python
url(r'^08-birthday/(\d{4})/(\d{2})/$', views.birthday, name="birth")
<a href="{% url 'birth' '1990' '12' %}">访问 birth</a>
~~~

#### 带参数的url

> 使用正则表达式的子组进行传参 - ()
> 一个子组表示一个参数,多个参数需要使用多个子组,并且使用个 / 隔开

~~~python
#http://localhost:8000/show-02/xxxx/
#xxxx:表示任意的四位数字
urlpatterns = [
    url(r'^show-02/(\d{4})/$',views.show_02),]
#############views.py##################
def show_02(request,year):
	year:表示的就是url中第一个子组对应的参数
return HttpResponse("xxx")
~~~

**示例：**

> http://locahohst:8000                                                                                            http://localhost:8000/date/2015/12/11

~~~python
#view.py
from django.http import HttpResponse

def index(request):
    return HttpResponse("这是我的主页")

def date(request,*data):
    return HttpResponse("生日为:%s年%s月%s日"%data)

#urls.py
from django.conf.urls import url
from django.contrib import admin
from . import views

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^$', views.index),
    url(r'^date/(\d{4})/(\d{1,2})/(\d{1,2})$', views.date),
]
~~~

## 应用 - app

>Django中,主文件夹是不处理用户具体请求的.主文件夹的作用是做项目的初始化以及请求的分发(分布式请求处理).具体的请求是由应用来进行处理的

### 创建应用

1. 创建应用

+ python3 manage.py startapp 应用名称

2. 在主文件夹的settings.py配置应用

+ INSTALLED_APPS=[...,自定义应用名称...]

### 应用的结构

| 文件名           | 用处                       |
| ---------------- | -------------------------- |
| migrations文件夹 | 保存数据迁移的中间文件     |
| __ init __.py    | 应用的初始化文件           |
| admin.py         | 应用的后台管理配置文件     |
| apps.py          | 应用的属性配置文件         |
| model.py         | 与数据库相关的模型映射文件 |
| tests.py         | 应用的单元测试文件         |
| views.py         | 定义视图处理函数的文件     |

### 分布式路由系统

> 在应用中，创建urls.py，结构参考主路由配置
>
> http://localhost:8000/login/

~~~python
#index应用内的urls.py
from . import views
urlpatterns = [
    url(r'^$',views.index ),
    url(r'^login/$',views.login ),
    url(r'^register/$',views.register ),

]
#主文件夹内的urls.py
from django.conf.urls import url,include
from django.contrib import admin

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^sport/',include("sport.urls")),
    url(r'^news/',include("news.urls")),
    #如果r'^index',只有http://localhost:8000/index/login/可访问
    url(r'',include("index.urls")),
]
~~~

## Templates

> Django中的模板是由Django自己提供的,而非Jinja2
> 所以Django中的模板语法与Flask中的模板语法会稍有不同

### 配置

> 在settings.py中有一个TEMPLATES变量

| backend  | 指定模板的引擎           |
| -------- | ------------------------ |
| DIRS     | 指定保存模板的所有目录   |
| APP_DIRS | 是否要在应用中搜索模板本 |
| OPTIONS  | 有关模板的所有选项       |

~~~python
TEMPLATES = [
    {...,'DIRS': [os.path.join(BASE_DIR, 'templates')],...}]
~~~

### 加载方式

1. 通过loader获取模板，通过HttpResponse进行响应

~~~python
from django.template import loader
#1.通过loader加载模板
t = loader.get_template("模板名称")
#2.将t转换成字符串
html = t.render([dic])#dic字典形式传递给模板的变量
#3.响应
return HttpResponse(html)
~~~

2. 使用 render() 直接加载并响应模板	

~~~python
from django.shortcuts import render
return render(request,'模板的名称'[,dic])#dic字典形式传递给模板的变量
~~~

### 模板语法

> 必须将变量封装到字典[dic]中才允许传递到模板上：如上

> 在模板中使用变量

+ {{变量名}}

### 标签

**内置变量-forloop**

> flask中的loop被替换为forloop

1. 得到当前循环遍历的次数,从1开始：forloop.counter
2. 得到当前循环遍历的下标,从0开始：forloop.counter0
3. forloop.first
4. forloop.last

### 过滤器

> 在变量输出前对变量的值进行筛选

+ {{变量|过滤器:参数值}}

> 有用的过滤器

+ :one: ​default	:two:default_if_none	:three:floatformat	:four:truncatechars
  :five:truncatewords

### 静态文件

> :one:	主文件夹中的settings.py

~~~python
#1.配置静态文件的访问路径
STATIC_URL = '/static/'
#2.配置静态文件的存储路径---------静态文件在服务器端的保存位置(设置后前端才能读取静态文件内的内容)
STATICFILES_DIRS=(os.path.join(BASE_DIR,'static'),)
~~~

> :two:	访问静态文件

~~~html
<!--1.使用静态文件的访问路径进行访问-->
<img src="/static/images/a.jpg">
<img src="http://127.0.0.1:8000/static/images/a.jpg">
<!--2.通过 {% static %}标签访问静态文件-->
<1.加载 static/>
{% load static %}
<img src="{% static 'images/a.jpg' %}"> 
~~~

### 模板继承

> 模板继承时,服务器端的动态内容无法继承，其余与flask相似

## 模型

### 配置数据库

> 创建数据库

~~~sqlite
create database webdb default charset utf8 collate utf8_general_ci;
~~~

> 主目录的settings.py

~~~python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),}}
~~~

> 主目录的__ init __.py【如果数据库要导包】

~~~python
import pymysql
pymysql.install_as_MYSQLdb()
~~~

**databases属性一览**

| name     | 指定要连接的数据库名称“test”                     |
| -------- | ------------------------------------------------ |
| ENGINE   | 指定数据库引擎，如：<br>django.db.backends.mysql |
| USER     | 指定登陆到数据库的用户名:“root”                  |
| PASSWORD | 指定登陆到数据库的密码：”123456”                 |
| HOST     | 指定要连接的主机：”localhost”                    |
| PORT     | 指定要连接的主机上的端口号:“3306”                |

### 数据库迁移

> 将每个应用下的models.py文件生成一个中间文件,并保存在migrations文件夹中

+ python3 manage.py makemigrations

>将每个应用下的migrations目录中的中间文件同步回数据库

+ python3 manage.py migrate

### 编写Model

> 相关指令

+ 通过数据库自动导出models
  + python3 manage.py inspectdb > xxx.py

---

~~~python
class CLASSNAME(models.Model):
	NAME=models.FIELD_TYPE(FIELD_OPTIONS)
~~~

> CLASSNAME

+ 实体类名,表名组成一部分
+ 默认表名组成规范:应用名称_classname

> NAME

+ 属性名,映射回数据库就是字段名

> FIELD_TYPE

+ 字段类型:映射到表中的字段的类型

| 字段类型                                                     | python类型           | 数据库类型      |
| ------------------------------------------------------------ | -------------------- | --------------- |
| BooleanField()                                               | True或False          | tinyint【1或0】 |
| CharField(max_length=30)                                     | String               | varchar         |
| DateField()                                                  | String【日期】       | date            |
| DateTimeField()                                              | String【日期和时间】 | datetime        |
| DecimalField()                                               | Float                | decimal         |
| FloatField()                                                 | Float                | float           |
| EmailField()                                                 | String               | varchar         |
| IntegerField()                                               | Int                  | int             |
| URLField()                                                   | String               | varchar         |
| ImageField(upload_to="static/images")<br> :anger:upload_to:指定图片的上传路径 | String               | varchar         |
| TextField()                                                  | String               | text            |

> FIELD_OPTIONS

+ 字段选项,指定创建的列的额外的信息
+ 允许出现多个字段选项,多个选项之间使用,隔开

| 字段选项    | 说明                                            |
| ----------- | ----------------------------------------------- |
| primary_key | 如果设置为True,表示该列为主键                   |
| null        | 如果设置为True,表示该列值允许为空默认为False    |
| default     | 设置所在列的默认值                              |
| db_index    | 如果设置为True，表示为该列添加索引              |
| unique      | 如果设置为True,表示该列的值唯一                 |
| db_column   | 指定列的名称,如果不指定的话则采用属性名作为列名 |

~~~python
#创建一个属性,表示用户名称,长度30个字符,必须是唯一的,不能为空,添加索引
name=models.CharField(max_length=30,unique=True,null=False,db_index=True)
~~~

### 数据库操作

> QuerySet对象属性
>
> query：对应的sql查询语句

#### 增加数据

+ Entry.objects.create(属性=值，属性=值)
  + 返回值：创建好的实体对象

+ 创建Entry对象，并调用save()进行保存
  + obj = Entry(属性=值，属性=值)

  + obj.属性=值

  + author.save()

  + 无返回值，保存成功后，obj会被重新复制

+ 使用字典创建对象，并调用save()进行保存

~~~python
dic ={'属性1':"值"，"属性2":"值2"}
obj = Entry(**dic)
obj.save()
~~~

~~~python
from .models import Book
from django.http.response import HttpResponse

def add(request):
    #方式1
    Book.objects.create(
        title='钢铁是咋练成的',
        publicate_date='1988-10-12'
    )
	#方式2
    book1 = Book()
    book1.title = "西游记"
    book1.publicate_date = '1986-08-11'
    book1.save()
	#方式3
    dic = {
        'title': '红楼梦',
        'publicate_date': '1992-03-12'
    }
    book2 = Book(**dic)
    book2.save()

    return HttpResponse("Create Success!!!")
~~~

#### 查询数据

> 通过Entry.objects属性，调用查询接口

~~~python
QuerySet = Entry.objects.all()
~~~

+ 查询返回指定列的字典（{列名1：值1,列名2：值2}）

  + Entry.objects.values(‘列1’,‘列2’)
  + 返回值：QuerySet会将查询出来的数据封装到元组中，再封装到列表中

+ 查询返回指定列的元组（值1，值2）

  + Entry.objects.values_list(‘列1’,‘列2’)

+ 排序查询

  + Entry.objects.order_by(‘-列1’,‘列2’)
  + 默认是按照升序排序，降序排序则需要在列前增加‘-’表示

~~~python
querySet = Author.objects.all()
set1 = Author.objects.values("name","age")
set2 = Author.objects.values_list("name","age")
set3 = Author.objects.order_by("-age","name")
for author in set3:
    print(author.age)
return HttpResponse("Query Success!!!")
~~~

> 条件查询：Entry.objects.filter(条件)

~~~python
#查询Author实体中id为1并且isActive为True的
authors=Author.objects.filter(id=1,isActive=True)
~~~

+ 非等值条件的构建,需要使用查询谓词(Field Lookup)

| 查询谓词     | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| __exact      | 等值匹配                                                     |
| __contains   | **包含指定值**<br> Author.objects.filter(name__contains='w')<br> select * from author where name like '%w%' |
| __gt         | ＞                                                           |
| __gte        | ＞=                                                          |
| __ lt,__lte  | ＜,＜=                                                       |
| __startswith | 查找数据是否从指定字符串开始                                 |
| __in         | **查找数据是否在指定范围内**<br> Publisher.objects.filter(country__in=['中国','日本','韩国'])<br> select * from publisher where country in ('中国','日本','韩国') |
| __range      | 查找数据是否在指定的区间范围内<br> Author.objects.filter(age__range=(35,50)) |

~~~python
# 查询Author表中age大于等于85的信息
Author.objects.filter(age__gte=85)
# 查询Author表中姓巴的人的信息
Author.objects.filter(name__startswith="巴")
#查询Author表中Email中包含in的人的信息
Author.objects.filter(email__contains="in")
# 查询Author表中Age大于"巴金"的age的信息
Author.objects.filter(age__gt=(
    Author.objects.filter(name="巴金").values("age")))
~~~

> 不等的条件筛选

+ Entry.objects.exclude(条件)

> 查询只返回一条数据

+ Entry.objects.get(条件)
  + 查询结果不是一条数据的话，会抛异常

> 聚合查询

~~~python
from django.db.models import Sum,Avg,Count,Max,Min
~~~

+ 不带分组
  
+ Entry.objects.aggregate(变量名=聚合函数(‘列’))
  
+ 带分组

  + Entry.objects.values(“分组列名1”,“分组列名2”)

    ​	.annotate(变量名=聚合函数(‘列’))

    ​	.values(“查询列名1”,查询列名2)

~~~python
# 查询Author表中所有人的平均年龄和总年龄
result = Author.objects.aggregate(avg = Avg('age'),sum = Sum('age'))
# 查询Author表中年纪>=80的人的平均年龄
result = Author.objects.filter(age__gte=80).aggregate(avg=Avg('age'))
# 按 isActive 进行分组,求每组的人数
result = Author.objects.values("isActive").annotate(count=Count('*')).values("isActive","count")
#查询1986(不包含)年之后所出版的图书的数量
Book.objects.filter(publicate_date__year__gt=1986).aggregate(count=Count('*'))
~~~

#### 修改数据

> 修改单个实体

1. 通过get()得到要修改的实体对象
2. 通过对象.属性的方式修改数据
3. 通过对象.save()保存数据

> 修改QuerySet

+ 直接调用QuerySet的update（属性=值）实现批量修改

~~~python
au=Author.objects.get(name='老舍')
au.email='laoshe@sina.com'
au.save()
# 将所有isActive为0的修改成1
Author.objects.filter(isActive=False).update(isActive=True)
~~~

#### 删除数据

> 删除单个对象

+ Author.objects.get(id=1).delete()

> 删除查询结果集

+ Author.objects.filter(inActive=True).delete()

~~~python
au = Author.objects.get(id=id)
au.isActive = False
au.save()

#将所有isActive为1的删除
Author.objects.filter(isActive=1).delete()
~~~

#### F、Q查询

> **F查询**：在执行过程中获取某列的值

~~~python
from django.db.models import F
Author.objects.all().update(age=F('age') + 10)
~~~

> **Q查询**：在条件中用来实现or的操作

~~~python
from django.db.models import Q
Author.objects.filter( Q(id=1) | Q(age__gte=80) )
~~~

#### 原生操作

> 查询：Entry.objects.raw(“SQL语句”)

> 增删改

~~~python
from django.db import connection

def doSQL():
    #更新index_author表中所有的数据的isActive=1
    with connection.cursor() as cursor:
        cursor.execute('update index_author set isActive=1')
    return HttpResponse('xxx')
~~~

### 后台管理Models

1. 后台的配置

   > http://localhost:8000/admin
   >
   > 使用指令创建后台管理员:
   > 			python3 manage.py createsuperuser

2. 基本的数据管理

   > 在应用中的admin.py中注册要管理的实体类

   ~~~python
   from django.contrib import admin
   from .models import *
   
   admin.site.register(Entry)
   admin.site.register(Entry)
   ~~~

   > 定义Models的展现形式

   1. 通过实体类的__ str __()定义展现名称
   2. 通过verbose_name字段选项，修改名称
   3. 通过Meta内部类，修改展现形式

   ~~~python
   class Author(models.Model):
       def __str__(self):
           return self.name
   
       class Meta:
           verbose_name = "作者一览"
           verbose_name_plural = "作者列表"
   
   class Book(models.Model):
       def __str__(self):
           return self.title
   
       class Meta:
           verbose_name = "书籍一览"
           verbose_name_plural = "书籍列表"
   ~~~

3. 高级的数据管理

   > 在admin.py中创建高级管理类并注册

~~~python
from django.contrib import admin
from .models import *

class AuthorAdmin(admin.ModelAdmin):
    #定义在列表页上显示的字段
    list_display = ("name","age","email")
    # 定义在列表页中能链接到详情页的字段们
    list_display_links = ("name","email")
    # 定义在列表页中就允许编辑的字段们
    # 注意: 取值不能出现在list_display_links中的
    list_editable = ("age",)
    # list_filter : 列表页的右侧增加一个过滤器实现筛选
    list_filter = ('isActive',)

    # search_fields : 添加允许被搜索的字段们
    search_fields = ('name', 'email')

    # fields : 定义在详情页中要显示的字段及其顺序
    # fields = ('isActive','name','email')

    # fieldsets : 定义在详情页中的字段分组
    # 注意:fieldsets 属性 和 fields 不能共存
    fieldsets = (
        # 分组1
        ('基本选项', {
            'fields': ('name', 'email'),
        }),
        # 分组2
        ('可选选项', {
            'fields': ('age', 'isActive'),
            'classes': ('collapse',)
        })
    )

class BookAdmin(admin.ModelAdmin):
  # date_hierarchy : 在列表页中增加一个时间分层选择器
  date_hierarchy = "publicate_date"

admin.site.register(Author,AuthorAdmin)
admin.site.register(Book,BookAdmin)
~~~

## 关系映射

### 一对一映射

>在关联的两个类中的任何一个类中：
>		属性 = models.OneToOneField(Entry)

~~~python
class Wife(models.Model):
  name = models.CharField(max_length=30)
  age = models.IntegerField()
  #一对一关系映射:引用自Author
  author = models.OneToOneField(Author)
~~~

**数据查询**

~~~python
#wife查Author
wife=Wife.objects.get(name='舒夫人')
author = wife.author
#Author查wife
author = Author.objects.get(name='巴金')
wife = author.wife
~~~

### 一对多映射

>在"多"实体中,对"一"的实体进行引用
>			属性 = models.ForeignKey(Entry)

~~~python
#一个出版社允许出版多本图书；一本图书只能属于一个出版社
class Book(models.Model):
  #增加Publisher的一对多的引用
  publisher = models.ForeignKey(Publisher,null=True)
~~~

**数据查询**

~~~python
# 查询任意书对应的一个出版社的信息
book = Book.objects.get(title='西游记')
pub = book.publisher
# 查询 清华大学出版社 对应的所有的 书籍们
pub = Publisher.objects.get(name='清华大学出版社')
books = pub.book_set.all()
~~~

### 多对多映射

>在关联的两个类中的任意一个类中,增加：
>		属性 = models.ManyToManyField(Entry)

~~~python
#一个作者可以出版多本图书，一本图书可以被多名作者同时编写
class Book(models.Model):
    # 增加对Author的多对多的引用
    authors = models.ManyToManyField(Author)
~~~

**数据查询**

~~~python
# 查询 红楼梦 对应的所有的作者们
book = Book.objects.get(title='红楼梦')
authors = book.authors.all()

# 查询 巴金 对应的所有的书籍
author = Author.objects.get(name='巴金')
books = author.book_set.all()
~~~

## Request

> HttpRequest , 在Django中就是请求对象,默认会被封装到视图处理函数的参数中 - request

**request中的成员(属性)**

| request.成员  | 说明                                                         |
| ------------- | ------------------------------------------------------------ |
| scheme        | 请求协议                                                     |
| body          | 请求主体（POST，PUT）                                        |
| path          | 请求的具体资源路径                                           |
| get_full_path | 请求的完整路径                                               |
| get_host()    | 请求的主机                                                   |
| method        | 请求方式                                                     |
| GET           | get请求方式中封装的数据                                      |
| POST          | post请求方式中封装的数据                                     |
| COOKIES       | 请求中的cookies的相关数据                                    |
| META          | 请求中的元数据（消息头）<br> request.META['HTTP_REFERER'] : 请求源地址 |

### GET

>request.GET['参数名']
>request.GET.get('参数名','默认值')
>request.GET.getlist('参数名')

### POST

>				request.POST['参数名']
>				request.POST.get('参数名','默认值')
>				request.POST.getlist('参数名')

#### CSRF验证

| C      | s    | r       | f      |
| ------ | ---- | ------- | ------ |
| Cross- | Site | Request | Forgey |
| 跨     | 站点 | 请求    | 伪装   |

**解决方案：**

1. 取消 csrf 验证(不推荐)

   + 删除 settings.py 中 MIDDLEWARE 中的 CsrfViewsMiddleWare 的中间件

2. 开放验证

   + 在视图处理函数增加 : @csrf_protect

     ~~~python
     @csrf_protect
     def post_views(request):	pass
     ~~~

3. 通过验证
   
   + 需要在表单中增加一个标签 ： {% csrf_token %}

~~~html
<form action="/login/" method="post">
{% csrf_token %}
~~~

## forms

>通过 forms 模块,允许将表单与class相结合,允许通过 class 生成表单

1. 在应用中创建 forms.py 

~~~python
from django import forms
class ClassName(form.Form):	
	#属性对应到表单中是一个控件
~~~

2. 属性 = forms.类型（参数）

| 类型                | 对应html标签        |
| ------------------- | ------------------- |
| forms.CharField()   | <input type="text"> |
| forms.ChoiceField() | <select>            |
| forms.DateField()   | <input type="date"> |

| 参数    | 说明                                            |
| ------- | ----------------------------------------------- |
| label   | 控件前的文本                                    |
| widget  | 指定小部件                                      |
| initial | 空间的初始值<br> （主要针对文本框类型）value=“” |

3. 在模板中解析form对象

   > :one:.需要自定义 <form>
   > :two:.表单中的按钮需要自定义

   1. 在视图中创建form对象并发送到模板中解析.

   ~~~python
   form = RemarkForm()
   return render(request,'xx.html',locals())
   ~~~

   > 手动解析

   ~~~python
   {% for field in form %}
   	field : 表示的是form对象中的每个属性(控件)
       {{field.label}} : 表示的是label参数值
       {{field}} : 表示的就是控件
   {% endfor %}
   ~~~

   > 自动解析

| 模板标签          | 说明                                                         |
| ----------------- | ------------------------------------------------------------ |
| {{form.as_p}}     | 将 form 中的每个属性(控件/文本)都使用p标记包裹起来再显示     |
| {{form.as_ul}}    | 将 form 中的每个属性(控件/文本)都使用li标记包裹起来再显示<br>:warning:必须手动提供ol 或 ul 标签 |
| {{form.as_table}} | 将 form 中的每个属性(控件/文本)都使用tr标记包裹起来再显示<br/>:warning:必须手动提供table标签 |

4. 通过 forms 模块获取表单数据

~~~python
#1.通过 forms.Form 子类的构造器来接收 post 数据
form = RemarkForm(request.POST)
#2.必须是 form 通过验证后(返回为True),才能取值
if form.is_valid():
	#3.通过 form.cleaned_data 属性接收数据（为字典类型）
	data = form.cleaned_data
~~~

### 将Models实体类和Forms模块结合

~~~python
from django.forms import ModelForm
class AuthorForm(forms.ModelForm):
  class Meta:
    # 1. 指定关联的Model类
    model = Author
    # 2. 指定从Model类中取那些属性生成控件
    fields = "__all__"
    # 3. 指定Model类中的属性对应的label值
    labels = {
      'name': '姓名',
      'age': '年龄',
      'email': '邮箱',
      'isActive': '激活'
    }
~~~

### 内置小部件 - widget

> 小部件：表示的是生成到网页上的控件以及一些其他的html属性

~~~python
message=forms.CharField(widget=forms.Textarea)
upwd=forms.CharField(widget=forms.PasswordInput)
~~~

| 常用小部件类型         | 相应HTML标签          |
| ---------------------- | --------------------- |
| TextInput              | type=“text”           |
| PasswordInput          | type='password'       |
| NumberInput            | type="number"         |
| EmailInput             | type="email"          |
| URLInput               | type="url"            |
| HiddenInput            | type="hidden"         |
| CheckboxInput          | type="checkbox"       |
| CheckboxSelectMultiple | type="checkbox"       |
| RadioSelect            | type="radio"          |
| Textarea               | <textarea></textarea> |
| Select                 | <select></select>     |
| SelectMultiple         | <select multiple>     |

#### 小部件的使用

~~~html
  <style>
    .form-input{
      border:none;
      border-bottom:1px solid #333;
      font-size:14px;
      font-weight:bold;
    }
  </style>
</head>
<body>
<form action="">
  {{ form.as_p }}
</form>
</body>
~~~

> forms.Form

~~~python
class WidgetForm(forms.Form):
  uname = forms.CharField(
    label='用户名称',
    widget = forms.TextInput(
      attrs={
        'placeholder': '请输入用户名',
        'class': 'form-input',
      }
    )
  )
  upwd = forms.CharField(
    label='用户密码',
    widget=forms.PasswordInput(
      attrs={
        'placeholder': '请输入密码',
        'class': 'form-input',
      }
    )
  )
~~~

> forms.ModelForm

~~~python
class WidgetModelForm(forms.ModelForm):
  class Meta:
    model = Author
    fields = ["name","email"]
    labels = {
      'name' : "姓名",
      'email': "邮箱",
    }
    widgets = {
      'name': forms.TextInput(
        attrs={
          'placeholder': '请输入姓名',
          'class':'form-input',
        }
      ),
      'email': forms.EmailInput(
        attrs = {
          'placeholder': '请输入您的邮箱',
          'class': 'form-input'
        }
      )
    }
~~~

## Cookies

> 保存在客户端浏览器上的一段存储空间

1. 使用 响应对象 将cookie保存进客户端
   + resp = HttpResponse()
   + resp = render(request,'xxx.html',locals())
   + resp = redirect('/')

2. 保存cookie
   + resp.set_cookie(key,value,expires)
     + key：cookie的名字
     + value：cookie的值
     + expires：保存时长，以s为单位的数字
3. 获取cookie
   + request.COOKIES
     + 封装了当前站点下所有的cookie - 字典

~~~python
uphone = request.COOKIES.get('uphone','None')
return HttpResponse("cookie的值为:"+uphone)
~~~

## Session

> 在服务器上开辟一段空间用于保留浏览器和服务器交互时的重要数据

1. 保存 session 的值到服务器

   + request.session['KEY'] = VALUE

2. 获取session的值

   + VALUE = request.session['KEY']

3. 删除session的值

   + del request.session['KEY']

4. 在 settings.py 中有关session的设置

   > 指定sessionid在cookies中的保存时长

   + SESSION_COOKIE_AGE = 60*30

   > 设置只要浏览器关闭时,session就失效

   + SESSION_EXPIRE_AT_BROWSER_CLOSE = True

# 常见错误

~~~python
IntegrityError at /
(1062, "Duplicate entry '老舍' for key 'name'")
~~~

> 设置了unique，然后插入相同值会产生该错误

---

~~~python
AttributeError: 'str' object has no attribute 'get'
~~~

> 页面中如果显示这个错误，一般view.py的返回值为“”要换成HttpResponse()等方法处理数据

---

~~~python
django.db.utils.InternalError: (1054, "Unknown column 'publicate_date' in 'field list'")
~~~

> 检查赋值的属性是否和model内的属性一致！如果不一致，修改了model的属性需要重新输入以下指令
>
> python3 manage.py makemigrations
>
> python3 manage.py migrate



#for Windows
python -m pip install -U pip
#for Linux
pip install -U pip