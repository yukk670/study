# 显示操作数据库的语句

> 需要在配置中加上这个

~~~python
LOGGING = {
    'version':1,
    'disable_existing_loggers':False,
    'handlers':{
        'console':{
            'level':'DEBUG',
            'class':'logging.StreamHandler',
        },
    },
    'loggers':{
        'django.db.backends':{
            'handlers':['console'],
            'propagate':True,
            'level':'DEBUG'
        },
    }
}
~~~

# 数据迁移

> 首先要保证,目前的migration文件和数据库是同步的，通过执行

```
python3 manage.py makemigrations 
如果看到 这样的提示: No changes detected，则可以继续接下来的步骤
```

> 查看目前存在的App

```
python3 manage.py showmigrations
结果，可以看到当前项目，所有的app及对应的已经生效的migration文件如下：
```

```
admin
[X] 0001_initial
[X] 0002_logentry_remove_auto_add
[X] 0003_logentry_add_action_flag_choices
auth
[X] 0001_initial
[X] 0002_alter_permission_name_max_length
[X] 0003_alter_user_email_max_length
[X] 0004_alter_user_username_opts
[X] 0005_alter_user_last_login_null
[X] 0006_require_contenttypes_0002
[X] 0007_alter_validators_add_error_messages
[X] 0008_alter_user_username_max_length
[X] 0009_alter_user_last_name_max_length
contenttypes
[X] 0001_initial
[X] 0002_remove_content_type_name
nginxAuto
[X] 0001_initial
[X] 0002_servername_key_test
[X] 0003_remove_servername_key_test
sessions
[X] 0001_initial
```

1.通过执行

```
python manage.py migrate --fake nginxAuto zero
```

> 这里的 nginxAuto就是你要重置的app之后再执行 python manage.py showmigrations，你会发现 文件前的 [x] 变成了[ ]

2.删除pay 这个 app下的migrations模块中 除 **init.py** 之外的所有文件。

```
python manage.py makemigrations
```

3.同步migration文件与数据库进度对应

```
python manage.py migrate --fake-initial
```

> --fake-inital 会在数据库中的 migrations表中记录当前这个app 执行到 0001_initial.py ，但是它不会真的执行该文件中的 代码。

> 这样就做到了，既不对现有的数据库改动，而又可以重置 migraion 文件，

# timezone

> 当settings中设定USE_TZ = True，TIME_ZONE和本地不同,将会对日期进行强行的时区操作导致时间不对，可用以下方式处理成正确时区

~~~python
from datetime import datetime
import pytz
utc_zone = pytz.timezone("Asia/Shanghai")
date = datetime.strptime('2019-09-19 17:47:41', "%Y-%m-%d %H:%M:%S").replace(tzinfo=utc_zone)
~~~

```python
# 获取字段的组合信息
self.annotate(
            density=F('population') / F('land_area_km')
        )
# 合并字段选项（要为字符串类型）并去重
BigfishUser.objects.filter(realname__contains="姚申").aggregate(user_ids=StringAgg("realname",",",distinct=True))
```

### 子查询

> SELECT * FROM users_userklassrelationship WHERE (user_id IN (SELECT U0."id" FROM "users_bigfishuser" U0 LIMIT 10) AND is_default= true);

~~~python
user_ids = BigfishUser.objects.values('id')[:10]
# 方式1
UserKlassRelationship.objects.filter(user_id__in=user_ids, is_default=True)
# 方式2
list(UserKlassRelationship.objects.filter(user_id__in=Subquery(user_ids), is_default=True))
~~~

