# AJAX

> Asynchronous（异步的） Javascript And Xml

> xml : eXtensible Markup Language
> JSON : 通过JSON来取代XML

**同步访问**

> 当客户端向服务器发送请求时,服务器在处理的过程中,浏览器只能等待,效率较低

**异步访问**

> 当客户端向服务器发送请求时,服务器在处理的过程中,客户端浏览器可以做其他的操作,不需要一直等待

**Ajax优点**

1. 异步 访问
2. 局部 刷新

**使用场合**

1. 搜索建议
2. 表单验证
3. 前后端完全分离

## 异步对象

> 异步对象XMLHttpRequest，简称为 xhr，代替浏览器向服务器发送异步的请求并接收响应

> xhr 是由JS来提供的

**创建异步对象**

>主流的异步对象是XMLHttpRequest 类型的(W3C)
>主流的浏览器全部支持XMLHttpRequest
>			IE7+,Chrome,.Firefox,Safari,Opera
>但在IE低版本中就不支持XMLHttpRequest,需要使用ActiveXObject 来创建异步对象

~~~javascript
function createXhr(){
    //判断浏览器对xhr的支持性
    if(window.XMLHttpRequest){
        return new XMLHttpRequest();
    }else{
        return new ActiveXObject("Microsoft.XMLHTTP");
}}
~~~

**异步对象的成员**

> 创建请求

+ xhr.open(method,url,async)
  + method：请求方式,取值 'get' 或 'post'
  + url：请求地址,取值为字符串
  + async：是否采用异步方式发送请求

~~~javascript
xhr.open('get','/server01',true);
~~~

> xhr的请求状态,通过不同的请求状态来表示xhr与服务器的交互情况.由0-4共5个值来表示5个不同的状态

| 0    | 请求尚未初始化          |
| ---- | ----------------------- |
| 1    | xhr已经与服务器建立连接 |
| 2    | 服务器端已经接受请求    |
| 3    | 服务器正在处理请求      |
| 4    | 响应已经完成            |

> 服务器端的响应状态码

| 200  | 服务器正确给出所有响应 |
| ---- | ---------------------- |
| 404  | 请求资源不存在         |
| 500  | 服务器内部错误         |

> 响应文本：responseText

> 通知xhr向服务器端发送请求

+ get 请求:xhr.send(null)
+ post 请求:xhr.send("请求数据")

> 每当xhr的readyState发生改变时都要触发的操作 - 回调函数

~~~javascript
//当readyState的值为4并且status的值为200的时候,就可以接收响应数据(responseText)
xhr.onreadystatechange=function(){
	if(xhr.readyState==4&&xhr.status==200){
		console.log(xhr.responseText);}}
~~~

## 请求步骤

**Ajax-GET**

~~~javascript
//1.创建 xhr 对象
var xhr = createXhr();
//2.创建请求 - open
xhr.open('get','/02-server?uname=1',true);
//3.设置回调函数 - onreadystatechange
xhr.onreadystatechange = function(){
    //将响应数据显示在#show元素中
    if(xhr.readyState==4&&xhr.status==200){
        document.getElementById("show").innerHTML 
            = xhr.responseText;
    }}
//4.发送请求 - send
xhr.send(null);
~~~

**Ajax-POST**

> ajax post 请求中,默认会将 Content-Type更改为 text/plain , 导致数据无法正常提交

+ xhr.setRequestHeader("Content-Type","application/x-www-form-urlencoded");

~~~javascript
//1.创建xhr
var xhr = createXhr();
//2.创建请求 - open,post请求方式
xhr.open('post','/04-server',true);
//3.设置回调函数
xhr.onreadystatechange = function(){
    if(xhr.readyState==4&&xhr.status==200){
        $("#show").html(xhr.responseText);
    }
}
//4.设置Content-Type请求消息头(POST专属)
xhr.setRequestHeader("Content-Type","application/x-www-form-urlencoded");
//5.发送请求 - send,有请求体
var uname = $("#uname").val();
var upwd = $("#upwd").val();
var data = "uname="+uname+"&upwd="+upwd;
xhr.send(data);
~~~

## JQuery解析数组

> 使用JQuery的each()函数迭代数组

+ $arr.each()
  + $arr:JQuery中的数组
  + $("div") : 返回的就是JQuery的数组

~~~javascript
$arr.each(function(i,obj){
		i:迭代的元素的下标索引
		obj:迭代出来的元素
});
~~~

+ $.each()
  + $ : 表示的就是 jQuery

~~~javascript
//arr : 表示的是JS中的原生数组
$.each(arr,function(i,obj){
});
~~~

## JSON

**服务端**

> 使用 json.dumps 时,要保证字典|元组|列表中的内容也必须能被JSON序列化(Serializable),允许被JSON序列化的内容(**字符串|字典|元组|列表**)

~~~python
import json
jsonStr = json.dumps(字典|元组|列表)
return jsonStr
~~~

**前端**

> JS对象=JSON.parse(JSON格式字符串);

~~~javascript
var resTxt = xhr.responseText;
//通过JSON.parse()将resTxt转换为JS对象
var arr = JSON.parse(resTxt);
//通过$.each()循环遍历arr
$.each(arr,function(i,obj){
    console.log("姓名:"+obj.uname);
    console.log("密码:"+obj.upwd);
    console.log("邮箱:"+obj.uemail);
});
~~~

## JQuey-Ajax

### $.get()

+ $.get(url[,data] [,callback] [,type])

  + data : 请求提交的数据,可以省略

    1. 可以是字符串
       	"name1=value&name2=value2"
    2. 可以是json对象
       	{"name1":"value1","name2":"value2"}

  + callback : 请求完成后要执行的操作,可以省略

    ~~~javascript
    function(resText){
    	resText:表示的是响应回来的数据}
    ~~~

  + type : 响应回来的数据的类型

    | html | 响应的文本是html文本     |
    | ---- | ------------------------ |
    | text | 响应的文本是普通文本     |
    | json | 响应回来的数据是JSON对象 |

~~~javascript
$.get('/01-search','uname='+this.value,function(data){
      //data是响应回来的数据,直接被转换成了JS数组
      if(data.length > 0){
        $("#show").html('');
        $("#show").css('display','block');
        $.each(data,function(i,obj){
          var $p = $("<p>"+obj+"</p>");
          $p.click(function(){
            $("#uname").val(this.innerHTML);
          });
          $("#show").append($p);
        });
      }else{
        $("#show").css('display','none');
      }
},'json');
~~~

### $.post()

+ $.post(url[,data] [,callback] [,type])

~~~javascript
var params = {
    "uname":$("#uname").val(),
    "upwd":$("#upwd").val(),
    "uemail":$("#uemail").val()
}
//2.post请求
$.post('/02-post',params,function(data){
    //data 表示响应回来的数据
    alert(data);
});
~~~

### $.ajax()

> 自定义所有的请求参数,向服务器端发送请求

+ $.ajax(JSON)

+ JSON中允许包含的属性:

  | url        | 请求地址                                                     |
  | ---------- | ------------------------------------------------------------ |
  | type       | 表示请求方式 'get' 或 'post'                                 |
  | data       | 传递到服务器端的请求参数                                     |
  | async      | 是否使用异步方式发送请求                                     |
  | dataType   | 响应回来的数据格式                                           |
  | success    | 回调函数,请求和响应成功时的回调函数                          |
  | error      | 回调函数,请求或响应失败时的回调函数                          |
  | beforeSend | 回调函数,发送ajax请求之前要执行的操作,如果返回 true,则继续发送请求,返回false 则终止请求的发送 |

~~~javascript
var params = {
      "uname":$("#uname").val(),
      "upwd":$("#upwd").val(),
      "uemail":$("#uemail").val()}
//$.ajax()
$.ajax({
    "url":"/02-post",
    "type":"post",
    "data":params,
    "async":true,
    "success":function(data){
        //data表示响应回来的数据
        alert(data);
    },
    "error":function(){
        alert("程序内部错误!");
    }
});
~~~

## 跨域

> HTTP中有一个策略 - "同源策略"
> 同源:在多个地址中,相同协议,相同域名,相同端口被视为是"同源"

~~~html
<http://www.tedu.cn/a.html
<http://www.tedu.cn/a_server
<!--以上地址是"同源"地址-->
<http://www.tedu.cn/a.html
<https://www.tedu.cn/a_server
<!--以上地址是"非同源"地址，协议不同-->
<http://localhost:5000/a.html
<http://127.0.0.1:5000/a_server
<!--以上地址是"非同源"地址(域名不一样)-->
~~~

>1.非同源地址是不能发送ajax请求的
>2.<img> 和 <script> 是不受同源策略限制的

> 解决同源策略：通过 <script> 向服务器发送请求

~~~javascript
$("#btnSend").click(function(){
        //1.动态创建<script>元素
        var script=document.createElement("script");
        //2.为script元素设置属性 - type
        script.type="text/javascript";
        //3.为script元素设置属性 - src(等同于设置请求地址)
        script.src="http://127.0.0.1:5000/05-server?callback=aaa";
        //4.将script元素加载到DOM树上即加载到网页中,也就是向src的地址发送请求并接收响应.
        var body=$("body")[0]; //获取body元素
        body.append(script); //等同于向src发送请求
});
~~~