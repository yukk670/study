create table customer(
    cust_id varchar(32),
    cust_name varchar(32),
    tel_no varchar(32)
)default charset = utf8;

insert into customer values
('C0001','Jerry','13512345678'),
('C0002','Dekie','13522334455'),
('C0003','Dckas','15844445555'),
('C0004', 'Jerry', '13512345678'),
('C0005', 'Dekie', '13522334455'),
('C0006', 'Dckas', '15844445555'),
('C0007', 'Jerry', '13512345678'),
('C0008', 'Dekie', '13522334455'),
('C0009', 'Dckas', '15844445555'),
('C0010', 'Dckas', '15844445555');

update customer set tel_no = null where cust_id = 'C0003';

-- 模糊查询不精确匹配，查询速度慢，劲量避免%前置
select * from customer where cust_name like 'D%';
-- 去重复
select DISTINCT(cust_name) from customer;

select count(*) from customer where tel_no like '135%';

