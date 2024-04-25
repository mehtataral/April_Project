create database Zomato;    
use zomato;

show tables;
create table zomato_details(r_id int, 
							r_name varchar(50),
							cuisine varchar(50),
                            rating float(3,2),
                            type varchar(20),
                            city varchar(50),
                            phone bigint
                            );
show tables;

-- 1st method
insert into zomato_details values(14,"Aruna","All",1.3,"Non-veg","Andheri",8123456781);

-- DATA TYPE
-- INT FLOAT CHAR VARCHAR -- DATE, 


select * from zomato_details;     

-- 2nd method
insert into zomato_details values(2,"Zaika","North Indian",3.9,"Veg","Lower Parel",8123456781),
								 (3,"Radha Krishna","North Indian",4.3,"Veg","Kandivali",7123456781),
                                 (4,"Bhagat Tarachand","South Indian",4.6,"Veg","Lower Parel",9123454450),
                                 (5,"Shree Nidhi","Chinese",4,"Veg","Andheri",9223776781),
                                 (6,"Amar ","All",5,"Veg","Kandivali",9223776756),
                                 (7,"Veena","All",2.5,"Veg","Malad",9223776781),
                                 (8,"Gurukrupa","South Indian",3.6,"Veg","Lower Parel",9223076744),
                                 (9,"Papilon","All",4.4,"Veg","Andheri",8223076744),
                                 (10,"Veg Sagar","All",2.9,"Veg","Borivali",7223076744),
                                 (11,"Navarnag","North Indian",3.1,"Veg","Malad",6223076744),
                                 (12,"ZSK","Chinese",0.6,"Non-Veg","Lower Parel",9223076720);
                                 
select * from zomato_details;

--  3rd Method
insert into zomato_details (r_id,r_name,cuisine,rating)
values(13,"Prasadam","North Indian",4.2);

insert into zomato_details (r_id,r_name,cuisine,rating)
values(14,"Sankalp","South Indian",4.6);

select * from zomato_details;       

-- where clause
select * from zomato_details
where cuisine ="North Indian";

select * from zomato_details
where cuisine ="All";

-- Conditional Operators  > >= < <= <> 

select * from zomato_details
where rating > 3;

select * from zomato_details
where rating <= 3;

select * from zomato_details
where type <> 'Veg';

-- Logical operators 
-- AND OR NOT

select * from zomato_details
where cuisine ="All" and rating >=3;

select * from zomato_details
where cuisine ="South Indian" or rating >=3;

select * from zomato_details
where cuisine ="South Indian" or rating >=3 and city ="Lower Parel";

select * from zomato_details
where type is null;

select * from zomato_details
where type is not null;
       
select * from zomato_details
where cuisine is null; --  Cuisine is not null 

-- update 
update zomato_details set type = 'veg'
where r_id =13;

select * from zomato_details;

update zomato_details set city ="Andheri"
where r_name = 'Prasadam';

select * from zomato_details;

-- Delete 

delete from zomato_details
where city is null;

select * from zomato_details;

delete from zomato_details
where r_name = "zsk";

select * from zomato_details;

update zomato_details set r_id =12
where r_name = 'Prasadam';

-- Agg. function & alising
select min(rating) as LowestRating from zomato_details ;

select max(rating) as HighestRating from zomato_details ;

select avg(rating) from zomato_details ;

select sum(rating) from zomato_details ;

select count(*) from zomato_details ;

-- between and in
-- IN  
select * from zomato_Details
where rating in(1,2,3,4,5);

select * from zomato_Details
where city in("Andheri","churchgate");

-- BETWEEN
select * from zomato_details
where rating between 1 and 3;

select * from zomato_details
where rating between 0 and 2;	

-- wildcard like 
select * from zomato_details
where city like "%i";

select * from zomato_details
where rating like "_.__";

select * from zomato_details
where city like "%al%";

select * from zomato_details
where r_id like "1%";

-- order by
select * from zomato_details
order by rating;

select * from zomato_details
order by rating desc;

select * from zomato_details
order by cuisine;

select * from zomato_details
order by cuisine asc, rating asc;  -- Multiple rows in ascening format

-- limit 
select * from zomato_details
order by rating desc
limit 5;

select * from zomato_details
order by rating desc
limit 1,1;  -- 2nd highest rating

select * from zomato_details
order by rating desc
limit 2,1;  -- 3rd highest rating

select * from zomato_details
order by rating
limit 5;

-- Distinct
select distinct city from zomato_details ;

select distinct *, city from zomato_details 
where rating >3 ;

select distinct * from zomato_details 
where rating >3 and cuisine ="all";

select distinct type from 
zomato_details;

-- Group By
select city , count(city) 
from zomato_details
group by city;

select city , count(city) 
from zomato_details
where city ="Andheri"
group by city;

select type , count(type)
from zomato_details
group by type ;

select city ,type , count(type)
from zomato_details
group by type ;

-- Having
select city , count(city) 
from zomato_details
group by city
having count(city) > 2;

-- select -> where --> group by --> having --> order by  -> limit
select city, count(city) from zomato_details 
where city ="Andheri" or city ="lower parel"
group by city
having count(city)>2
order by count(city) desc
limit 1;  

-- DAY 2

use zomato;
create table zomato_date(resto_id int,resto_name varchar(50), resto_age date);
select * from zomato_date;
-- YYYY-MM-DD
insert into zomato_date values (1,"Zaika","2023-02-25"),
								 (2,"Radha Krishna","2023-02-20"),
                                 (3,"Bhagat Tarachand","2023-01-25"),
                                 (4,"Shree Nidhi","2022-12-31"),
                                 (5,"Shree Gokul","2022-12-25"),
								 (6,"Subh sagar","2023-03-1");
select * from zomato_date;		
insert into zomato_date values (7,"Jalsa","2023-12-25");

select * from zomato_date
where year(resto_age) =2023;
				
select * from zomato_date
where year(resto_age) =2022;

select * from zomato_date
where month(resto_age) =12;

select * from zomato_date
where year(resto_age)>2022;

--  Alter 
alter table zomato_date
add city varchar(50);

select * from zomato_date;

update zomato_date set city ="Andheri" 
where resto_name ="Jalsa";

update zomato_date set city ="Lower Parel" 
where resto_name ="Subh sagar";

update zomato_date set city ="Andheri" 
where resto_name ="Zaika";

update zomato_date set city ="Andheri" 
where resto_name ="Bhagat Tarachand";

alter table zomato_date
drop city ;

alter table zomato_date
modify column resto_age date;

alter table zomato_date
modify column resto_id float;

select * from zomato_date;

describe zomato_date;

insert into zomato_date
values(8,"Amar","2022-10-4");

alter table zomato_date
rename column resto_id to r_id;

select * from zomato_details;

-- -------------------------------------------------------
create table details(id int, r_name varchar(50),review int);

insert into details values(2,"Zaika",230);
insert into details values(20,"Rangila",1320),
						  (5,"Shree Nidhi",111);
                          
select * from details;

-- Inner Join
select * from
zomato_details A inner join details B
on A.r_id = B.id;

-- Left Join
select * from
zomato_details A left join details B
on A.r_id = B.id;

-- Right Join
select * from
zomato_details A right join details B
on A.r_id = B.id;

-- Full Outer Join
select * from
zomato_details A left join details B
on A.r_id = B.id
union 
select * from
zomato_details A right join details B
on A.r_id = B.id;

use zomato;
select * from zomato_details;


-- ---------------------------------------------------------
-- row_number assign unique number to each row if value is repeated same like here rating 1.3 repeated it assign new number
select * ,   
row_number() over(partition by type order by rating desc)
from zomato_details;


-- dense_rank assign unique number to each row if value is repeated number will be same and next number will be 2 
select * ,   
dense_rank() over(partition by type order by rating desc)
from zomato_details;

-- rank assign unique number to each row if value is repeated number will be same and next number will be 3 
select * ,   
rank() over(partition by type order by rating desc)
from zomato_details;


select * , avg(rating)
over(partition by cuisine)as avg_rating_per_cuisine
from zomato_details;



-- rank of each resturant within cuisine type

select *,
rank() over(partition by cuisine order by rating desc) 
from zomato_details;

-- What is the percentile rank of each restaurant's rating within its cuisine type?
select *,
percent_rank() over(partition by type order by rating desc)
from zomato_details;



SELECT
    *,
    LAG(rating) OVER (PARTITION BY type ORDER BY rating desc) AS previous_rating,
    LEAD(rating) OVER (PARTITION BY type ORDER BY rating desc) AS next_rating
FROM
    zomato_details;