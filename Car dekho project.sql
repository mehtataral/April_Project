select * from `car dekho`;

select count(*) from `car dekho`;

-- total cars : to get count
select count(distinct name) from `car dekho`;

-- Clean Data bcz automatic comes 2 times
update `car dekho` 
set transmission  = trim(transmission);


--  trend to check price of cars like automatic and manual over year

select transmission, avg(selling_price)
from `car dekho`
group by transmission;

select transmission,year, avg(selling_price)
from `car dekho`
group by transmission,year
order by avg(selling_price) desc;

--  trend to check price of cars like automatic and manual over year
select fuel, year, avg(selling_price) 
from `car dekho`
group by fuel,year
order by year desc;

-- -- check trends in the market , which car has more preference
select year,fuel, count(*)
from `car dekho`
group by year,fuel
order by year desc;

select * from `car dekho`;

-- --------------------------------------------------------------

-- how many cars available for 2023
select count(*) from `car dekho`
where year = 2023;

select count(*) from `car dekho`
where year = 2021;
select count(*) from `car dekho`
where year = 2022;
select count(*) from `car dekho`
where year = 2020;

-- which year car has sell more  - 2017

select year,count(year) 
from `car dekho`
group by year
order by count(year) desc;

-- how many diesel car available in 2017
select count(*) from `car dekho` 
where fuel = "diesel" and year = 2017;

-- how many petrol car available in 2017

select  count(*) from `car dekho`
where fuel = "petrol" and year = 2017;

-- how many cng car available in 2017

select  count(*) from `car dekho`
where fuel = "cng" and year = 2017;


-- petrol ,cng and diesel car 

select fuel, count(*) from `car dekho`
group by fuel;

-- petrol ,cng and diesel car year wise
select year, count(*) from `car dekho`
group by year
order by year desc;


select year, count(*) from `car dekho`
where fuel ="petrol"
group by year
order by year desc;

select year, count(*) from `car dekho`
where fuel ="CNG"
group by year
order by year desc;

select year, count(*) from `car dekho`
where fuel ="diesel"
group by year
order by year desc;
-- -----------------avg mileage as per the seats and fuel type and trasnmission  type-----------

select seats, avg(mileage) from `car dekho`
where transmission ="automatic" and fuel  = "petrol"
group by seats;

select seats, avg(mileage) from `car dekho`
where transmission ="automatic" and fuel  = "diesel"
group by seats;


select seats, avg(mileage) from `car dekho`
where transmission ="Manual" and fuel  = "petrol"
group by seats;

select seats, avg(mileage) from `car dekho`
where transmission ="Manual" and fuel  = "diesel"
group by seats;

-- ------------------------------------

select seats, avg(mileage) from `car dekho`
where transmission ="manual"
group by seats;

select * from `car dekho`
where km_driven <5000 and year >2019 and owner = "First Owner" and transmission ="Automatic";

-- ----------------------------------------------------------------







