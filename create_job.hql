create database feigu3;

use feigu3;

----抓取帖子原始信息
drop table if exists stg_news;
create table if not exists stg_news
(
mysql_newsid          string  comment  'mysql thread id, PK',
news_title   string comment 'thread title',
content     string comment 'news content',
create_time        string comment 'thread post timestamp in feigu Dz'
)
comment 'all flat thread from Dz'
PARTITIONED BY (
  `pt` string  comment 'news dump date(format yyyymmdd)' )
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\001'
NULL DEFINED AS ''
STORED AS TEXTFILE;

----抓取职位原始信息
drop table if exists stg_job;
create table if not exists stg_job
(
web_id          string  comment  'web id',
web_type   string comment 'website type, fixed 01, 02,...',
job_url     string comment 'job url',
job_name        string comment 'job name',
job_location string   comment 'job location',
job_desc  string  comment 'job desc',
edu  string  comment 'education',
gender  string  comment 'gender',
language  string  comment 'language',
major  string  comment 'major',
work_year  string  comment 'work years',
salary  string  comment 'salary',
company_name  string  comment 'company name',
company_desc  string  comment 'company desc',
company_address  string  comment 'company address',
company_worktype  string  comment 'company worktype',
company_scale  string  comment 'company scale',
company_prop  string  comment 'company property',
company_website  string  comment 'company website',
curl_timestamp  string  comment 'curl timestamp'
)
comment 'all flat data from webpage'
PARTITIONED BY (
  `pt` string  comment 'job post date(format yyyymmdd)' )
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\001'
NULL DEFINED AS ''
STORED AS TEXTFILE;

----去噪后的职位信息表
drop table if exists s_job;
create table if not exists s_job
(
web_id          string  comment  'web id',
web_type   string comment 'website type, fixed 01, 02',
job_url     string comment 'job url',
job_name        string comment 'job name',
job_location string   comment 'job location',
job_desc  string  comment 'job desc',
edu  string  comment 'education',
gender  string  comment 'gender',
language  string  comment 'language',
major  string  comment 'major',
work_year  string  comment 'work years',
salary  string  comment 'salary',
company_name  string  comment 'company name',
company_desc  string  comment 'company desc',
company_address  string  comment 'company address',
company_worktype  string  comment 'company worktype',
company_scale  string  comment 'company scale',
company_prop  string  comment 'company property',
company_website  string  comment 'company website',
curl_timestamp  string  comment 'curl timestamp'
)
comment 'remove empty value for dims'
PARTITIONED BY (
  `pt` string  comment 'job post date(format yyyymmdd)' )
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\001'
NULL DEFINED AS ''
STORED AS sequencefile;


-----增加vip计算字段后的职位表
drop table if exists dm_job;
create table if not exists dm_job
(
web_id          string  comment  'web id',
web_type   string comment 'website type, fixed 01, 02',
job_url     string comment 'job url',
job_name        string comment 'job name',
job_location string   comment 'job location',
job_desc  string  comment 'job desc',
edu  string  comment 'education',
gender  string  comment 'gender',
language  string  comment 'language',
major  string  comment 'major',
work_year  string  comment 'work years',
salary  string  comment 'salary',
company_name  string  comment 'company name',
company_desc  string  comment 'company desc',
company_address  string  comment 'company address',
company_worktype  string  comment 'company worktype',
company_scale  string  comment 'company scale',
company_prop  string  comment 'company property',
company_website  string  comment 'company website',
curl_timestamp  string  comment 'curl timestamp',
vip_flg string comment 'is vip jobinfo, 0 or 1'
)
comment 'compute vip flag from s_job'
PARTITIONED BY (
  `pt` string  comment 'job post date(format yyyymmdd)' )
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\001'
NULL DEFINED AS ''
STORED AS sequencefile;

------学历维度表(dim_edu)
drop table if exists dim_edu;
create table if not exists dim_edu
(
web_type   string comment 'website type, fixed 01, 02',
job_name        string comment 'job name',
company_name  string  comment 'company name',
edu_detail  string  comment 'raw data from webpage edu info',
edu_type string comment 'edu enum type 01,02,03'
)
comment 'edu dimision info from s_job'
PARTITIONED BY (
  `pt` string  comment 'job post date(format yyyymmdd)' )
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\001'
NULL DEFINED AS ''
STORED AS sequencefile;


------工作年限维度表(dim_workyear)
drop table if exists dim_workyear;
create table if not exists dim_workyear
(
web_type   string comment 'website type, fixed 01, 02',
job_name        string comment 'job name',
company_name  string  comment 'company name',
workyear_detail  string  comment 'raw data from webpage work years info',
workyear_type string comment 'work year enum type 01,02,03'
)
comment 'work years dimision info from s_job'
PARTITIONED BY (
  `pt` string  comment 'job post date(format yyyymmdd)' )
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\001'
NULL DEFINED AS ''
STORED AS sequencefile;


------职位地域维度表(dim_joblocation)
drop table if exists dim_joblocation;
create table if not exists dim_joblocation
(
web_type   string comment 'website type, fixed 01, 02',
job_name        string comment 'job name',
company_name  string  comment 'company name',
joblocation_detail  string  comment 'raw data from webpage job location info',
joblocation_type string comment 'job location enum type 01,02,03'
)
comment 'job location dimision info from s_job'
PARTITIONED BY (
  `pt` string  comment 'job post date(format yyyymmdd)' )
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\001'
NULL DEFINED AS ''
STORED AS sequencefile;


------薪资维度表(dim_salary)
drop table if exists dim_salary;
create table if not exists dim_salary
(
web_type   string comment 'website type, fixed 01, 02',
job_name        string comment 'job name',
company_name  string  comment 'company name',
salary_detail  string  comment 'raw data from webpage salary info',
salary_type string comment 'salary enum type 01,02,03'
)
comment 'job salary dimision info from s_job'
PARTITIONED BY (
  `pt` string  comment 'job post date(format yyyymmdd)' )
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\001'
NULL DEFINED AS ''
STORED AS sequencefile;


-----和hbase对应的左连维度表后的职位表
drop table if exists rpt_job;
create table if not exists rpt_job
(
web_id          string  comment  'web id',
web_type   string comment 'website type, fixed 01, 02',
job_url     string comment 'job url',
job_name        string comment 'job name',
job_location string   comment 'job location',
joblocation_type string   comment 'dim_joblocation.joblocation_type',
job_desc  string  comment 'job desc',
edu  string  comment 'education',
edu_type string comment 'dim_edu.edu_type',
gender  string  comment 'gender',
language  string  comment 'language',
major  string  comment 'major',
work_year  string  comment 'work years',
workyear_type  string comment  'dim_workyear.workyear_type',
salary  string  comment 'salary',
salary_type string comment 'dim_salary.salary_type',
company_name  string  comment 'company name',
company_desc  string  comment 'company desc',
company_address  string  comment 'company address',
company_worktype  string  comment 'company worktype',
company_scale  string  comment 'company scale',
company_prop  string  comment 'company property',
company_website  string  comment 'company website',
curl_timestamp  string  comment 'curl timestamp',
vip_flg string comment 'is vip jobinfo, 0 or 1'
)
comment 'dm_job left join dim_* , the output data dump into hbase'
PARTITIONED BY (
  `pt` string  comment 'job post date(format yyyymmdd)' )
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\001'
NULL DEFINED AS ''
STORED AS sequencefile;


------每日维度统计表(daily_dim_sum)
drop table if exists daily_dim_sum;
create table if not exists daily_dim_sum
(
dim_type  string  comment 'dim types',
cnt_val  int comment 'count(job_name) group by dim_type'
)
comment 'daily dimisons sum info'
PARTITIONED BY (
  `pt` string  comment 'job post date(format yyyymmdd)' )
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\001'
NULL DEFINED AS ''
STORED AS sequencefile;
