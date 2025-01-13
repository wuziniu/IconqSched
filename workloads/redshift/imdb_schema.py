from typing import List


REDSHIFT_IMDB_SCHEMA: List[str] = [
    'CREATE TABLE aka_name (id BIGINT, person_id BIGINT, name VARCHAR(MAX), imdb_index VARCHAR(256), name_pcode_cf VARCHAR(256), name_pcode_nf VARCHAR(256), surname_pcode VARCHAR(256), md5sum VARCHAR(256), PRIMARY KEY (id));',
    'CREATE TABLE aka_title (id BIGINT, movie_id BIGINT, title VARCHAR(MAX), imdb_index VARCHAR(256), kind_id BIGINT, production_year BIGINT, phonetic_code VARCHAR(256), episode_of_id BIGINT, season_nr BIGINT, episode_nr BIGINT, note VARCHAR(256), md5sum VARCHAR(256), PRIMARY KEY (id));',
    'CREATE TABLE cast_info (id BIGINT, person_id BIGINT, movie_id BIGINT, person_role_id BIGINT, note VARCHAR(MAX), nr_order BIGINT, role_id BIGINT, PRIMARY KEY (id));',
    'CREATE TABLE char_name (id BIGINT, name VARCHAR(MAX), imdb_index VARCHAR(256), imdb_id BIGINT, name_pcode_nf VARCHAR(256), surname_pcode VARCHAR(256), md5sum VARCHAR(256), PRIMARY KEY (id));',
    'CREATE TABLE comp_cast_type (id BIGINT, kind VARCHAR(256), PRIMARY KEY (id));',
    'CREATE TABLE company_name (id BIGINT, name VARCHAR(MAX), country_code VARCHAR(256), imdb_id BIGINT, name_pcode_nf VARCHAR(256), name_pcode_sf VARCHAR(256), md5sum VARCHAR(256), PRIMARY KEY (id));',
    'CREATE TABLE company_type (id BIGINT, kind VARCHAR(256), PRIMARY KEY (id));',
    'CREATE TABLE complete_cast (id BIGINT, movie_id BIGINT, subject_id BIGINT, status_id BIGINT, PRIMARY KEY (id));',
    'CREATE TABLE info_type (id BIGINT, info VARCHAR(256), PRIMARY KEY (id));',
    'CREATE TABLE keyword (id BIGINT, keyword VARCHAR(MAX), phonetic_code VARCHAR(256), PRIMARY KEY (id));',
    'CREATE TABLE kind_type (id BIGINT, kind VARCHAR(256), PRIMARY KEY (id));',
    'CREATE TABLE link_type (id BIGINT, link VARCHAR(256), PRIMARY KEY (id));',
    'CREATE TABLE movie_companies (id BIGINT, movie_id BIGINT, company_id BIGINT, company_type_id BIGINT, note VARCHAR(MAX), PRIMARY KEY (id));',
    'CREATE TABLE movie_info_idx (id BIGINT, movie_id BIGINT, info_type_id BIGINT, info VARCHAR(MAX), note VARCHAR(256), PRIMARY KEY (id));',
    'CREATE TABLE movie_keyword (id BIGINT, movie_id BIGINT, keyword_id BIGINT, PRIMARY KEY (id));',
    'CREATE TABLE movie_link (id BIGINT, movie_id BIGINT, linked_movie_id BIGINT, link_type_id BIGINT, PRIMARY KEY (id));',
    'CREATE TABLE name (id BIGINT, name VARCHAR(MAX), imdb_index VARCHAR(256), imdb_id BIGINT, gender VARCHAR(256), name_pcode_cf VARCHAR(256), name_pcode_nf VARCHAR(256), surname_pcode VARCHAR(256), md5sum VARCHAR(256), PRIMARY KEY (id));',
    'CREATE TABLE role_type (id BIGINT, role VARCHAR(256), PRIMARY KEY (id));',
    'CREATE TABLE title (id BIGINT, title VARCHAR(MAX), imdb_index VARCHAR(256), kind_id BIGINT, production_year BIGINT, imdb_id BIGINT, phonetic_code VARCHAR(256), episode_of_id BIGINT, season_nr BIGINT, episode_nr BIGINT, series_years VARCHAR(256), md5sum VARCHAR(256), PRIMARY KEY (id));',
    'CREATE TABLE movie_info (id BIGINT, movie_id BIGINT, info_type_id BIGINT, info VARCHAR(MAX), note VARCHAR(MAX), PRIMARY KEY (id));',
    'CREATE TABLE person_info (id BIGINT, person_id BIGINT, info_type_id BIGINT, info VARCHAR(MAX), note VARCHAR(MAX), PRIMARY KEY (id));'
 ]


REDSHIFT_IMDB_TABLE_NAMES: List[str] = ["aka_name", "aka_title", "cast_info", "char_name", "comp_cast_type", "company_name",
                               "company_type", "complete_cast", "info_type", "keyword", "kind_type", "link_type",
                               "movie_companies", "movie_info_idx", "movie_keyword", "movie_link", "name",
                               "role_type", "title", "movie_info", "person_info"]

REDSHIFT_IMDB_LOAD_TEMPLATE: str = """
    COPY {table_name} FROM '{s3_path}'
    IAM_ROLE '{s3_iam_role}'
    CSV IGNOREHEADER 1 delimiter '|' BLANKSASNULL
"""
