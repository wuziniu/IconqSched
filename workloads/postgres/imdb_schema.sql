CREATE TABLE aka_name (
    id integer NOT NULL PRIMARY KEY,
    person_id integer,
    name character varying,
    imdb_index character varying(3),
    name_pcode_cf character varying(11),
    name_pcode_nf character varying(11),
    surname_pcode character varying(11),
    md5sum character varying(65)
);

CREATE TABLE aka_title (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer,
    title character varying,
    imdb_index character varying(4),
    kind_id integer,
    production_year integer,
    phonetic_code character varying(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    note character varying(72),
    md5sum character varying(32)
);

CREATE TABLE cast_info (
    id integer NOT NULL PRIMARY KEY,
    person_id integer,
    movie_id integer,
    person_role_id integer,
    note character varying,
    nr_order integer,
    role_id integer
);

CREATE TABLE char_name (
    id integer NOT NULL PRIMARY KEY,
    name character varying,
    imdb_index character varying(2),
    imdb_id integer,
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32)
);


CREATE TABLE comp_cast_type (
    id integer NOT NULL PRIMARY KEY,
    kind character varying(32)
);

CREATE TABLE company_name (
    id integer NOT NULL PRIMARY KEY,
    name character varying,
    country_code character varying(6),
    imdb_id integer,
    name_pcode_nf character varying(5),
    name_pcode_sf character varying(5),
    md5sum character varying(32)
);

CREATE TABLE company_type (
    id integer NOT NULL PRIMARY KEY,
    kind character varying(32)
);

CREATE TABLE complete_cast (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer,
    subject_id integer,
    status_id integer
);

CREATE TABLE info_type (
    id integer NOT NULL PRIMARY KEY,
    info character varying(32)
);

CREATE TABLE keyword (
    id integer NOT NULL PRIMARY KEY,
    keyword character varying,
    phonetic_code character varying(5)
);

CREATE TABLE kind_type (
    id integer NOT NULL PRIMARY KEY,
    kind character varying(15)
);

CREATE TABLE link_type (
    id integer NOT NULL PRIMARY KEY,
    link character varying(32)
);

CREATE TABLE movie_companies (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer,
    company_id integer,
    company_type_id integer,
    note character varying
);

CREATE TABLE movie_info_idx (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer,
    info_type_id integer,
    info character varying,
    note character varying(1)
);

CREATE TABLE movie_keyword (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer,
    keyword_id integer
);

CREATE TABLE movie_link (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer,
    linked_movie_id integer,
    link_type_id integer
);

CREATE TABLE name (
    id integer NOT NULL PRIMARY KEY,
    name character varying,
    imdb_index character varying(9),
    imdb_id integer,
    gender character varying(1),
    name_pcode_cf character varying(5),
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32)
);

CREATE TABLE role_type (
    id integer NOT NULL PRIMARY KEY,
    role character varying(32)
);

CREATE TABLE title (
    id integer NOT NULL PRIMARY KEY,
    title character varying,
    imdb_index character varying(5),
    kind_id integer,
    production_year integer,
    imdb_id integer,
    phonetic_code character varying(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    series_years character varying(49),
    md5sum character varying(32)
);

CREATE TABLE movie_info (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer,
    info_type_id integer,
    info character varying,
    note character varying
);

CREATE TABLE person_info (
    id integer NOT NULL PRIMARY KEY,
    person_id integer,
    info_type_id integer,
    info character varying,
    note character varying
);