with revenue_s as (
		select
			l_suppkey as supplier_no,
			sum(l_extendedprice * (1-l_discount)) as total_revenue
		from
			lineitem
		where
			l_shipdate >= date ':1'
			and l_shipdate < date ':1' + interval '3' month
		group by
			l_suppkey
)
select
	s_suppkey,
	s_name,
	s_address,
	s_phone,
	total_revenue
from
	supplier,
	revenue_s
where
	s_suppkey = supplier_no
	and total_revenue = (
		select
			max(total_revenue)
		from
			revenue_s
	)
order by
	s_suppkey;
