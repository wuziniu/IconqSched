import numpy as np
import torch


def simple_queueing_func(x, a1, a2, b1):
    """
    a1 represents the average exec-time of a random query under concurrency
    b1 represents the max level of concurrency in a system
    a2 represents the average impact on a query's runtime when executed concurrently with other queries
    """
    num_concurrency, isolated_runtime = x
    return (a1 * np.maximum(num_concurrency - b1, 0)) + (
        1 + a2 * np.minimum(num_concurrency, b1)
    ) * isolated_runtime


def interaction_func_scipy(
    x,
    q1,
    i1,
    i2,
    c1,
    m1,
    m2,
    m3,
    cm1,
    r1,
    r2,
    max_concurrency,
    avg_io_speed,
    memory_size,
):
    """
    An analytical function that can consider 3 types of resource sharing/contention: IO, memory, CPU
    x:: input tuple containing:
        isolated_runtime: the isolated runtime without concurrency of a query
        avg_runtime: average or median observed runtime of a query under any concurrency
        num_concurrency: number of concurrent queries running with this query
        sum_concurrent_runtime: sum of the estimated runtime of all queries concurrently running with this query (CPU)
        est_scan: estimated MB of data that this query will need to scan (IO)
        est_concurrent_scan: estimated MB of data that the concurrently running queries will need to scan (IO)
        scan_sharing_percentage: estimated percentage of data in cache (sharing) according to concurrent queries
        max_est_card: maximum estimated cardinality in the query plan of this query (reflect peak memory usage)
        avg_est_card: average estimated cardinality in the query plan of this query (reflect average memory usage)
        max_concurrent_card: maximum estimated cardinality for all concurrent queries
        avg_concurrent_card: average estimated cardinality for all concurrent queries
    TODO: adding memory and CPU information
    """
    (
        isolated_runtime,
        avg_runtime,
        num_concurrency,
        sum_concurrent_runtime,
        est_scan,
        est_concurrent_scan,
        scan_sharing_percentage,
        max_est_card,
        avg_est_card,
        max_concurrent_card,
        avg_concurrent_card,
    ) = x
    # fraction of running queries (as opposed to queueing queries)
    running_frac = np.minimum(num_concurrency, max_concurrency) / np.maximum(
        num_concurrency, 1
    )
    # estimate queueing time of a query based on the sum of concurrent queries' run time
    queueing_time = (
        q1
        * (
            np.maximum(num_concurrency - max_concurrency, 0)
            / np.maximum(num_concurrency, 1)
        )
        * sum_concurrent_runtime
    )
    # estimate io_speed of a query assuming each query has a base io_speed of i1 + the io speed due to contention
    io_speed = i1 + avg_io_speed / np.minimum(
        np.maximum(num_concurrency, 1), max_concurrency
    )
    # estimate time speed on IO as the (estimated scan - data in cache) / estimated io_speed
    # use i2 to adjust the estimation error in est_scan and scan_sharing_percentage
    io_time = i2 * est_scan * (1 - scan_sharing_percentage) / io_speed
    # estimate the amount of CPU work/time as the weighted average of isolated_runtime and avg_runtime - io_time
    cpu_time_isolated = (r1 * isolated_runtime + r2 * avg_runtime) - io_time
    # estimate the amount of CPU work imposed by the concurrent queries (approximated by their estimate runtime)
    cpu_concurrent = (running_frac * sum_concurrent_runtime) / avg_runtime
    # estimate the amount of memory load imposed by the concurrent queries
    max_mem_usage_perc = max_concurrent_card / (max_concurrent_card + max_est_card)
    avg_mem_usage_perc = avg_concurrent_card / (avg_concurrent_card + avg_est_card)
    memory_concurrent = np.log(
        m1
        * np.maximum(max_concurrent_card + max_est_card - memory_size, 0.01)
        * max_mem_usage_perc
        + m2
        * np.maximum(avg_concurrent_card + avg_est_card - memory_size, 0.01)
        * avg_mem_usage_perc
        + 0.0001
    ) * np.log(m1 * max_est_card + m2 * avg_est_card + 0.0001)
    memory_concurrent = np.maximum(memory_concurrent, 0)
    # estimate the CPU time of a query by considering the contention of CPU and memory of other queries
    cpu_time = (
        1
        + c1 * cpu_concurrent
        + m3 * memory_concurrent
        + cm1 * np.sqrt(cpu_concurrent * memory_concurrent)
    ) * cpu_time_isolated
    # final runtime of a query is estimated to be the queueing time + io_time + cpu_time
    return np.maximum(queueing_time + io_time + cpu_time, 0.01)


def interaction_func_torch(
    x,
    q1,
    i1,
    i2,
    c1,
    m1,
    m2,
    m3,
    cm1,
    r1,
    r2,
    max_concurrency,
    avg_io_speed,
    memory_size,
):
    # See interaction_func_scipy for explanation
    (
        isolated_runtime,
        avg_runtime,
        num_concurrency,
        sum_concurrent_runtime,
        est_scan,
        est_concurrent_scan,
        scan_sharing_percentage,
        max_est_card,
        avg_est_card,
        max_concurrent_card,
        avg_concurrent_card,
    ) = x.T
    num_query = len(num_concurrency)
    running_frac = torch.minimum(num_concurrency, max_concurrency) / torch.maximum(
        num_concurrency, torch.tensor(1)
    )
    # estimate queueing time of a query based on the sum of concurrent queries' run time
    queueing_time = (
        q1
        * (
            torch.maximum(num_concurrency - max_concurrency, torch.tensor(0))
            / torch.maximum(num_concurrency, torch.tensor(1))
        )
        * sum_concurrent_runtime
    )
    # estimate io_speed of a query assuming each query has a base io_speed of i1 + the io speed due to contention
    io_speed = i1 + avg_io_speed / torch.minimum(
        torch.maximum(num_concurrency, torch.tensor(1)), max_concurrency
    )
    # estimate time speed on IO as the (estimated scan - data in cache) / estimated io_speed
    # use i2 to adjust the estimation error in est_scan and scan_sharing_percentage
    io_time = i2 * est_scan * (1 - scan_sharing_percentage) / io_speed
    # estimate the amount of CPU work/time as the weighted average of isolated_runtime and avg_runtime - io_time
    cpu_time_isolated = (r1 * isolated_runtime + r2 * avg_runtime) - io_time
    # estimate the amount of CPU work imposed by the concurrent queries (approximated by their estimate runtime)
    cpu_concurrent = (running_frac * sum_concurrent_runtime) / avg_runtime
    # estimate the amount of memory load imposed by the concurrent queries
    max_mem_usage_perc = max_concurrent_card / (max_concurrent_card + max_est_card)
    avg_mem_usage_perc = avg_concurrent_card / (avg_concurrent_card + avg_est_card)
    memory_concurrent = torch.log(
        m1
        * torch.maximum(
            max_concurrent_card + max_est_card - memory_size, torch.tensor(0) + 0.01
        )
        * max_mem_usage_perc
        + m2
        * torch.maximum(
            avg_concurrent_card + avg_est_card - memory_size, torch.tensor(0) + 0.01
        )
        * avg_mem_usage_perc
        + 0.0001
    ) * torch.log(m1 * max_est_card + m2 * avg_est_card + 0.0001)
    memory_concurrent = torch.maximum(memory_concurrent, torch.tensor(0))
    # estimate the CPU time of a query by considering the contention of CPU and memory of other queries
    cpu_time = (
        1
        + c1 * cpu_concurrent
        + m3 * memory_concurrent
        + cm1 * torch.sqrt(cpu_concurrent * memory_concurrent)
    ) * cpu_time_isolated
    # final runtime of a query is estimated to be the queueing time + io_time + cpu_time
    return torch.maximum(queueing_time + io_time + cpu_time, torch.tensor(0) + 0.01)


def interaction_separation_func_scipy(
    x,
    n1,
    q1,
    i1,
    i2,
    c1,
    c2,
    m1,
    m2,
    m3,
    m4,
    m5,
    cm1,
    r1,
    r2,
    max_concurrency,
    avg_io_speed,
    memory_size,
):
    """
    An analytical function that can consider 3 types of resource sharing/contention: IO, memory, CPU
    x:: input tuple containing:
        isolated_runtime: the isolated runtime without concurrency of a query
        avg_runtime: average or median observed runtime of a query under any concurrency
        num_concurrency: number of concurrent queries running with this query
        sum_concurrent_runtime: sum of the estimated runtime of all queries concurrently running with this query (CPU)
        est_scan: estimated MB of data that this query will need to scan (IO)
        est_concurrent_scan: estimated MB of data that the concurrently running queries will need to scan (IO)
        scan_sharing_percentage: estimated percentage of data in cache (sharing) according to concurrent queries
        max_est_card: maximum estimated cardinality in the query plan of this query (reflect peak memory usage)
        avg_est_card: average estimated cardinality in the query plan of this query (reflect average memory usage)
        max_concurrent_card: maximum estimated cardinality for all concurrent queries
        avg_concurrent_card: average estimated cardinality for all concurrent queries
    TODO: adding memory, vCPU, and bandwidth information
    """
    (
        isolated_runtime,
        avg_runtime,
        num_concurrency_pre,
        num_concurrency_post,
        sum_concurrent_runtime_pre,
        sum_concurrent_runtime_post,
        avg_time_elapsed_pre,
        sum_time_overlap_post,
        est_scan,
        est_concurrent_scan_pre,
        est_concurrent_scan_post,
        scan_sharing_percentage,
        max_est_card,
        avg_est_card,
        max_concurrent_card_pre,
        max_concurrent_card_post,
        avg_concurrent_card_pre,
        avg_concurrent_card_post,
    ) = x
    # fraction of running queries (as opposed to queueing queries)
    running_frac = np.minimum(
        num_concurrency_pre + num_concurrency_post, max_concurrency
    ) / np.maximum(num_concurrency_pre + num_concurrency_post, 1)
    # estimate queueing time of a query based on the sum of concurrent queries' run time
    queueing_time = (
        q1
        * (
            np.maximum(
                num_concurrency_pre + n1 * num_concurrency_post - max_concurrency, 0
            )
            / np.maximum(num_concurrency_pre + n1 * num_concurrency_post, 1)
        )
        * (
            sum_concurrent_runtime_pre
            + n1 * sum_concurrent_runtime_post
            - avg_time_elapsed_pre * num_concurrency_pre
        )
    )
    queueing_time = np.maximum(queueing_time, 0)
    discount_pre = (
        (sum_concurrent_runtime_pre - avg_time_elapsed_pre * num_concurrency_pre)
        * running_frac
        / np.maximum(sum_concurrent_runtime_pre, 0.1)
    )
    discount_pre = np.maximum(discount_pre, 0)
    discount_post = sum_time_overlap_post / np.maximum(sum_concurrent_runtime_post, 0.1)
    # estimate io_speed of a query assuming each query has a base io_speed of i1 + the io speed due to contention
    io_speed = i1 + avg_io_speed / np.minimum(
        np.maximum(
            num_concurrency_pre * discount_pre
            + n1 * num_concurrency_post * discount_post,
            1,
        ),
        max_concurrency,
    )
    # estimate time speed on IO as the (estimated scan - data in cache) / estimated io_speed
    # use i2 to adjust the estimation error in est_scan and scan_sharing_percentage
    io_time = i2 * est_scan * (1 - scan_sharing_percentage * running_frac) / io_speed
    # estimate the amount of CPU work/time as the weighted average of isolated_runtime and avg_runtime - io_time
    cpu_time_isolated = np.maximum(
        (r1 * isolated_runtime + r2 * avg_runtime) - io_time, 0.1
    )
    # estimate the amount of CPU work imposed by the concurrent queries (approximated by their estimate runtime)
    cpu_concurrent_pre = (sum_concurrent_runtime_pre * discount_pre) / avg_runtime
    cpu_concurrent_post = sum_time_overlap_post / avg_runtime
    # print("cpu_concurrent_pre:", np.min(cpu_concurrent_pre))
    # print("cpu_concurrent_post:", np.min(cpu_concurrent_post))
    # estimate the amount of memory load imposed by the concurrent queries
    max_mem_usage_perc_pre = max_est_card / (max_concurrent_card_pre + max_est_card)
    avg_mem_usage_perc_pre = avg_est_card / (avg_concurrent_card_pre + avg_est_card)
    max_mem_usage_perc_post = max_est_card / (max_concurrent_card_post + max_est_card)
    avg_mem_usage_perc_post = avg_est_card / (avg_concurrent_card_post + avg_est_card)
    peak_mem_usage = (
        m1
        * np.maximum(max_concurrent_card_pre + max_est_card - memory_size, 0.01)
        * max_mem_usage_perc_pre
        + m1
        * np.maximum(max_concurrent_card_post + max_est_card - memory_size, 0.01)
        * max_mem_usage_perc_post
    ) / memory_size
    avg_mem_usage = (
        m2
        * np.maximum(avg_concurrent_card_pre + avg_est_card - memory_size, 0.01)
        * avg_mem_usage_perc_pre
        + m2
        * np.maximum(avg_concurrent_card_post + avg_est_card - memory_size, 0.01)
        * avg_mem_usage_perc_post
    ) / memory_size
    mem_usage = m3 * peak_mem_usage + m4 * avg_mem_usage
    # estimate the CPU time of a query by considering the contention of CPU and memory of other queries

    cpu_time_scale_factor = (c1 * (cpu_concurrent_pre + c2 * cpu_concurrent_post)) * (
        1
        + m5 * mem_usage
        + cm1 * np.sqrt((cpu_concurrent_pre + cpu_concurrent_post) * mem_usage)
    )
    cpu_time = (1 + cpu_time_scale_factor) * cpu_time_isolated
    # final runtime of a query is estimated to be the queueing time + io_time + cpu_time
    # print(np.min(queueing_time), np.min(io_time), np.min(cpu_time))
    return np.maximum(queueing_time + io_time + cpu_time, 0.01)
