import "regent"

-- Helper module to handle command line arguments
local PageRankConfig = require("pagerank_config")

local c = regentlib.c

fspace Page {

  rank         : double,
  -- TODO: Add more fields as you need.
  num_out_edges : uint64,
  delta        : double,
  old_rank     : double,
  pr_diff      : double
}

fspace error_field {
    error : double
}

--
-- TODO: Define fieldspace 'Link' which has two pointer fields,
--       one that points to the source and another to the destination
fspace Link(r : region(Page)) { 
    src_page : ptr(Page, r),
    dst_page   : ptr(Page, r)
}

terra skip_header(f : &c.FILE)
  var x : uint64, y : uint64
  c.fscanf(f, "%llu\n%llu\n", &x, &y)
end

terra read_ids(f : &c.FILE, page_ids : &uint32)
  return c.fscanf(f, "%d %d\n", &page_ids[0], &page_ids[1]) == 2
end

task initialize_graph(r_pages   : region(Page),
                      -- TODO: Give the right region type here.
                      r_links   : region(Link(r_pages)),
                      damp      : double,
                      num_pages : uint64,
                      filename  : int8[512])
where
  reads writes(r_pages, r_links)
do
  var ts_start = c.legion_get_current_time_in_micros()
  for page in r_pages do
    page.rank = 1.0 / num_pages
    -- TODO: Initialize your fields if you need
    page.num_out_edges = 0
    page.delta = 0
    page.pr_diff = 0
    page.old_rank = 0
  end

  var f = c.fopen(filename, "rb")
  skip_header(f)
  var page_ids : uint32[2]
  for link in r_links do
    regentlib.assert(read_ids(f, page_ids), "Less data that it should be")
    var src_page = dynamic_cast(ptr(Page, r_pages), page_ids[0])
    var dst_page = dynamic_cast(ptr(Page, r_pages), page_ids[1])
    -- TODO: Initialize the link with 'src_page' and 'dst_page'
    link.src_page = src_page
    link.dst_page = dst_page
  end

  for link in r_links do
     link.src_page.num_out_edges += 1
  end
  c.fclose(f)
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("Graph initialization took %.4f sec\n", (ts_stop - ts_start) * 1e-6)
end

-- TODO: Implement PageRank. You can use as many tasks as you want.

task update_delta(r_pages_src : region(Page),
                  damp    : double,
                  num_pages : uint64)
where
    reads writes(r_pages_src)
do
    for page in r_pages_src do
        page.delta = 0
        page.old_rank = page.rank
        if page.num_out_edges > 0 then
            page.delta = (page.rank * damp) / page.num_out_edges
        end
        page.rank = (1-damp) / num_pages
   end
end

task update_rank(r_pages : region(Page),
                 r_links : region(Link(r_pages)))
 
where
    reads(r_links),
    reads writes(r_pages)
do
    for link in r_links do
        link.dst_page.rank += link.src_page.delta
    end
end

task update_diff(r_pages : region(Page),
                 num_pages : uint64)
where
    reads(r_pages.rank, r_pages.old_rank),
    writes(r_pages.pr_diff)
do 
    for page in r_pages do
        var diff = page.rank - page.old_rank
        page.pr_diff = diff * diff
    end
end

task find_error(r_pages : region(Page))
where
    reads(r_pages.pr_diff)
do
    var error = 0.0
    for page in r_pages do
        error += page.pr_diff
    end
    return error
end

task dump_ranks(r_pages  : region(Page),
                filename : int8[512])
where
  reads(r_pages.rank)
do
  var f = c.fopen(filename, "w")
  for page in r_pages do c.fprintf(f, "%g\n", page.rank) end
  c.fclose(f)
end

task toplevel()
  var config : PageRankConfig
  config:initialize_from_command()
  c.printf("**********************************\n")
  c.printf("* PageRank                       *\n")
  c.printf("*                                *\n")
  c.printf("* Number of Pages  : %11lu *\n",  config.num_pages)
  c.printf("* Number of Links  : %11lu *\n",  config.num_links)
  c.printf("* Damping Factor   : %11.4f *\n", config.damp)
  c.printf("* Error Bound      : %11g *\n",   config.error_bound)
  c.printf("* Max # Iterations : %11u *\n",   config.max_iterations)
  c.printf("* # Parallel Tasks : %11u *\n",   config.parallelism)
  c.printf("**********************************\n")

  -- Create a region of pages
  var r_pages = region(ispace(ptr, config.num_pages), Page)
  -- TODO: Create a region of links.
  --       It is your choice how you allocate the elements in this region.
  var r_links = region(ispace(ptr, config.num_links), Link(wild))

  -- Initialize the page graph from a file
  initialize_graph(r_pages, r_links, config.damp, config.num_pages, config.input)
  var colors = ispace(int1d, config.parallelism)

  var error_indices = ispace(int1d, config.parallelism)
  var error_region = region(error_indices, error_field)
  fill(error_region.error, 0.0)

  --
  -- TODO: Create partitions for links and pages.
  --       You can use as many partitions as you want.
  -- partition pages(nodes) equally , source node
  var src_partition = partition(equal, r_pages, colors)

  -- partition links - source partition
  var link_partition = preimage(r_links, src_partition, r_links.src_page)

  -- get destination nodes
  var dst_partition = image(r_pages, link_partition, r_links.dst_page)


  var num_iterations = 0
  var converged = false
  __fence(__execution, __block) -- This blocks to make sure we only time the pagerank computation
  var ts_start = c.legion_get_current_time_in_micros()
  while not converged do
    num_iterations += 1
    ---- TODO: Launch the tasks that you implemented above.
    ----       (and of course remove the break statement here.)
    for c in src_partition.colors do
        update_delta (src_partition[c], config.damp, config.num_pages)
    end
    for c in link_partition.colors do
        update_rank (dst_partition[c], link_partition[c])
    end
    for c in src_partition.colors do
        update_diff (src_partition[c], config.num_pages)
    end

    for c in src_partition.colors do
       error_region[c].error =  find_error(src_partition[c])
    end

    var error = 0.0
    for err in error_region do
        error += err.error
    end
    if c.sqrt(error) <= config.error_bound then
        converged = true
    end
    if num_iterations >= config.max_iterations  then
        break
    end
  end
  __fence(__execution, __block) -- This blocks to make sure we only time the pagerank computation
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("PageRank converged after %d iterations in %.4f sec\n",
    num_iterations, (ts_stop - ts_start) * 1e-6)

  if config.dump_output then dump_ranks(r_pages, config.output) end
end

regentlib.start(toplevel)
