import "regent"

-- Helper module to handle command line arguments
local PageRankConfig = require("pagerank_config")

local c = regentlib.c

fspace Page {
  rank               : double,
  num_outgoing_edges : double,
  old_rank           : double,
  delta              : double;
}

fspace Link(r : region(Page)) {
    src_page : ptr(Page, r),
    dest_page : ptr(Page, r)
}

fspace error_field {
    error : double
}

terra skip_header(f : &c.FILE)
  var x : uint64, y : uint64
  c.fscanf(f, "%llu\n%llu\n", &x, &y)
end

terra read_ids(f : &c.FILE, page_ids : &uint32)
  return c.fscanf(f, "%d %d\n", &page_ids[0], &page_ids[1]) == 2
end

task initialize_graph(r_pages   : region(Page),
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
    page.delta= 0
    page.old_rank= 0
    page.num_outgoing_edges = 0
  end

  var f = c.fopen(filename, "rb")
  skip_header(f)
  var page_ids : uint32[2]
  for link in r_links do
    regentlib.assert(read_ids(f, page_ids), "Less data that it should be")
    var src_page = dynamic_cast(ptr(Page, r_pages), page_ids[0])
    var dst_page = dynamic_cast(ptr(Page, r_pages), page_ids[1])
    link.src_page = src_page
    link.dest_page = dst_page
  end
  c.fclose(f)
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("Graph initialization took %.4f sec\n", (ts_stop - ts_start) * 1e-6)
end


task calculate_outgoing_nodes( r_pages: region(Page),
                      r_links   : region(Link(r_pages)))
where 
  reads (r_links),
  reduces+ (r_pages.num_outgoing_edges)
do
  for link in r_links do
    link.src_page.num_outgoing_edges += 1
  end
end

task calculate_delta(r_src_pages: region(Page), 
                      damp      : double,
                      num_pages : uint64)
where 
  reads (r_src_pages.num_outgoing_edges),
  writes (r_src_pages.delta),
  reads writes (r_src_pages.rank),
  writes (r_src_pages.old_rank) 
do
-- Need to update delta only for src pages
  for page in r_src_pages do
    -- Calculate the contribution of this node to the output 
    page.delta = 0
    page.old_rank = page.rank
    if page.num_outgoing_edges > 0 then
      page.delta = damp*(page.rank/page.num_outgoing_edges)
      -- c.printf("delta = %f\n", page.delta)
    else        
      page.delta = 0
    end
    page.rank = (1-damp)/num_pages
  end
end 

task init_rank(r_dest_pages: region(Page),
                      damp      : double,
                      num_pages : uint64)
where 
  reads writes (r_dest_pages.rank),
  reads writes (r_dest_pages.old_rank) -- TODO: convert it back to writes
do
-- Need to update rank only for dest pages 
  for page in r_dest_pages do
    page.old_rank = page.rank
    -- TODO: might need to update rank in a different kernel -- otherwise src page might end up reading newer rank? 
    -- Alternatively, Is there a way to design partitions to make them disjoin to avoid this issue? 
    page.rank = (1-damp)/num_pages
  end
end


task update_rank(r_pages: region(Page),
                      r_links   : region(Link(r_pages)))
where 
  reads writes (r_pages.rank),
  reads (r_pages.delta, r_pages.old_rank),
  reads (r_links)
do
  for link in r_links do
    link.dest_page.rank += link.src_page.delta
  end
end

task update_diff(r_pages: region(Page))
where 
  reads (r_pages.old_rank, r_pages.rank)
do
  var l2_norm = 0.0
  for page in r_pages do
    l2_norm += (page.old_rank - page.rank)*(page.old_rank - page.rank)
  end
  return l2_norm
end

-- TODO: use terra

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

  -- Create a region of links.
  -- TODO: what's the differece between wild and this? Perf penalthy?
   var r_links = region(ispace(ptr, config.num_links), Link(wild)) 
   -- var r_links = region(ispace(ptr, config.num_links), Link(r_pages)) 

  -- Initialize the page graph from a file
  initialize_graph(r_pages, r_links, config.damp, config.num_pages, config.input)
  calculate_outgoing_nodes(r_pages, r_links)

  var num_iterations = 0
  var converged = false
  __fence(__execution, __block) -- This blocks to make sure we only time the pagerank computation
  var ts_start = c.legion_get_current_time_in_micros()

  var num_pieces = config.parallelism
  var links_part = partition(equal, r_links, ispace(int1d, num_pieces))

  -- Can make it disjoint maybe to reduce repeated computations?
  var pages_src_part = image(disjoint, complete, r_pages, links_part, r_links.src_page)
  var pages_dest_part = image(r_pages, links_part, r_links.dest_page)


-- TODO: this seems like a hack : how to actually create partitions ? 
  -- var pages_part_disjoint = image(disjoint, complete, r_pages, links_part, r_links.dest_page)
  var pages_part_disjoint = partition(equal, r_pages, ispace(int1d, num_pieces))
-- TODO: is the number of src and dest partitions same? -- and is it same as # of link partitions  --- or do I need to use union?

  var pages_union = (pages_src_part | pages_dest_part)


 --   for part in links_part.colors do
 --       c.printf("========\n")
 --       for link in links_part[part] do
 --           c.printf("link : %d -> %d\n", link.src_page, link.dest_page)
 --       end
 --   end

 --   for part in pages_part_disjoint.colors do
 --       c.printf("========\n")
 --       for page in pages_part_disjoint[part] do
 --           c.printf("dest: %d\n", page)
 --       end
 --   end

 var error_indices = ispace(int1d, config.parallelism)
  var error_region = region(error_indices, error_field)
  fill(error_region.error, 0.0)

  while not converged do
    num_iterations += 1
    var l2_norm = 0.0
    
    for part in pages_part_disjoint.colors do
      calculate_delta(pages_part_disjoint[part], config.damp, config.num_pages)
    end

    -- If a dest node is present in 2 parts, it should be processed twice
    for part in links_part.colors do
      update_rank(pages_dest_part[part], links_part[part])
    end
    -- We don't want a diff to be counted twice hence disjoint
    for part in pages_part_disjoint.colors do
      error_region[part].error = update_diff(pages_part_disjoint[part])
      --c.printf("l2_norm = %f\n", l2_norm)
    end

    for err in error_region do
        l2_norm += err.error
    end
    if(c.sqrt(l2_norm) <= config.error_bound) then 
      converged = true 
    end
    if(num_iterations >= config.max_iterations) then
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
