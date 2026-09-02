foreach required {PPA_TARGETED_INPUT_ODB PPA_TARGETED_NETS PPA_TARGETED_MODE} {
  if {![info exists ::env($required)] || $::env($required) eq ""} {
    error "missing required environment variable $required"
  }
}

set mode $::env(PPA_TARGETED_MODE)
if {$mode ni {audit apply}} {
  error "PPA_TARGETED_MODE must be audit or apply"
}
if {$mode eq "apply"} {
  if {![info exists ::env(PPA_TARGETED_REROUTE_ENABLE)]
      || $::env(PPA_TARGETED_REROUTE_ENABLE) ne "1"} {
    error "apply mode requires PPA_TARGETED_REROUTE_ENABLE=1"
  }
  if {![info exists ::env(PPA_TARGETED_OUTPUT_ODB)]
      || $::env(PPA_TARGETED_OUTPUT_ODB) eq ""} {
    error "apply mode requires PPA_TARGETED_OUTPUT_ODB"
  }
  set input_path [file normalize $::env(PPA_TARGETED_INPUT_ODB)]
  set output_path [file normalize $::env(PPA_TARGETED_OUTPUT_ODB)]
  if {$input_path eq $output_path} {
    error "targeted reroute seed refuses to overwrite its input ODB"
  }
  if {[file exists $output_path]} {
    error "targeted reroute seed output already exists: $output_path"
  }
}

set_thread_count 1
read_db $::env(PPA_TARGETED_INPUT_ODB)
set block [ord::get_db_block]
set selected_nets [split $::env(PPA_TARGETED_NETS) |]
set selected_wires {}
set total_shapes 0
set total_iterms 0
set total_bterms 0
foreach net_name $selected_nets {
  if {$net_name eq ""} {
    error "empty net name in PPA_TARGETED_NETS"
  }
  set net [$block findNet $net_name]
  if {$net == "NULL"} {
    error "targeted reroute net not found: $net_name"
  }
  set wire [$net getWire]
  if {$wire == "NULL"} {
    error "targeted reroute net has no routed dbWire: $net_name"
  }
  set shapes [$wire count]
  set iterms [$net getITermCount]
  set bterms [$net getBTermCount]
  incr total_shapes $shapes
  incr total_iterms $iterms
  incr total_bterms $bterms
  lappend selected_wires [list $net_name $net $wire]
  puts "MLX_TARGETED_REROUTE_NET name=$net_name wire_shapes=$shapes iterms=$iterms bterms=$bterms"
}

if {$mode eq "apply"} {
  foreach selected $selected_wires {
    lassign $selected net_name net wire
    odb::dbWire_destroy $wire
    if {[$net getWire] != "NULL"} {
      error "dbWire destroy verification failed for $net_name"
    }
  }
  write_db $::env(PPA_TARGETED_OUTPUT_ODB)
}

puts "MLX_TARGETED_REROUTE_SUMMARY mode=$mode selected_nets=[llength $selected_nets] wire_shapes=$total_shapes iterms=$total_iterms bterms=$total_bterms input=$::env(PPA_TARGETED_INPUT_ODB) output=[expr {$mode eq {apply} ? $::env(PPA_TARGETED_OUTPUT_ODB) : {none}}]"
