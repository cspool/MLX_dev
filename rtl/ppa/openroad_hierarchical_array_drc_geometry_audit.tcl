foreach required {PPA_AUDIT_ODB PPA_AUDIT_DRC PPA_AUDIT_MACRO_LEF} {
  if {![info exists ::env($required)] || $::env($required) eq ""} {
    error "missing required environment variable $required"
  }
}

set_thread_count 1
read_db $::env(PPA_AUDIT_ODB)
set block [ord::get_db_block]
set dbu [$block getDbUnitsPerMicron]

# Index the compact integration view's actual OBS rectangles by routing layer.
set lef_channel [open $::env(PPA_AUDIT_MACRO_LEF) r]
set in_obstructions 0
set obstruction_layer ""
set obstructions [dict create]
while {[gets $lef_channel line] >= 0} {
  if {[string match "  OBS*" $line]} {
    set in_obstructions 1
    continue
  }
  if {!$in_obstructions} {
    continue
  }
  if {[regexp {^\s*LAYER\s+(\S+)\s*;} $line -> layer]} {
    set obstruction_layer $layer
    continue
  }
  if {[regexp {^\s*RECT\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)\s*;} $line -> x0 y0 x1 y1]} {
    if {$obstruction_layer ne ""} {
      dict lappend obstructions $obstruction_layer [list $x0 $y0 $x1 $y1]
    }
  }
}
close $lef_channel

set drc_channel [open $::env(PPA_AUDIT_DRC) r]
set drc_text [read $drc_channel]
close $drc_channel
set marker_pattern {violation type: ([^\n]+)\n[ \t]*srcs: ([^\n]+)\n[ \t]*bbox = \(([-0-9.]+), ([-0-9.]+)\) - \(([-0-9.]+), ([-0-9.]+)\) on Layer ([^\n]+)}
set matches [regexp -all -inline -- $marker_pattern $drc_text]

set marker_count 0
set macro_marker_count 0
set net_marker_count 0
set uncovered_macro_markers 0
set affected_tiles [dict create]
for {set offset 0} {$offset < [llength $matches]} {incr offset 8} {
  incr marker_count
  set violation_type [string trim [lindex $matches [expr {$offset + 1}]]]
  set sources [string trim [lindex $matches [expr {$offset + 2}]]]
  set global_x0 [expr {round(double([lindex $matches [expr {$offset + 3}]]) * $dbu)}]
  set global_y0 [expr {round(double([lindex $matches [expr {$offset + 4}]]) * $dbu)}]
  set global_x1 [expr {round(double([lindex $matches [expr {$offset + 5}]]) * $dbu)}]
  set global_y1 [expr {round(double([lindex $matches [expr {$offset + 6}]]) * $dbu)}]
  set layer [string trim [lindex $matches [expr {$offset + 7}]]]
  set instance_name ""
  set net_names {}
  foreach source [split $sources] {
    if {[string match {inst:*} $source]} {
      set instance_name [string range $source 5 end]
    } elseif {[string match {net:*} $source]} {
      lappend net_names [string range $source 4 end]
    }
  }

  if {$instance_name eq ""} {
    incr net_marker_count
    puts "MLX_DRC_GEOMETRY_MARKER id=$marker_count class=top_net_intersection type=$violation_type layer=$layer nets=[join $net_names ,]"
    continue
  }

  incr macro_marker_count
  set inst [$block findInst $instance_name]
  if {$inst == "NULL"} {
    error "DRC instance not found in ODB: $instance_name"
  }
  set orient [$inst getOrient]
  if {$orient ne "R0"} {
    error "unsupported non-R0 audit instance $instance_name orientation=$orient"
  }
  if {[regexp {GENERATE_TILES[^0-9]*([0-9]+)} $instance_name -> tile_index]} {
    dict set affected_tiles $tile_index 1
  }
  set box [$inst getBBox]
  set local_x0 [expr {double($global_x0 - [$box xMin]) / $dbu}]
  set local_y0 [expr {double($global_y0 - [$box yMin]) / $dbu}]
  set local_x1 [expr {double($global_x1 - [$box xMin]) / $dbu}]
  set local_y1 [expr {double($global_y1 - [$box yMin]) / $dbu}]
  set covered 0
  set covering_rect ""
  if {[dict exists $obstructions $layer]} {
    foreach rectangle [dict get $obstructions $layer] {
      lassign $rectangle rect_x0 rect_y0 rect_x1 rect_y1
      if {$rect_x1 >= $local_x0 && $rect_x0 <= $local_x1
          && $rect_y1 >= $local_y0 && $rect_y0 <= $local_y1} {
        set covered 1
        set covering_rect [join $rectangle ,]
        break
      }
    }
  }
  if {!$covered} {
    incr uncovered_macro_markers
  }
  puts [format "MLX_DRC_GEOMETRY_MARKER id=%d class=macro_obstruction_intersection type=%s layer=%s instance=%s nets=%s local_bbox_um=%.6f,%.6f,%.6f,%.6f obstruction_covered=%d covering_rect_um=%s" \
    $marker_count $violation_type $layer $instance_name [join $net_names ,] \
    $local_x0 $local_y0 $local_x1 $local_y1 $covered $covering_rect]
}

set affected_tile_indices [lsort -integer [dict keys $affected_tiles]]
puts "MLX_DRC_GEOMETRY_SUMMARY markers=$marker_count macro_obstruction_intersections=$macro_marker_count top_net_intersections=$net_marker_count uncovered_macro_markers=$uncovered_macro_markers affected_tiles=[join $affected_tile_indices ,] dbu_per_micron=$dbu"
