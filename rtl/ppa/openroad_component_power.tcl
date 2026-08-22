if {[info exists ::env(PPA_THREADS)]} {
  set_thread_count $::env(PPA_THREADS)
}
set use_odb [expr {[info exists ::env(PPA_ODB)] && ($::env(PPA_ODB) ne "")}]
if {$use_odb} {
  read_db $::env(PPA_ODB)
} else {
  read_lef $::env(PPA_TECH_LEF)
  read_lef $::env(PPA_MACRO_LEF)
  if {[info exists ::env(PPA_EXTRA_LEFS)] && ($::env(PPA_EXTRA_LEFS) ne "")} {
    foreach lef $::env(PPA_EXTRA_LEFS) { read_lef $lef }
  }
}
read_liberty $::env(PPA_LIBERTY)
if {[info exists ::env(PPA_EXTRA_LIBS)] && ($::env(PPA_EXTRA_LIBS) ne "")} {
  foreach lib $::env(PPA_EXTRA_LIBS) { read_liberty $lib }
}
if {!$use_odb} {
  read_verilog $::env(PPA_NETLIST)
  link_design $::env(PPA_TOP)
  if {[info exists ::env(PPA_DEF)] && ($::env(PPA_DEF) ne "")} {
    read_def -floorplan_initialize $::env(PPA_DEF)
  }
}

if {$::env(PPA_HAS_CLOCK) == "1"} {
  create_clock -name clk -period $::env(PPA_CLOCK_PERIOD_NS) [get_ports clk]
  set_propagated_clock [all_clocks]
} else {
  create_clock -name activity_clock -period $::env(PPA_CLOCK_PERIOD_NS)
}
set_input_transition 0.05 [all_inputs]
set_load 0.01 [all_outputs]
set_wire_rc -signal -layer metal3 -clock -layer metal6
if {[info exists ::env(PPA_SPEF)] && ($::env(PPA_SPEF) ne "")} {
  read_spef $::env(PPA_SPEF)
}
set vcd_scope component_activity
if {[info exists ::env(PPA_VCD_SCOPE)] && ($::env(PPA_VCD_SCOPE) ne "")} {
  set vcd_scope $::env(PPA_VCD_SCOPE)
}
read_vcd -scope $vcd_scope $::env(PPA_VCD)

puts "PPA_TIMING_BEGIN"
set power_only [expr {[info exists ::env(PPA_POWER_ONLY)] && ($::env(PPA_POWER_ONLY) == "1")}]
if {$power_only} {
  puts "POWER_ONLY_TIMING_SKIPPED"
} elseif {$::env(PPA_HAS_CLOCK) == "1"} {
  report_checks -path_delay max -fields {slew cap} -digits 6
} else {
  puts "COMBINATIONAL_NO_CLOCK"
}
puts "PPA_TIMING_END"
puts "PPA_POWER_BEGIN"
puts "MLX_COMPONENT_WORKLOAD_POWER_BEGIN"
report_power
puts "MLX_COMPONENT_WORKLOAD_POWER_END"
puts "PPA_POWER_END"
