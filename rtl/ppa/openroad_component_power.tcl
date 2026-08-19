read_lef $::env(PPA_TECH_LEF)
read_lef $::env(PPA_MACRO_LEF)
read_liberty $::env(PPA_LIBERTY)
read_verilog $::env(PPA_NETLIST)
link_design $::env(PPA_TOP)

if {$::env(PPA_HAS_CLOCK) == "1"} {
  create_clock -name clk -period $::env(PPA_CLOCK_PERIOD_NS) [get_ports clk]
} else {
  create_clock -name virtual_clk -period $::env(PPA_CLOCK_PERIOD_NS)
}
set_input_transition 0.05 [all_inputs]
set_load 0.01 [all_outputs]
read_vcd -scope $::env(PPA_VCD_SCOPE) $::env(PPA_VCD)

puts "PPA_TIMING_BEGIN"
if {$::env(PPA_HAS_CLOCK) == "1"} {
  report_checks -path_delay max -fields {slew cap} -digits 6
} else {
  puts "COMBINATIONAL_NO_CLOCK"
}
puts "PPA_TIMING_END"
puts "PPA_POWER_BEGIN"
report_power
puts "PPA_POWER_END"
