read_db $::env(PPA_PRECHECK_ODB)
puts "MLX_CHANNEL_PRECHECK_DIAGNOSE checkpoint=$::env(PPA_PRECHECK_ODB)"
check_placement -verbose
