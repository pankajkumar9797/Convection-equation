#set style line 1 lw 5 lt 2 ps 2
set style line 1 lt 1 lc rgb "#A00000" lw 2 pt 7 ps 1.5
set style line 2 lt 1 lc rgb "#00A000" lw 2 pt 11 ps 1.5
plot "data2.dat" using 1:2 title '3500 cells'with linespoints, "data2.dat" using 1:3 title '50000 cells'with linespoints
set autoscale
set xrange [0.0:1.0]
set xlabel "time" 
set ylabel "L2_error "
set terminal png font arial 22 size 1024,768
set output "L2_error_time.png"
replot

