
export OMP_NUM_THREADS

ITE=$(seq 10) # nombre de mesures
  
THREADS=$(seq 2 2 24) # nombre de threads

PARAM="-n -i 100 -s 256 -a" # parametres commun à toutes les executions 

execute (){
EXE="../prog $* $PARAM"
OUTPUT="$(echo $EXE | tr -d ' ')"
for nb in $ITE; do for OMP_NUM_THREADS in $THREADS; do  echo -n "$OMP_NUM_THREADS " >> $OUTPUT ; $EXE 2>> $OUTPUT; done; done
}

# execute -v 3
# execute -v 4
# execute -v 5
# execute -v 6
execute -v 7



