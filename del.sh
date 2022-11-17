ls -lR | grep "/2022" | awk -F ":" '{print "rm -rf " $1}' | sh
ls -lR | grep "/__pycache__" | awk -F ":" '{print "rm -rf " $1}' | sh
