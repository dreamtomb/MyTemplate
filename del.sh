isort .
ls -lR | grep "/__pycache__" | awk -F ":" '{print "rm -rf " $1}' | sh
# ls -ltR | grep "2022-11-18_16-11-18" | grep -v "drwxr" | awk -F ":" '{print "rm -rf " $1}' | sh
