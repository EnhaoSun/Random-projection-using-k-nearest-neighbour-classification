#!/bin/sh

if [ $1 == "q1" ]
then
	python3 source/q1.py $2 $3
elif [ $1 == "q2" ]
then
	python3 source/q2.py $2 $3 $4 $5 $6
fi
