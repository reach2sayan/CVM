#!/bin/bash

for dir in `ls -d sqs_lev\=*`
do
  cd $dir
  elems=`echo $dir | egrep -o '(Mo|Ta|V|W|Nb)'`
  latstr=''
  for elem in $elems
  do
    latstr+="$elem,"
  done
  sed -i "s/Mo,V,W,Nb,Ta/${latstr::-1}/g" lat.in
  cd ..
done
