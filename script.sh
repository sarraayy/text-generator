#!/bin/bash
# Author: Saray Alvarado (obj163)
# Purpose: practice basic linux shell commands



# creating directory
mkdir -p big-directory

# creating another directory
mkdir -p big-directory/small-directory

# creating file in small directory
touch big-directory/small-directory/file-1.txt

# writing name in file using echo
echo "Saray" > big-directory/small-directory/file-1.txt

# viewing content
cat big-directory/small-directory/file-1.txt

# append the numbers from 1 to 20
for i in $(seq 1 20)
do
    echo $i >> big-directory/small-directory/file-1.txt
done

#viewing content of file-1.txt
cat big-directory/small-directory/file-1.txt

# counting words in file-1.txt and saving value in file-2.txt without saving path
wc -w < big-directory/small-directory/file-1.txt >> big-directory/small-directory/file-2.txt

#viewing content of file-1 & -2
cat big-directory/small-directory/file-1.txt
cat big-directory/small-directory/file-2.txt

# copying file-1.txt in big-directory
cp big-directory/small-directory/file-1.txt big-directory/

# renaming file-1.txt in big-directory to file-3.txt
mv big-directory/file-1.txt big-directory/file-3.txt

# deleting small-directory
rm -r big-directory/small-directory

#executing commands ls -l and ls -a
ls -l big-directory
ls -a big-directory

# deleting big-directory
rm -r big-directory

