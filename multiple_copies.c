// multiple_copies.c
// Name: Saray Alvarado
// abc123: obj163


#include <stdio.h> //printf
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h> //open
#include <unistd.h> //read, write, close



int main(int argc, char *argv[]) {

    if (argc != 4) {
        printf("Usage: multiple_copies source_file destination_file1 destination_file2\n");
    return 1;    
}

const char *source_file = argv[1];
const char *dest_file1 = argv[2];
const char *dest_file2 = argv[3];


// open source file
int src_fd = open(source_file, O_RDONLY);
if (src_fd == -1) {
    printf("Open Error!: %s\n", source_file);
    return 1;
}else {
    printf("File `%s` open successfully\n", source_file);
}

// open first destination file
int dest_fd1 = open(dest_file1, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IROTH);
if (dest_fd1 == -1){
    printf("Error opening destination file: %s\n", dest_file1);
    close(src_fd);
    return 1;
}else{
    printf("File `%s` opened successfully\n", dest_file1);
}

// open second destination
int dest_fd2 = open(dest_file2, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IROTH);
if (dest_fd2 == -1) {
    printf("Error opening destination file: %s\n", dest_file2);
    close(src_fd);
    close(dest_fd1);
    return 1;
}else {
    printf("File `%s` opened successfully\n", dest_file2);
}
char buffer[1024];
ssize_t bytes_read;

// read source file and write both destinations
while ((bytes_read = read(src_fd, buffer, sizeof(buffer))) > 0) {
    if (write(dest_fd1, buffer, bytes_read) != bytes_read) {
        printf("Error writing to %s\n", dest_file1);
        break;
    }
    if (write(dest_fd2, buffer, bytes_read) != bytes_read) {
        printf("Error writing to %s\n", dest_file2);
        break;
    }
}
if (bytes_read == -1) {
    printf("Error reading from %s\n", source_file);
}else {
    printf("File copied successfully to `%s` and `%s`\n", dest_file1, dest_file2);
}

// close all files
close(src_fd);
close(dest_fd1);
close(dest_fd2);

return 0;
}

