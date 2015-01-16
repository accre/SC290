/* Segmentation Fault demo */

#include<stdio.h>

main()
{
    int a;
    int *b;
    printf("\nBad things are about to happen!\n\n");
    b = 0;
    a = *b;
}

