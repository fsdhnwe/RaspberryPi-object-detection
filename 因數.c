#include <stdio.h>

int main(){
int a, b, c;
scanf("%d%d%d", &a, &b, &c);
if(a%c==0 && b%c==0) printf("Yes");
else printf("No");
return 0;
}
